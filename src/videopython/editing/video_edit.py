from __future__ import annotations

import copy
import importlib
import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Mapping, Sequence, Union, get_args, get_origin, get_type_hints

import numpy as np

from videopython.base.audio import Audio
from videopython.base.effects import AudioEffect, Effect, Fade
from videopython.base.registry import (
    OperationCategory,
    OperationSpec,
    ParamSpec,
    get_operation_spec,
    get_operation_specs,
)
from videopython.base.streaming import EffectScheduleEntry, StreamingSegmentPlan, concat_files, stream_segment
from videopython.base.transforms import Transformation
from videopython.base.video import ALLOWED_VIDEO_FORMATS, ALLOWED_VIDEO_PRESETS, Video, VideoMetadata

__all__ = [
    "SegmentConfig",
    "VideoEdit",
]


@dataclass(frozen=True)
class _StepRecord:
    """Canonical step data + live operation instance.

    `args` and `apply_args` are snapshots of parsed input (deep copied) and are the
    source of truth for serialization.
    """

    op_id: str
    args: dict[str, Any]
    apply_args: dict[str, Any]
    operation: Transformation | Effect

    @classmethod
    def create(
        cls,
        op_id: str,
        args: Mapping[str, Any] | None,
        apply_args: Mapping[str, Any] | None,
        operation: Transformation | Effect,
    ) -> _StepRecord:
        args_copy = copy.deepcopy(dict(args or {}))
        apply_args_copy = copy.deepcopy(dict(apply_args or {}))
        _validate_step_record_apply_args_contract(apply_args_copy)
        return cls(
            op_id=op_id,
            args=args_copy,
            apply_args=apply_args_copy,
            operation=operation,
        )


@dataclass
class SegmentConfig:
    """Configuration for a single video segment in an editing plan."""

    source_video: Path
    start_second: float
    end_second: float
    transform_records: tuple[_StepRecord, ...] = field(default_factory=tuple)
    effect_records: tuple[_StepRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.transform_records = tuple(self.transform_records)
        self.effect_records = tuple(self.effect_records)
        for record in self.transform_records:
            if not isinstance(record.operation, Transformation):
                raise TypeError(
                    "SegmentConfig.transform_records must contain "
                    f"Transformation operations, got {type(record.operation)}"
                )
        for record in self.effect_records:
            if not isinstance(record.operation, Effect):
                raise TypeError(
                    f"SegmentConfig.effect_records must contain Effect operations, got {type(record.operation)}"
                )

    def load_segment(
        self,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Video:
        """Load the raw segment from disk (cut only, no transforms or effects).

        Optional fps/width/height are applied during decoding via ffmpeg filters.
        """
        return Video.from_path(
            str(self.source_video),
            start_second=self.start_second,
            end_second=self.end_second,
            fps=fps,
            width=width,
            height=height,
        )

    def apply_operations(self, video: Video, context: dict[str, Any] | None = None) -> Video:
        """Apply per-segment transforms and effects to a loaded video."""
        for record in self.transform_records:
            video = _apply_transform_with_context(record, video, context)
        for record in self.effect_records:
            if not isinstance(record.operation, Effect):
                raise TypeError(
                    f"SegmentConfig.effect_records must contain Effect operations, got {type(record.operation)}"
                )
            video = record.operation.apply(
                video,
                start=_coerce_optional_number(record.apply_args.get("start"), "start"),
                stop=_coerce_optional_number(record.apply_args.get("stop"), "stop"),
            )
        return video

    def process_segment(self, context: dict[str, Any] | None = None) -> Video:
        """Load the segment and apply transforms then effects."""
        return self.apply_operations(self.load_segment(), context)


class VideoEdit:
    """Represents a complete multi-segment video editing plan."""

    def __init__(
        self,
        segments: Sequence[SegmentConfig],
        post_transform_records: Sequence[_StepRecord] | None = None,
        post_effect_records: Sequence[_StepRecord] | None = None,
        match_to_lowest_fps: bool = True,
        match_to_lowest_resolution: bool = True,
    ):
        if not segments:
            raise ValueError("VideoEdit requires at least one segment")
        self.segments: tuple[SegmentConfig, ...] = tuple(segments)
        self.post_transform_records: tuple[_StepRecord, ...] = tuple(post_transform_records or ())
        self.post_effect_records: tuple[_StepRecord, ...] = tuple(post_effect_records or ())
        self.match_to_lowest_fps: bool = match_to_lowest_fps
        self.match_to_lowest_resolution: bool = match_to_lowest_resolution

        for record in self.post_transform_records:
            if not isinstance(record.operation, Transformation):
                raise TypeError(
                    "VideoEdit.post_transform_records must contain "
                    f"Transformation operations, got {type(record.operation)}"
                )
        for record in self.post_effect_records:
            if not isinstance(record.operation, Effect):
                raise TypeError(
                    f"VideoEdit.post_effect_records must contain Effect operations, got {type(record.operation)}"
                )

    @classmethod
    def from_json(cls, text: str) -> VideoEdit:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid VideoEdit JSON: {e.msg} at line {e.lineno} column {e.colno}") from e
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoEdit:
        if not isinstance(data, dict):
            raise ValueError("VideoEdit plan must be a JSON object")

        segments_data = data.get("segments")
        if segments_data is None:
            raise ValueError("VideoEdit plan is missing required key 'segments'")
        if not isinstance(segments_data, list):
            raise ValueError("VideoEdit plan 'segments' must be a list")
        if not segments_data:
            raise ValueError("VideoEdit plan 'segments' must not be empty")

        post_transforms_data = data.get("post_transforms", [])
        post_effects_data = data.get("post_effects", [])
        if not isinstance(post_transforms_data, list):
            raise ValueError("VideoEdit plan 'post_transforms' must be a list")
        if not isinstance(post_effects_data, list):
            raise ValueError("VideoEdit plan 'post_effects' must be a list")

        segments: list[SegmentConfig] = []
        for i, segment_data in enumerate(segments_data):
            location = f"segments[{i}]"
            segments.append(_parse_segment(segment_data, location))

        post_transform_records = [
            _parse_transform_step(step, f"post_transforms[{i}]") for i, step in enumerate(post_transforms_data)
        ]
        post_effect_records = [
            _parse_effect_step(step, f"post_effects[{i}]") for i, step in enumerate(post_effects_data)
        ]

        return cls(
            segments=segments,
            post_transform_records=post_transform_records,
            post_effect_records=post_effect_records,
            match_to_lowest_fps=data.get("match_to_lowest_fps", True),
            match_to_lowest_resolution=data.get("match_to_lowest_resolution", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical JSON-compatible dict.

        Serialization uses `_StepRecord` snapshots as the source of truth. Mutating
        live operation objects after parsing/construction does not affect output.
        """
        result: dict[str, Any] = {
            "segments": [self._segment_to_dict(segment) for segment in self.segments],
            "post_transforms": [_step_to_dict(record, include_apply=False) for record in self.post_transform_records],
            "post_effects": [_step_to_dict(record, include_apply=True) for record in self.post_effect_records],
        }
        if not self.match_to_lowest_fps:
            result["match_to_lowest_fps"] = False
        if not self.match_to_lowest_resolution:
            result["match_to_lowest_resolution"] = False
        return result

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Return a JSON Schema for `VideoEdit` plans."""
        transform_specs = _videoedit_supported_specs_for_category(OperationCategory.TRANSFORMATION)
        effect_specs = _videoedit_supported_specs_for_category(OperationCategory.EFFECT)

        transform_step_schemas = [
            _videoedit_step_schema_from_spec(spec, include_apply=False) for spec in transform_specs
        ]
        effect_step_schemas = [_videoedit_step_schema_from_spec(spec, include_apply=True) for spec in effect_specs]

        segment_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source video path."},
                "start": {"type": "number", "description": "Segment start time in seconds."},
                "end": {"type": "number", "description": "Segment end time in seconds."},
                "transforms": {
                    "type": "array",
                    "items": {"oneOf": transform_step_schemas},
                },
                "effects": {
                    "type": "array",
                    "items": {"oneOf": effect_step_schemas},
                },
            },
            "required": ["source", "start", "end"],
            "additionalProperties": False,
        }

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": segment_schema,
                    "minItems": 1,
                },
                "post_transforms": {
                    "type": "array",
                    "items": {"oneOf": transform_step_schemas},
                },
                "post_effects": {
                    "type": "array",
                    "items": {"oneOf": effect_step_schemas},
                },
            },
            "required": ["segments"],
        }

    def run(self, context: dict[str, Any] | None = None) -> Video:
        """Execute the editing plan and return the final video.

        Args:
            context: Optional side-channel data for context-dependent operations.
                Operations whose registry spec has a ``requires_transcript`` tag
                receive ``context["transcription"]`` as a keyword argument.
        """
        video = self._assemble_segments(context)
        for record in self.post_transform_records:
            video = _apply_transform_with_context(record, video, context)
        for record in self.post_effect_records:
            if not isinstance(record.operation, Effect):
                raise TypeError(
                    f"VideoEdit.post_effect_records must contain Effect operations, got {type(record.operation)}"
                )
            video = record.operation.apply(
                video,
                start=_coerce_optional_number(record.apply_args.get("start"), "start"),
                stop=_coerce_optional_number(record.apply_args.get("stop"), "stop"),
            )
        return video

    def run_to_file(
        self,
        output_path: str | Path,
        format: ALLOWED_VIDEO_FORMATS = "mp4",
        preset: ALLOWED_VIDEO_PRESETS = "medium",
        crf: int = 23,
        context: dict[str, Any] | None = None,
    ) -> Path:
        """Execute the editing plan, streaming directly to a file.

        Memory usage is O(1) w.r.t. video length for fully streamable pipelines.
        Falls back to eager mode (run + save) for non-streamable operations.

        Args:
            output_path: Destination file path.
            format: Output container format.
            preset: x264 encoding preset.
            crf: Constant rate factor (quality).
            context: Optional side-channel data for context-dependent operations.

        Returns:
            Path to the output file.
        """
        output_path = Path(output_path).with_suffix(f".{format}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Fall back to eager if post-transforms or non-streamable post-effects exist
        if self.post_transform_records:
            return self._run_to_file_eager(output_path, format, preset, crf, context)

        for record in self.post_effect_records:
            if not isinstance(record.operation, Effect) or not record.operation.supports_streaming:
                return self._run_to_file_eager(output_path, format, preset, crf, context)

        # Compute matching targets
        target_fps, target_w, target_h = self._compute_matching_targets()

        # Analyze each segment
        plans: list[StreamingSegmentPlan | None] = []
        for segment in self.segments:
            plan = self._build_streaming_plan(segment, target_fps, target_w, target_h, context)
            plans.append(plan)

        # If any segment can't stream, fall back entirely
        if any(p is None for p in plans):
            return self._run_to_file_eager(output_path, format, preset, crf, context)

        streaming_plans: list[StreamingSegmentPlan] = plans  # type: ignore[assignment]

        # Fold post-effects into plans (they apply to the full assembled video)
        # For simplicity, fold into single-segment plans; multi-segment post-effects
        # require a second pass which we skip for now
        if self.post_effect_records and len(streaming_plans) > 1:
            return self._run_to_file_eager(output_path, format, preset, crf, context)

        if self.post_effect_records and len(streaming_plans) == 1:
            plan = streaming_plans[0]
            total_frames = round((plan.end_second - plan.start_second) * plan.output_fps)
            for record in self.post_effect_records:
                start_s = _coerce_optional_number(record.apply_args.get("start"), "start")
                stop_s = _coerce_optional_number(record.apply_args.get("stop"), "stop")
                start_f = round(start_s * plan.output_fps) if start_s is not None else 0
                end_f = round(stop_s * plan.output_fps) if stop_s is not None else total_frames
                assert isinstance(record.operation, Effect)
                plan.effect_schedule.append(EffectScheduleEntry(record.operation, start_f, end_f))

        import tempfile

        if len(streaming_plans) == 1:
            plan = streaming_plans[0]
            audio = self._load_segment_audio(self.segments[0], plan, context)
            return stream_segment(plan, output_path, audio=audio, format=format, preset=preset, crf=crf)
        else:
            # Multi-segment: stream each to temp, then concat
            temp_files: list[Path] = []
            try:
                for segment, plan in zip(self.segments, streaming_plans):
                    temp = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
                    temp.close()
                    audio = self._load_segment_audio(segment, plan, context)
                    stream_segment(plan, Path(temp.name), audio=audio, format=format, preset=preset, crf=crf)
                    temp_files.append(Path(temp.name))
                return concat_files(temp_files, output_path)
            finally:
                for f in temp_files:
                    f.unlink(missing_ok=True)

    def _run_to_file_eager(
        self,
        output_path: Path,
        format: ALLOWED_VIDEO_FORMATS,
        preset: ALLOWED_VIDEO_PRESETS,
        crf: int,
        context: dict[str, Any] | None,
    ) -> Path:
        """Fallback: run eagerly and save."""
        video = self.run(context=context)
        return video.save(output_path, format=format, preset=preset, crf=crf)

    def _compute_matching_targets(self) -> tuple[float | None, int | None, int | None]:
        """Compute fps/width/height matching targets across segments."""
        target_fps, target_w, target_h = None, None, None
        if len(self.segments) > 1 and (self.match_to_lowest_fps or self.match_to_lowest_resolution):
            source_metas = [VideoMetadata.from_path(str(seg.source_video)) for seg in self.segments]
            if self.match_to_lowest_fps:
                target_fps = min(m.fps for m in source_metas)
            if self.match_to_lowest_resolution:
                target_w = min(m.width for m in source_metas)
                target_h = min(m.height for m in source_metas)
        return target_fps, target_w, target_h

    def _build_streaming_plan(
        self,
        segment: SegmentConfig,
        target_fps: float | None,
        target_w: int | None,
        target_h: int | None,
        context: dict[str, Any] | None,
    ) -> StreamingSegmentPlan | None:
        """Try to build a streaming plan for a segment. Returns None if not streamable."""
        source_meta = VideoMetadata.from_path(str(segment.source_video))
        vf_filters: list[str] = []

        # Start with matching targets (applied as decode filters)
        out_fps = target_fps or source_meta.fps
        out_w = target_w or source_meta.width
        out_h = target_h or source_meta.height

        if target_w and target_h and (target_w != source_meta.width or target_h != source_meta.height):
            vf_filters.append(f"scale={target_w}:{target_h}")
        if target_fps and target_fps != source_meta.fps:
            vf_filters.append(f"fps={target_fps}")

        # Compile transforms to ffmpeg filters
        for record in segment.transform_records:
            vf = _compile_transform_to_vf(record, out_w, out_h, out_fps)
            if vf is None:
                return None  # Non-streamable transform
            if vf.filter_expr:
                vf_filters.append(vf.filter_expr)
            out_w = vf.out_width
            out_h = vf.out_height
            out_fps = vf.out_fps

        # Check effects are streamable
        effect_schedule: list[EffectScheduleEntry] = []
        duration = segment.end_second - segment.start_second
        total_frames = round(duration * out_fps)

        for record in segment.effect_records:
            if not isinstance(record.operation, Effect):
                return None
            if not record.operation.supports_streaming:
                return None
            # Compute frame range
            start_s = _coerce_optional_number(record.apply_args.get("start"), "start")
            stop_s = _coerce_optional_number(record.apply_args.get("stop"), "stop")
            start_f = round(start_s * out_fps) if start_s is not None else 0
            end_f = round(stop_s * out_fps) if stop_s is not None else total_frames
            effect_schedule.append(EffectScheduleEntry(record.operation, start_f, end_f))

        return StreamingSegmentPlan(
            source_path=segment.source_video,
            start_second=segment.start_second,
            end_second=segment.end_second,
            output_fps=out_fps,
            output_width=out_w,
            output_height=out_h,
            vf_filters=vf_filters,
            effect_schedule=effect_schedule,
        )

    def _load_segment_audio(
        self,
        segment: SegmentConfig,
        plan: StreamingSegmentPlan,
        context: dict[str, Any] | None,
    ) -> Audio | None:
        """Load and process audio for a segment."""
        import warnings

        from videopython.base.audio import AudioLoadError

        try:
            audio = Audio.from_path(str(segment.source_video))
            audio = audio.slice(segment.start_second, segment.end_second)
        except (AudioLoadError, FileNotFoundError, Exception):
            duration = segment.end_second - segment.start_second
            warnings.warn(f"No audio found for `{segment.source_video}`, using silent track.")
            audio = Audio.create_silent(duration_seconds=round(duration, 2), stereo=True, sample_rate=44100)

        # Apply audio effects (AudioEffect subclasses + Fade audio component)
        for entry in plan.effect_schedule:
            effect = entry.effect
            start_s = entry.start_frame / plan.output_fps
            stop_s = entry.end_frame / plan.output_fps
            if isinstance(effect, AudioEffect):
                effect._apply_audio(audio, start_s, stop_s, plan.output_fps)
            elif isinstance(effect, Fade) and audio is not None and not audio.is_silent:
                # Apply Fade's audio portion
                _apply_fade_audio(effect, audio, start_s, stop_s)

        return audio

    def validate(self, context: dict[str, Any] | None = None) -> VideoMetadata:
        """Validate the editing plan without loading video data.

        Requires source video files to be present on disk (uses ``VideoMetadata.from_path``).
        For validation without file access, use :meth:`validate_with_metadata`.

        Args:
            context: Optional side-channel data for context-dependent operations.
                Operations whose registry spec has a ``requires_transcript`` tag
                use ``context["transcription"]`` for metadata prediction.
        """
        source_metas = [self._validate_source_meta(i, seg) for i, seg in enumerate(self.segments)]
        source_metas = self._match_metas(source_metas)
        segment_metas = [
            self._apply_segment_meta_ops(i, seg, meta, context)
            for i, (seg, meta) in enumerate(zip(self.segments, source_metas))
        ]
        return self._validate_assembled(segment_metas, context)

    def validate_with_metadata(
        self,
        source_metadata: VideoMetadata | dict[str, VideoMetadata],
        context: dict[str, Any] | None = None,
    ) -> VideoMetadata:
        """Validate the editing plan using pre-built metadata instead of loading from file.

        Same validation as validate() but accepts a VideoMetadata directly,
        avoiding the need for the source video file to be on disk.

        Args:
            source_metadata: VideoMetadata for the source video (duration, dimensions, fps).
                For multi-source plans, pass a dict mapping source paths to their metadata.
            context: Optional side-channel data for context-dependent operations.
                Operations whose registry spec has a ``requires_transcript`` tag
                use ``context["transcription"]`` for metadata prediction.

        Returns:
            Predicted output VideoMetadata after all operations.

        Raises:
            ValueError: If any validation check fails.
        """
        if isinstance(source_metadata, VideoMetadata):
            meta_map: dict[str, VideoMetadata] = {str(seg.source_video): source_metadata for seg in self.segments}
        else:
            meta_map = source_metadata

        source_metas: list[VideoMetadata] = []
        for i, segment in enumerate(self.segments):
            source_key = str(segment.source_video)
            if source_key not in meta_map:
                raise ValueError(
                    f"Segment {i}: no metadata provided for source '{source_key}'. Available keys: {sorted(meta_map)}"
                )
            source_metas.append(self._validate_source_meta(i, segment, meta_map[source_key]))
        source_metas = self._match_metas(source_metas)
        segment_metas = [
            self._apply_segment_meta_ops(i, seg, meta, context)
            for i, (seg, meta) in enumerate(zip(self.segments, source_metas))
        ]
        return self._validate_assembled(segment_metas, context)

    def _validate_assembled(
        self, segment_metas: list[VideoMetadata], runtime_context: dict[str, Any] | None = None
    ) -> VideoMetadata:
        if len(segment_metas) > 1:
            first = segment_metas[0]
            for j, other in enumerate(segment_metas[1:], start=1):
                if first.fps != other.fps:
                    raise ValueError(
                        f"Segment 0 output fps ({first.fps}) != segment {j} output fps ({other.fps}). "
                        f"All segments must have identical fps for concatenation."
                    )
                if (first.width, first.height) != (other.width, other.height):
                    raise ValueError(
                        f"Segment 0 output dimensions ({first.width}x{first.height}) != "
                        f"segment {j} output dimensions ({other.width}x{other.height}). "
                        f"All segments must have identical dimensions for concatenation."
                    )

        meta = VideoMetadata(
            height=segment_metas[0].height,
            width=segment_metas[0].width,
            fps=segment_metas[0].fps,
            frame_count=sum(m.frame_count for m in segment_metas),
            total_seconds=round(sum(m.total_seconds for m in segment_metas), 4),
        )

        for record in self.post_transform_records:
            meta = _predict_transform_metadata(
                meta,
                record.op_id,
                record.args,
                context=f"post-assembly ({record.op_id})",
                runtime_context=runtime_context,
            )
        for record in self.post_effect_records:
            _validate_effect_bounds(record, meta.total_seconds, context="post-assembly")

        return meta

    def _segment_to_dict(self, segment: SegmentConfig) -> dict[str, Any]:
        return {
            "source": str(segment.source_video),
            "start": segment.start_second,
            "end": segment.end_second,
            "transforms": [_step_to_dict(record, include_apply=False) for record in segment.transform_records],
            "effects": [_step_to_dict(record, include_apply=True) for record in segment.effect_records],
        }

    def _validate_source_meta(
        self, index: int, segment: SegmentConfig, source_meta: VideoMetadata | None = None
    ) -> VideoMetadata:
        """Validate segment bounds and return cut source metadata (no transforms/effects)."""
        ctx = f"Segment {index}"
        if segment.start_second < 0:
            raise ValueError(f"{ctx}: start_second ({segment.start_second}) must be >= 0")
        if segment.end_second <= segment.start_second:
            raise ValueError(
                f"{ctx}: end_second ({segment.end_second}) must be > start_second ({segment.start_second})"
            )
        meta = source_meta if source_meta is not None else VideoMetadata.from_path(str(segment.source_video))
        if segment.end_second > meta.total_seconds:
            raise ValueError(
                f"{ctx}: end_second ({segment.end_second}) exceeds source duration ({meta.total_seconds}s)"
            )
        return meta.cut(segment.start_second, segment.end_second)

    def _apply_segment_meta_ops(
        self,
        index: int,
        segment: SegmentConfig,
        meta: VideoMetadata,
        runtime_context: dict[str, Any] | None = None,
    ) -> VideoMetadata:
        """Apply per-segment transform/effect metadata predictions."""
        ctx = f"Segment {index}"
        for record in segment.transform_records:
            meta = _predict_transform_metadata(
                meta, record.op_id, record.args, context=f"{ctx} ({record.op_id})", runtime_context=runtime_context
            )
        for record in segment.effect_records:
            _validate_effect_bounds(record, meta.total_seconds, context=ctx)
        return meta

    def _match_metas(self, metas: list[VideoMetadata]) -> list[VideoMetadata]:
        """Apply matching to source metadata list."""
        if len(metas) <= 1:
            return metas
        if self.match_to_lowest_fps:
            min_fps = min(m.fps for m in metas)
            metas = [m.resample_fps(min_fps) if m.fps != min_fps else m for m in metas]
        if self.match_to_lowest_resolution:
            min_w = min(m.width for m in metas)
            min_h = min(m.height for m in metas)
            metas = [m.resize(width=min_w, height=min_h) if (m.width, m.height) != (min_w, min_h) else m for m in metas]
        return metas

    def _assemble_segments(self, context: dict[str, Any] | None = None) -> Video:
        # Compute matching targets from source metadata before loading.
        target_fps, target_w, target_h = None, None, None
        if len(self.segments) > 1 and (self.match_to_lowest_fps or self.match_to_lowest_resolution):
            source_metas = [VideoMetadata.from_path(str(seg.source_video)) for seg in self.segments]
            if self.match_to_lowest_fps:
                target_fps = min(m.fps for m in source_metas)
            if self.match_to_lowest_resolution:
                target_w = min(m.width for m in source_metas)
                target_h = min(m.height for m in source_metas)

        # Load segments with matching applied via ffmpeg, then apply per-segment ops.
        videos = [
            segment.apply_operations(
                segment.load_segment(fps=target_fps, width=target_w, height=target_h),
                context,
            )
            for segment in self.segments
        ]
        result = videos[0]
        for video in videos[1:]:
            result = result + video
        return result


@dataclass
class _VfResult:
    """Result of compiling a transform to an ffmpeg -vf filter."""

    filter_expr: str  # Empty string if handled by -ss/-t (e.g. CutSeconds)
    out_width: int
    out_height: int
    out_fps: float


# Transforms that can be compiled to ffmpeg -vf filters
_STREAMABLE_TRANSFORM_OPS = {"cut", "cut_frames", "resize", "resample_fps", "crop", "speed_change"}


def _compile_transform_to_vf(
    record: _StepRecord,
    cur_width: int,
    cur_height: int,
    cur_fps: float,
) -> _VfResult | None:
    """Compile a transform to an ffmpeg -vf filter. Returns None if not streamable."""
    if record.op_id not in _STREAMABLE_TRANSFORM_OPS:
        return None

    args = record.args

    if record.op_id in ("cut", "cut_frames"):
        # Cut is handled by -ss/-t on the FrameIterator, not a -vf filter
        return _VfResult("", cur_width, cur_height, cur_fps)

    if record.op_id == "resize":
        w = args.get("width")
        h = args.get("height")
        # Resolve aspect ratio if only one dimension given
        if w is None and h is not None:
            w = round(cur_width * h / cur_height)
            if w % 2 != 0:
                w += 1
        elif h is None and w is not None:
            h = round(cur_height * w / cur_width)
            if h % 2 != 0:
                h += 1
        if w is None or h is None:
            return None
        return _VfResult(f"scale={w}:{h}", w, h, cur_fps)

    if record.op_id == "resample_fps":
        fps = args.get("fps", cur_fps)
        return _VfResult(f"fps={fps}", cur_width, cur_height, float(fps))

    if record.op_id == "crop":
        cw = args.get("width", cur_width)
        ch = args.get("height", cur_height)
        # Convert float fractions to pixels
        if isinstance(cw, float) and 0 < cw <= 1:
            cw = round(cw * cur_width)
        if isinstance(ch, float) and 0 < ch <= 1:
            ch = round(ch * cur_height)
        cx = args.get("x", 0)
        cy = args.get("y", 0)
        if isinstance(cx, float) and 0 <= cx <= 1:
            cx = round(cx * cur_width)
        if isinstance(cy, float) and 0 <= cy <= 1:
            cy = round(cy * cur_height)
        mode = args.get("mode", "center")
        if mode == "center":
            cx = (cur_width - cw) // 2
            cy = (cur_height - ch) // 2
        return _VfResult(f"crop={cw}:{ch}:{cx}:{cy}", int(cw), int(ch), cur_fps)

    if record.op_id == "speed_change":
        speed = args.get("speed", 1.0)
        end_speed = args.get("end_speed")
        if end_speed is not None:
            return None  # Speed ramps are not streamable
        return _VfResult(f"setpts=PTS/{speed}", cur_width, cur_height, cur_fps)

    return None


def _apply_fade_audio(fade: Fade, audio: Audio, start_s: float, stop_s: float) -> None:
    """Apply the audio portion of a Fade effect."""
    from videopython.base.effects import _compute_curve

    sample_rate = audio.metadata.sample_rate
    audio_start = round(start_s * sample_rate)
    audio_end = min(round(stop_s * sample_rate), len(audio.data))
    n_samples = audio_end - audio_start
    fade_samples = min(round(fade.duration * sample_rate), n_samples)

    alpha = np.ones(n_samples, dtype=np.float32)
    if fade.mode in ("in", "in_out"):
        t = np.linspace(0, 1, fade_samples, dtype=np.float32)
        alpha[:fade_samples] = _compute_curve(t, fade.curve)
    if fade.mode in ("out", "in_out"):
        t = np.linspace(1, 0, fade_samples, dtype=np.float32)
        alpha[-fade_samples:] = np.minimum(alpha[-fade_samples:], _compute_curve(t, fade.curve))

    if audio.data.ndim == 1:
        audio.data[audio_start:audio_end] *= alpha
    else:
        audio.data[audio_start:audio_end] *= alpha[:, np.newaxis]
    np.clip(audio.data, -1.0, 1.0, out=audio.data)


def _apply_transform_with_context(record: _StepRecord, video: Video, context: dict[str, Any] | None) -> Video:
    """Apply a transform, injecting context data for operations that require it."""
    spec = get_operation_spec(record.op_id)
    if spec is not None and "requires_transcript" in spec.tags and context and "transcription" in context:
        return record.operation.apply(video, transcription=context["transcription"])  # type: ignore[call-arg]
    return record.operation.apply(video)


def _parse_segment(segment_data: Any, location: str) -> SegmentConfig:
    if not isinstance(segment_data, dict):
        raise ValueError(f"{location} must be an object")

    allowed_keys = {"source", "start", "end", "transforms", "effects"}
    unknown = sorted(set(segment_data) - allowed_keys)
    if unknown:
        raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")

    for key in ("source", "start", "end"):
        if key not in segment_data:
            raise ValueError(f"{location} is missing required key '{key}'")

    source = segment_data["source"]
    if not isinstance(source, str):
        raise ValueError(f"{location}.source must be a string path")

    start = _require_number(segment_data["start"], f"{location}.start")
    end = _require_number(segment_data["end"], f"{location}.end")

    transforms_data = segment_data.get("transforms", [])
    effects_data = segment_data.get("effects", [])
    if not isinstance(transforms_data, list):
        raise ValueError(f"{location}.transforms must be a list")
    if not isinstance(effects_data, list):
        raise ValueError(f"{location}.effects must be a list")

    transform_records = [
        _parse_transform_step(step, f"{location}.transforms[{i}]") for i, step in enumerate(transforms_data)
    ]
    effect_records = [_parse_effect_step(step, f"{location}.effects[{i}]") for i, step in enumerate(effects_data)]

    return SegmentConfig(
        source_video=Path(source),
        start_second=start,
        end_second=end,
        transform_records=tuple(transform_records),
        effect_records=tuple(effect_records),
    )


def _parse_transform_step(step: Any, location: str) -> _StepRecord:
    step_dict = _require_step_object(step, location)
    if "apply" in step_dict:
        raise ValueError(f"{location}: transforms do not accept apply params")

    allowed_keys = {"op", "args"}
    unknown = sorted(set(step_dict) - allowed_keys)
    if unknown:
        raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")
    if "op" not in step_dict:
        raise ValueError(f"{location} is missing required key 'op'")

    requested_op = step_dict["op"]
    if not isinstance(requested_op, str):
        raise ValueError(f"{location}.op must be a string")

    args = step_dict.get("args", {})
    spec = _resolve_and_validate_step_spec(requested_op, OperationCategory.TRANSFORMATION, location)

    op_cls = _load_operation_class(spec, requested_op, location)
    _ensure_json_instantiable(op_cls, spec, location)
    _validate_object_arg_map(args, spec.params, f"{location}.args")
    _validate_param_values(args, spec.params, f"{location}.args")
    _validate_step_semantics(spec.id, args, f"{location}.args")
    normalized_args = _normalize_constructor_args_for_class(op_cls, args, f"{location}.args")
    try:
        operation = op_cls(**normalized_args)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{location}: Failed to instantiate operation '{requested_op}' (canonical '{spec.id}'): {e}"
        ) from e

    if not isinstance(operation, Transformation):
        raise ValueError(f"{location}: Operation '{spec.id}' did not instantiate a Transformation")
    return _StepRecord.create(spec.id, args, {}, operation)


def _parse_effect_step(step: Any, location: str) -> _StepRecord:
    step_dict = _require_step_object(step, location)
    allowed_keys = {"op", "args", "apply"}
    unknown = sorted(set(step_dict) - allowed_keys)
    if unknown:
        raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")
    if "op" not in step_dict:
        raise ValueError(f"{location} is missing required key 'op'")

    requested_op = step_dict["op"]
    if not isinstance(requested_op, str):
        raise ValueError(f"{location}.op must be a string")

    args = step_dict.get("args", {})
    apply_args = step_dict.get("apply", {})

    spec = _resolve_and_validate_step_spec(requested_op, OperationCategory.EFFECT, location)

    op_cls = _load_operation_class(spec, requested_op, location)
    _ensure_json_instantiable(op_cls, spec, location)
    _validate_object_arg_map(args, spec.params, f"{location}.args")
    _validate_param_values(args, spec.params, f"{location}.args")
    _validate_step_semantics(spec.id, args, f"{location}.args")
    _validate_object_arg_map(apply_args, spec.apply_params, f"{location}.apply")
    _validate_param_values(apply_args, spec.apply_params, f"{location}.apply")
    normalized_apply_args = _normalize_effect_apply_args(apply_args, f"{location}.apply")
    normalized_args = _normalize_constructor_args_for_class(op_cls, args, f"{location}.args")
    try:
        operation = op_cls(**normalized_args)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{location}: Failed to instantiate operation '{requested_op}' (canonical '{spec.id}'): {e}"
        ) from e

    if not isinstance(operation, Effect):
        raise ValueError(f"{location}: Operation '{spec.id}' did not instantiate an Effect")
    return _StepRecord.create(spec.id, args, normalized_apply_args, operation)


def _require_step_object(step: Any, location: str) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError(f"{location} must be an object")
    return step


def _resolve_and_validate_step_spec(
    requested_op: str,
    expected_category: OperationCategory,
    location: str,
) -> OperationSpec:
    spec = get_operation_spec(requested_op)
    if spec is None:
        raise ValueError(
            f"{location}: Unknown operation '{requested_op}'. If this is an AI operation "
            "(e.g. face_crop, split_screen), ensure `import videopython.ai` is called before parsing the plan."
        )
    _ensure_videoedit_step_category_and_tags(spec, expected_category, location)
    return spec


def _load_operation_class(spec: OperationSpec, requested_op: str, location: str) -> type[Any]:
    try:
        module = importlib.import_module(spec.module_path)
        cls = getattr(module, spec.class_name)
        if not isinstance(cls, type):
            raise TypeError(f"Resolved attribute '{spec.class_name}' is not a class")
        return cls
    except (ImportError, AttributeError, TypeError) as e:
        raise ValueError(f"{location}: Failed to load operation '{requested_op}' (canonical '{spec.id}'): {e}") from e


def _ensure_json_instantiable(op_cls: type[Any], spec: OperationSpec, location: str) -> None:
    first_missing = _get_non_json_instantiable_missing_param(op_cls, spec)
    if first_missing is not None:
        raise ValueError(
            f"{location}: Operation '{spec.id}' is registered but not JSON-instantiable because required constructor "
            f"parameter '{first_missing}' is not included in the registry spec."
        )


def _validate_object_arg_map(value: Any, params: tuple[ParamSpec, ...], location: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    allowed = {param.name for param in params}
    required = [param.name for param in params if param.required]
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"{location} is missing required keys: {', '.join(missing)}")


def _normalize_effect_apply_args(apply_args: Mapping[str, Any], location: str) -> dict[str, Any]:
    """Normalize effect apply args that are consumed directly (not via constructor).

    Phase 2 defers broad primitive type validation, but `start`/`stop` are read directly
    during execution/validation, so validate/coerce them at parse time for actionable errors.
    """
    normalized = dict(apply_args)
    if "start" in normalized:
        normalized["start"] = _coerce_optional_number(normalized["start"], "start", location=f"{location}.start")
    if "stop" in normalized:
        normalized["stop"] = _coerce_optional_number(normalized["stop"], "stop", location=f"{location}.stop")
    return normalized


def _validate_param_values(value: Mapping[str, Any], params: tuple[ParamSpec, ...], location: str) -> None:
    param_map = {param.name: param for param in params}
    for key, raw in value.items():
        param = param_map.get(key)
        if param is None:
            continue
        _validate_param_value(raw, param, f"{location}.{key}")


def _validate_param_value(value: Any, param: ParamSpec, location: str) -> None:
    if value is None:
        if param.nullable:
            return
        raise ValueError(f"{location} must be a {param.json_type}")

    _validate_json_type(value, param.json_type, location)

    if param.enum is not None and value not in param.enum:
        allowed = ", ".join(repr(v) for v in param.enum)
        raise ValueError(f"{location} must be one of: {allowed}")

    if param.json_type in {"integer", "number"}:
        numeric_value = float(value)
        if param.minimum is not None and numeric_value < param.minimum:
            raise ValueError(f"{location} must be >= {param.minimum}")
        if param.maximum is not None and numeric_value > param.maximum:
            raise ValueError(f"{location} must be <= {param.maximum}")
        if param.exclusive_minimum is not None and numeric_value <= param.exclusive_minimum:
            raise ValueError(f"{location} must be > {param.exclusive_minimum}")
        if param.exclusive_maximum is not None and numeric_value >= param.exclusive_maximum:
            raise ValueError(f"{location} must be < {param.exclusive_maximum}")

    if param.json_type == "array" and param.items_type is not None:
        assert isinstance(value, list)
        for i, item in enumerate(value):
            _validate_json_type(item, param.items_type, f"{location}[{i}]")


def _validate_json_type(value: Any, json_type: str, location: str) -> None:
    if json_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{location} must be an integer")
        return

    if json_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{location} must be a number")
        return

    if json_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{location} must be a string")
        return

    if json_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{location} must be a boolean")
        return

    if json_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{location} must be an array")
        return

    if json_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{location} must be an object")
        return


def _normalize_constructor_args_for_class(
    op_cls: type[Any],
    args: Mapping[str, Any],
    location: str,
) -> dict[str, Any]:
    normalized = dict(args)
    try:
        module = inspect.getmodule(op_cls)
        globalns = vars(module) if module is not None else None
        type_hints = get_type_hints(op_cls.__init__, globalns=globalns, localns=globalns)
    except (AttributeError, NameError, TypeError):
        type_hints = {}

    for key, value in list(normalized.items()):
        annotation = type_hints.get(key, inspect.Signature.empty)
        normalized[key] = _normalize_value_for_annotation(value, annotation, f"{location}.{key}")
    return normalized


def _normalize_value_for_annotation(value: Any, annotation: Any, location: str) -> Any:
    if annotation is inspect.Signature.empty or value is None:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if _is_union_origin(origin):
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _normalize_value_for_annotation(value, non_none_args[0], location)
        return value

    if origin is tuple and isinstance(value, list):
        return tuple(value)

    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        try:
            return annotation(value)
        except ValueError as e:
            raise ValueError(f"{location} has invalid enum value {value!r}") from e

    return value


def _is_union_origin(origin: Any) -> bool:
    return origin in (Union, UnionType)


def _validate_step_semantics(op_id: str, args: Mapping[str, Any], location: str) -> None:
    if op_id == "resize":
        width = args.get("width")
        height = args.get("height")
        if width is None and height is None:
            raise ValueError(f"{location} must include at least one non-null value for 'width' or 'height'")


def _validate_step_record_apply_args_contract(apply_args: Mapping[str, Any]) -> None:
    for key in ("start", "stop"):
        if key not in apply_args:
            continue
        value = apply_args[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                "_StepRecord.apply_args values for 'start'/'stop' must be numeric or None before execution/validation"
            )


def _step_to_dict(record: _StepRecord, *, include_apply: bool) -> dict[str, Any]:
    step: dict[str, Any] = {"op": record.op_id}
    args_copy = copy.deepcopy(record.args)
    if args_copy:
        step["args"] = args_copy
    if include_apply:
        apply_copy = copy.deepcopy(record.apply_args)
        if apply_copy:
            step["apply"] = apply_copy
    return step


def _videoedit_supported_specs_for_category(category: OperationCategory) -> list[OperationSpec]:
    supported: list[OperationSpec] = []
    for spec in sorted(get_operation_specs().values(), key=lambda s: s.id):
        ok, _ = _is_videoedit_json_supported_spec(spec, category)
        if ok:
            supported.append(spec)
    return supported


def _is_videoedit_json_supported_spec(
    spec: OperationSpec, expected_category: OperationCategory
) -> tuple[bool, str | None]:
    if spec.category != expected_category:
        return False, f"category '{spec.category.value}'"

    blocked_tag = _get_unsupported_videoedit_tag(spec)
    if blocked_tag is not None:
        return False, f"tag '{blocked_tag}'"

    try:
        op_cls = _load_operation_class(spec, spec.id, "VideoEdit.json_schema()")
    except ValueError:
        return False, "failed to load class"
    missing_param = _get_non_json_instantiable_missing_param(op_cls, spec)
    if missing_param is not None:
        return False, f"non-JSON-instantiable ({missing_param})"

    return True, None


def _videoedit_step_schema_from_spec(spec: OperationSpec, *, include_apply: bool) -> dict[str, Any]:
    required: list[str] = ["op"]
    args_schema = spec.to_json_schema()
    if spec.id == "resize":
        args_schema = dict(args_schema)
        args_schema["anyOf"] = [
            {
                "required": ["width"],
                "properties": {
                    "width": {"not": {"type": "null"}},
                },
            },
            {
                "required": ["height"],
                "properties": {
                    "height": {"not": {"type": "null"}},
                },
            },
        ]
    properties: dict[str, Any] = {
        "op": {
            "const": spec.id,
            "description": "Canonical videopython operation ID.",
        },
        "args": args_schema,
    }

    if any(param.required for param in spec.params):
        required.append("args")
    elif spec.id == "resize":
        required.append("args")

    if include_apply:
        properties["apply"] = spec.to_apply_json_schema()
        if any(param.required for param in spec.apply_params):
            required.append("apply")

    return {
        "type": "object",
        "description": spec.description,
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _ensure_videoedit_step_category_and_tags(
    spec: OperationSpec,
    expected_category: OperationCategory,
    location: str,
) -> None:
    # 1) Category check
    if spec.category != expected_category:
        raise ValueError(
            f"{location}: Expected {expected_category.value} operation, got {spec.category.value} ('{spec.id}')"
        )

    # 2) Tag check
    blocked_tag = _get_unsupported_videoedit_tag(spec)
    if blocked_tag is not None:
        raise ValueError(
            f"{location}: Operation '{spec.id}' is not supported in VideoEdit JSON plans (tag '{blocked_tag}')"
        )


def _get_unsupported_videoedit_tag(spec: OperationSpec) -> str | None:
    for tag in ("multi_source", "multi_source_only"):
        if tag in spec.tags:
            return tag
    return None


def _get_non_json_instantiable_missing_param(op_cls: type[Any], spec: OperationSpec) -> str | None:
    sig = inspect.signature(op_cls.__init__)
    required_init_params = [
        p.name for p in sig.parameters.values() if p.name != "self" and p.default is inspect.Signature.empty
    ]
    spec_param_names = {param.name for param in spec.params}
    missing = [name for name in required_init_params if name not in spec_param_names]
    return missing[0] if missing else None


def _predict_transform_metadata(
    meta: VideoMetadata,
    op_id: str,
    args: Mapping[str, Any],
    context: str = "",
    runtime_context: dict[str, Any] | None = None,
) -> VideoMetadata:
    if op_id == "crop":
        try:
            return _predict_crop_metadata(meta, args)
        except (TypeError, ValueError) as e:
            prefix = f"{context}: " if context else ""
            raise ValueError(f"{prefix}Metadata prediction failed for transform op '{op_id}': {e}") from e

    spec = get_operation_spec(op_id)
    if spec is None or spec.category != OperationCategory.TRANSFORMATION or spec.metadata_method is None:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}Metadata prediction is not supported for transform op '{op_id}'. "
            "Only transforms with a registered metadata_method can be validated."
        )

    try:
        method = getattr(VideoMetadata, spec.metadata_method)
    except AttributeError as e:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}Registry metadata_method '{spec.metadata_method}' for op '{op_id}' "
            "is not implemented on VideoMetadata"
        ) from e

    accepted_params = {p.name for p in inspect.signature(method).parameters.values() if p.name != "self"}
    prepared = _prepare_metadata_args(meta, op_id, args, accepted_params)

    if spec is not None and "requires_transcript" in spec.tags:
        if runtime_context and "transcription" in runtime_context:
            if "transcription" in accepted_params:
                prepared["transcription"] = runtime_context["transcription"]
        else:
            prefix = f"{context}: " if context else ""
            raise ValueError(
                f"{prefix}Op '{op_id}' requires transcription context for metadata prediction. "
                "Pass context={'transcription': ...} to validate() or validate_with_metadata()."
            )

    try:
        return getattr(meta, spec.metadata_method)(**prepared)
    except (TypeError, ValueError) as e:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}Metadata prediction failed for transform op '{op_id}': {e}") from e


def _prepare_metadata_args(
    meta: VideoMetadata,
    op_id: str,
    args: Mapping[str, Any],
    accepted_params: set[str],
) -> dict[str, Any]:
    filtered = {k: v for k, v in args.items() if k in accepted_params}

    if op_id == "speed_change":
        end_speed = args.get("end_speed")
        if end_speed is not None and "speed" in filtered:
            filtered["speed"] = (filtered["speed"] + end_speed) / 2

    return filtered


def _predict_crop_metadata(meta: VideoMetadata, args: Mapping[str, Any]) -> VideoMetadata:
    width_raw = args.get("width")
    height_raw = args.get("height")
    if width_raw is None or height_raw is None:
        raise ValueError("crop metadata prediction requires both 'width' and 'height'")

    crop_width = _crop_value_to_pixels(width_raw, meta.width)
    crop_height = _crop_value_to_pixels(height_raw, meta.height)
    crop_x = _crop_value_to_pixels(args.get("x", 0), meta.width)
    crop_y = _crop_value_to_pixels(args.get("y", 0), meta.height)
    mode = _crop_mode_value(args.get("mode", "center"))

    if mode == "center":
        center_height = meta.height // 2
        center_width = meta.width // 2
        width_offset = crop_width // 2
        height_offset = crop_height // 2
        out_height = _slice_length(meta.height, center_height - height_offset, center_height + height_offset)
        out_width = _slice_length(meta.width, center_width - width_offset, center_width + width_offset)
    else:
        out_height = _slice_length(meta.height, crop_y, crop_y + crop_height)
        out_width = _slice_length(meta.width, crop_x, crop_x + crop_width)

    return meta.with_dimensions(out_width, out_height)


def _crop_value_to_pixels(value: Any, dimension: int) -> int:
    """Convert a crop value to pixels.

    Float values in the range (0, 1] are treated as fractions of *dimension*
    (e.g. 0.5 means 50%). All other numeric values (including integers) are
    treated as absolute pixel counts.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("crop values must be numeric")
    if isinstance(value, float) and 0 < value <= 1:
        return int(value * dimension)
    return int(value)


def _crop_mode_value(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise ValueError("crop mode must be a string or Enum")
    return value


def _slice_length(size: int, start: int, stop: int) -> int:
    normalized_start, normalized_stop, step = slice(start, stop).indices(size)
    return max(0, (normalized_stop - normalized_start + (step - 1)) // step)


def _validate_effect_bounds(
    record: _StepRecord,
    duration: float,
    context: str = "",
) -> None:
    prefix = f"{context}: " if context else ""
    effect_name = type(record.operation).__name__
    op_label = record.op_id

    start = _coerce_optional_number(record.apply_args.get("start"), "start")
    stop = _coerce_optional_number(record.apply_args.get("stop"), "stop")
    # Tolerance for floating-point rounding in frame-based duration calculations.
    eps = 1e-3

    if start is not None:
        if start < 0:
            raise ValueError(f"{prefix}Effect '{op_label}' ({effect_name}) start ({start}) must be >= 0")
        if start > duration + eps:
            raise ValueError(
                f"{prefix}Effect '{op_label}' ({effect_name}) start ({start}) exceeds timeline duration ({duration}s)"
            )
    if stop is not None:
        if stop < 0:
            raise ValueError(f"{prefix}Effect '{op_label}' ({effect_name}) stop ({stop}) must be >= 0")
        if stop > duration + eps:
            raise ValueError(
                f"{prefix}Effect '{op_label}' ({effect_name}) stop ({stop}) exceeds timeline duration ({duration}s)"
            )
    if start is not None and stop is not None and start > stop:
        raise ValueError(f"{prefix}Effect '{op_label}' ({effect_name}) start ({start}) must be <= stop ({stop})")


def _require_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    return float(value)


def _coerce_optional_number(value: Any, param_name: str, *, location: str | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        label = location if location is not None else f"Effect apply parameter '{param_name}'"
        raise ValueError(f"{label} must be a number")
    return float(value)

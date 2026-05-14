"""Multi-segment video editing plans.

`VideoEdit` is a thin Pydantic model: fields ARE the JSON wire format, validation
and (de)serialization are handled by Pydantic. Each segment carries an ordered
``operations`` list of :class:`videopython.base.operation.Operation` instances
resolved through the auto-registry on the ``op`` discriminator field.

Wire format::

    {"segments": [{"source": "a.mp4", "start": 0, "end": 5,
        "operations": [{"op": "resize", "width": 1280},
                       {"op": "blur_effect", "mode": "constant",
                        "iterations": 10,
                        "window": {"start": 1, "stop": 3}}]}],
     "post_operations": [...],
     "match_to_lowest_fps": true,
     "match_to_lowest_resolution": true}
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SerializeAsAny, model_validator

from videopython.base.audio import Audio, AudioLoadError
from videopython.base.effects import Effect, Fade, VolumeAdjust
from videopython.base.operation import FilterCtx, Operation
from videopython.base.streaming import EffectScheduleEntry, StreamingSegmentPlan, concat_files, stream_segment
from videopython.base.transforms import CutSeconds
from videopython.base.video import ALLOWED_VIDEO_FORMATS, ALLOWED_VIDEO_PRESETS, Video, VideoMetadata

__all__ = [
    "SegmentConfig",
    "VideoEdit",
]


def _resolve_operation(value: Any) -> Operation:
    """BeforeValidator: turn a dict into the right :class:`Operation` subclass.

    Uses the registry keyed on ``op`` to find the concrete subclass, then lets
    Pydantic validate the rest of the fields on that subclass. Already-resolved
    ``Operation`` instances pass through unchanged.
    """
    if isinstance(value, Operation):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Operation must be a dict or Operation instance, got {type(value).__name__}")
    op_id = value.get("op")
    if not isinstance(op_id, str):
        raise ValueError("Operation dict missing required 'op' field")
    try:
        cls = Operation.get(op_id)
    except KeyError as e:
        raise ValueError(str(e)) from e
    return cls.model_validate(value)


OperationInput = Annotated[SerializeAsAny[Operation], BeforeValidator(_resolve_operation)]


def _apply_with_context(op: Operation, video: Video, context: dict[str, Any] | None) -> Video:
    """Apply ``op`` to ``video``, threading ``op.requires`` keys from ``context``."""
    if op.requires and context:
        kwargs = {k: context[k] for k in op.requires if k in context}
        return op.apply(video, **kwargs)  # type: ignore[call-arg]
    return op.apply(video)


def _predict_with_context(
    op: Operation,
    meta: VideoMetadata,
    context: dict[str, Any] | None,
) -> VideoMetadata:
    """Run ``op.predict_metadata``, threading requires-keys from ``context``."""
    if op.requires and context:
        kwargs = {k: context[k] for k in op.requires if k in context}
        return op.predict_metadata(meta, **kwargs)  # type: ignore[call-arg]
    return op.predict_metadata(meta)


def _validate_effect_window(op: Operation, duration: float) -> None:
    """Bounds-check :attr:`Effect.window` against the predicted duration."""
    if not isinstance(op, Effect) or op.window is None:
        return
    eps = 1e-3
    if op.window.start is not None and op.window.start > duration + eps:
        raise ValueError(f"Effect '{op.op}' window.start ({op.window.start}) exceeds duration ({duration}s)")
    if op.window.stop is not None and op.window.stop > duration + eps:
        raise ValueError(f"Effect '{op.op}' window.stop ({op.window.stop}) exceeds duration ({duration}s)")


class SegmentConfig(BaseModel):
    """A single source segment with its operation chain."""

    model_config = ConfigDict(extra="forbid")

    source: Path
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    operations: list[OperationInput] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_range(self) -> SegmentConfig:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start

    def load(
        self,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Video:
        """Load the raw segment from disk with optional decode-time matching."""
        return Video.from_path(
            str(self.source),
            start_second=self.start,
            end_second=self.end,
            fps=fps,
            width=width,
            height=height,
        )

    def process(self, video: Video, context: dict[str, Any] | None = None) -> Video:
        """Apply every operation in this segment to ``video`` in order."""
        for op in self.operations:
            video = _apply_with_context(op, video, context)
        return video


class VideoEdit(BaseModel):
    """A multi-segment editing plan."""

    model_config = ConfigDict(extra="forbid")

    segments: list[SegmentConfig] = Field(min_length=1)
    post_operations: list[OperationInput] = Field(default_factory=list)
    match_to_lowest_fps: bool = True
    match_to_lowest_resolution: bool = True

    # ------------------------------------------------------------------ I/O

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoEdit:
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, text: str) -> VideoEdit:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid VideoEdit JSON: {e.msg} at line {e.lineno} column {e.colno}") from e
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """LLM-facing schema: a discriminated union of operations per slot."""
        op_schema = Operation.json_schema()
        segment_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Source video path."},
                "start": {"type": "number", "minimum": 0, "description": "Segment start time in seconds."},
                "end": {"type": "number", "minimum": 0, "description": "Segment end time in seconds."},
                "operations": {"type": "array", "items": op_schema, "default": []},
            },
            "required": ["source", "start", "end"],
            "additionalProperties": False,
        }
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "segments": {"type": "array", "items": segment_schema, "minItems": 1},
                "post_operations": {"type": "array", "items": op_schema, "default": []},
                "match_to_lowest_fps": {"type": "boolean", "default": True},
                "match_to_lowest_resolution": {"type": "boolean", "default": True},
            },
            "required": ["segments"],
            "additionalProperties": False,
        }

    # --------------------------------------------------------------- validate

    def validate(self, context: dict[str, Any] | None = None) -> VideoMetadata:  # type: ignore[override]
        """Dry-run the plan via metadata. Requires source files on disk.

        Shadows Pydantic v1's deprecated ``BaseModel.validate`` classmethod;
        use ``VideoEdit.from_dict``/``model_validate`` for plan parsing.
        """
        source_metas = [VideoMetadata.from_path(str(seg.source)) for seg in self.segments]
        return self._validate(source_metas, context)

    def validate_with_metadata(
        self,
        source_metadata: VideoMetadata | dict[str, VideoMetadata],
        context: dict[str, Any] | None = None,
    ) -> VideoMetadata:
        """Dry-run with pre-built metadata, avoiding disk access."""
        if isinstance(source_metadata, VideoMetadata):
            metas = [source_metadata for _ in self.segments]
        else:
            metas = []
            for i, seg in enumerate(self.segments):
                key = str(seg.source)
                if key not in source_metadata:
                    available = sorted(source_metadata)
                    raise ValueError(f"Segment {i}: no metadata for '{key}'. Available: {available}")
                metas.append(source_metadata[key])
        return self._validate(metas, context)

    def _validate(
        self,
        source_metas: list[VideoMetadata],
        context: dict[str, Any] | None,
    ) -> VideoMetadata:
        cut_metas: list[VideoMetadata] = []
        for i, (seg, meta) in enumerate(zip(self.segments, source_metas)):
            if seg.end > meta.total_seconds + 1e-3:
                raise ValueError(f"Segment {i}: end ({seg.end}) exceeds source duration ({meta.total_seconds}s)")
            cut_metas.append(CutSeconds(start=seg.start, end=seg.end).predict_metadata(meta))

        matched = self._apply_matching(cut_metas)
        segment_outputs = [
            self._predict_segment(i, seg, meta, context) for i, (seg, meta) in enumerate(zip(self.segments, matched))
        ]
        self._assert_concat_compatible(segment_outputs)

        first = segment_outputs[0]
        assembled = VideoMetadata(
            height=first.height,
            width=first.width,
            fps=first.fps,
            frame_count=sum(m.frame_count for m in segment_outputs),
            total_seconds=round(sum(m.total_seconds for m in segment_outputs), 4),
        )
        for op in self.post_operations:
            _validate_effect_window(op, assembled.total_seconds)
            assembled = _predict_with_context(op, assembled, context)
        return assembled

    def _predict_segment(
        self,
        index: int,
        segment: SegmentConfig,
        meta: VideoMetadata,
        context: dict[str, Any] | None,
    ) -> VideoMetadata:
        for op in segment.operations:
            _validate_effect_window(op, meta.total_seconds)
            try:
                meta = _predict_with_context(op, meta, context)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Segment {index}: metadata prediction failed for '{op.op}': {e}") from e
        return meta

    def _apply_matching(self, metas: list[VideoMetadata]) -> list[VideoMetadata]:
        if len(metas) <= 1:
            return metas
        result = metas
        if self.match_to_lowest_fps:
            min_fps = min(m.fps for m in result)
            result = [m.with_fps(min_fps) if m.fps != min_fps else m for m in result]
        if self.match_to_lowest_resolution:
            min_w = min(m.width for m in result)
            min_h = min(m.height for m in result)
            result = [m.with_dimensions(min_w, min_h) if (m.width, m.height) != (min_w, min_h) else m for m in result]
        return result

    @staticmethod
    def _assert_concat_compatible(metas: list[VideoMetadata]) -> None:
        if len(metas) <= 1:
            return
        first = metas[0]
        for j, other in enumerate(metas[1:], start=1):
            if first.fps != other.fps:
                raise ValueError(
                    f"Segment 0 fps ({first.fps}) != segment {j} fps ({other.fps}); "
                    "all segments must share fps for concatenation."
                )
            if (first.width, first.height) != (other.width, other.height):
                raise ValueError(
                    f"Segment 0 dimensions ({first.width}x{first.height}) != "
                    f"segment {j} ({other.width}x{other.height}); all segments must share dimensions."
                )

    # -------------------------------------------------------------------- run

    def run(self, context: dict[str, Any] | None = None) -> Video:
        """Execute the plan in memory and return the final ``Video``."""
        target_fps, target_w, target_h = self._matching_targets_from_disk()
        videos = [
            segment.process(segment.load(fps=target_fps, width=target_w, height=target_h), context)
            for segment in self.segments
        ]
        result = videos[0]
        for video in videos[1:]:
            result = result + video
        for op in self.post_operations:
            result = _apply_with_context(op, result, context)
        return result

    def run_to_file(
        self,
        output_path: str | Path,
        format: ALLOWED_VIDEO_FORMATS = "mp4",
        preset: ALLOWED_VIDEO_PRESETS = "medium",
        crf: int = 23,
        context: dict[str, Any] | None = None,
    ) -> Path:
        """Execute the plan, streaming directly to a file when possible.

        Falls back to eager (``self.run().save(...)``) for any operation that
        isn't streamable. Memory usage is O(1) w.r.t. video length for fully
        streamable pipelines.
        """
        output_path = Path(output_path).with_suffix(f".{format}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_fps, target_w, target_h = self._matching_targets_from_disk()
        plans: list[StreamingSegmentPlan] = []
        for segment in self.segments:
            plan = self._build_streaming_plan(segment, target_fps, target_w, target_h)
            if plan is None:
                return self._run_to_file_eager(output_path, format, preset, crf, context)
            plans.append(plan)

        # Post-ops only fold cleanly into a single segment plan; multi-segment
        # post-ops would need a second pass we don't bother with.
        if self.post_operations and len(plans) != 1:
            return self._run_to_file_eager(output_path, format, preset, crf, context)
        if self.post_operations:
            plan = plans[0]
            total_frames = round((plan.end_second - plan.start_second) * plan.output_fps)
            for op in self.post_operations:
                if not isinstance(op, Effect) or not op.streamable:
                    return self._run_to_file_eager(output_path, format, preset, crf, context)
                start_f, end_f = _effect_frame_range(op, plan.output_fps, total_frames)
                plan.effect_schedule.append(EffectScheduleEntry(op, start_f, end_f))

        if len(plans) == 1:
            plan = plans[0]
            audio = self._load_segment_audio(self.segments[0], plan)
            return stream_segment(plan, output_path, audio=audio, format=format, preset=preset, crf=crf)

        temp_files: list[Path] = []
        try:
            for segment, plan in zip(self.segments, plans):
                tmp = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
                tmp.close()
                audio = self._load_segment_audio(segment, plan)
                stream_segment(plan, Path(tmp.name), audio=audio, format=format, preset=preset, crf=crf)
                temp_files.append(Path(tmp.name))
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
        video = self.run(context=context)
        return video.save(output_path, format=format, preset=preset, crf=crf)

    # ----------------------------------------------------------------- helpers

    def _matching_targets_from_disk(self) -> tuple[float | None, int | None, int | None]:
        if len(self.segments) <= 1 or (not self.match_to_lowest_fps and not self.match_to_lowest_resolution):
            return None, None, None
        metas = [VideoMetadata.from_path(str(seg.source)) for seg in self.segments]
        fps = min(m.fps for m in metas) if self.match_to_lowest_fps else None
        w = min(m.width for m in metas) if self.match_to_lowest_resolution else None
        h = min(m.height for m in metas) if self.match_to_lowest_resolution else None
        return fps, w, h

    def _build_streaming_plan(
        self,
        segment: SegmentConfig,
        target_fps: float | None,
        target_w: int | None,
        target_h: int | None,
    ) -> StreamingSegmentPlan | None:
        source_meta = VideoMetadata.from_path(str(segment.source))
        out_fps = target_fps or source_meta.fps
        out_w = target_w or source_meta.width
        out_h = target_h or source_meta.height

        vf_filters: list[str] = []
        if target_w and target_h and (target_w != source_meta.width or target_h != source_meta.height):
            vf_filters.append(f"scale={target_w}:{target_h}")
        if target_fps and target_fps != source_meta.fps:
            vf_filters.append(f"fps={target_fps}")

        effect_schedule: list[EffectScheduleEntry] = []
        for op in segment.operations:
            if isinstance(op, Effect):
                if not op.streamable:
                    return None
                total_frames = round(segment.duration * out_fps)
                start_f, end_f = _effect_frame_range(op, out_fps, total_frames)
                effect_schedule.append(EffectScheduleEntry(op, start_f, end_f))
                continue
            # Non-effect transform: compile to ffmpeg filter if streamable.
            ctx = FilterCtx(width=out_w, height=out_h, fps=out_fps)
            filter_expr = op.to_ffmpeg_filter(ctx)
            if filter_expr is None:
                return None
            vf_filters.append(filter_expr)
            new_meta = op.predict_metadata(
                VideoMetadata(height=out_h, width=out_w, fps=out_fps, frame_count=1, total_seconds=1.0)
            )
            out_w, out_h, out_fps = new_meta.width, new_meta.height, new_meta.fps

        return StreamingSegmentPlan(
            source_path=segment.source,
            start_second=segment.start,
            end_second=segment.end,
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
    ) -> Audio | None:
        try:
            audio = Audio.from_path(str(segment.source))
            audio = audio.slice(segment.start, segment.end)
        except (AudioLoadError, FileNotFoundError, subprocess.CalledProcessError):
            warnings.warn(f"No audio found for `{segment.source}`, using silent track.")
            audio = Audio.create_silent(duration_seconds=round(segment.duration, 2), stereo=True, sample_rate=44100)

        for entry in plan.effect_schedule:
            effect = entry.effect
            if isinstance(effect, (Fade, VolumeAdjust)) and not audio.is_silent:
                start_s = entry.start_frame / plan.output_fps
                stop_s = entry.end_frame / plan.output_fps
                effect._apply_audio(audio, start_s, stop_s)

        return audio


def _effect_frame_range(op: Effect, fps: float, total_frames: int) -> tuple[int, int]:
    """Resolve an effect's ``window`` to a ``(start_frame, end_frame)`` pair."""
    if op.window is None:
        return 0, total_frames
    start_s = op.window.start
    stop_s = op.window.stop
    start_f = round(start_s * fps) if start_s is not None else 0
    end_f = round(stop_s * fps) if stop_s is not None else total_frames
    return start_f, end_f

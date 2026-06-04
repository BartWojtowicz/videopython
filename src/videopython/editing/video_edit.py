"""Multi-segment video editing plans.

`VideoEdit` is a thin Pydantic model: fields ARE the JSON wire format, validation
and (de)serialization are handled by Pydantic. Each segment carries an ordered
``operations`` list of :class:`videopython.editing.operation.Operation` instances
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
from typing import Annotated, Any, NamedTuple, Protocol, runtime_checkable

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SerializeAsAny, model_validator

from videopython.audio import Audio, AudioLoadError
from videopython.base.exceptions import PlanError, PlanErrorCode, PlanValidationError
from videopython.base.video import ALLOWED_VIDEO_FORMATS, ALLOWED_VIDEO_PRESETS, Video, VideoMetadata
from videopython.editing.effects import Effect, Fade, VolumeAdjust
from videopython.editing.operation import FilterCtx, Operation
from videopython.editing.streaming import EffectScheduleEntry, StreamingSegmentPlan, concat_files, stream_segment
from videopython.editing.transforms import DURATION_EPS, CutSeconds

__all__ = [
    "SegmentConfig",
    "VideoEdit",
    "WindowClamp",
]


class WindowClamp(NamedTuple):
    """A single ``window.stop`` repair record produced by :meth:`VideoEdit.repair`.

    ``location`` is the plan path of the clamped op (e.g.
    ``'segments[0].operations[1]'``); ``field`` is always ``'window.stop'``.
    """

    location: str
    field: str
    old: float
    new: float


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
        # Preserve the exact prose (str() on KeyError, incl. its quoting) so the
        # message stays byte-identical to the previous bare ValueError.
        raise PlanValidationError(
            str(e),
            [PlanError(code=PlanErrorCode.UNKNOWN_OP, op=op_id)],
        ) from e
    return cls.model_validate(value)


OperationInput = Annotated[SerializeAsAny[Operation], BeforeValidator(_resolve_operation)]


@runtime_checkable
class SegmentRebaseable(Protocol):
    """A runtime-context value carrying a source-absolute timeline.

    Any context entry implementing both ``slice(start, end)`` and
    ``offset(delta)`` -- e.g. :class:`videopython.base.transcription.Transcription`
    -- is automatically re-based onto each segment's 0-based local timeline by
    the runner, with no per-type wiring. Keying off structure rather than a
    concrete class keeps the context mechanism generic for future time-based
    context (beat maps, scene markers, ...) and avoids a layering dependency
    from the editing layer onto every such type.
    """

    def slice(self, start: float, end: float) -> SegmentRebaseable | None: ...

    def offset(self, delta: float) -> SegmentRebaseable: ...


def _rebaseable_keys(context: dict[str, Any] | None) -> set[str]:
    """Context keys whose value carries a re-baseable source-absolute timeline."""
    if not context:
        return set()
    return {k for k, v in context.items() if isinstance(v, SegmentRebaseable)}


def _segment_context(
    context: dict[str, Any] | None,
    start: float,
    end: float,
) -> dict[str, Any] | None:
    """Re-base time-based context entries onto a cut segment's local timeline.

    A cut segment is decoded 0-based -- its first frame is ``t=0`` -- but
    context values may carry source-absolute timestamps. Every value
    implementing :class:`SegmentRebaseable` (e.g. a ``Transcription``) is
    sliced to ``[start, end)`` and shifted by ``-start`` so segment operations
    (``add_subtitles``, ``silence_removal``) see segment-local time. Without
    this, subtitles on a segment cut from the middle of a video render blank.
    Values that don't implement the protocol pass through untouched.

    Slicing always runs (even for ``start == 0``) so out-of-range entries do
    not bleed in. When ``slice`` yields nothing the key is dropped rather than
    passed empty, so the consuming operation raises its own clear "requires
    ..." error instead of silently doing nothing.

    Scope: per-segment only. ``post_operations`` run on the assembled,
    concatenated timeline; re-basing time-based context across a multi-segment
    concat is unsupported and rejected up front by
    :meth:`VideoEdit._assert_post_ops_supported` (single-segment plans are
    unaffected).
    """
    if not context:
        return context
    rebaseable = {k: v for k, v in context.items() if isinstance(v, SegmentRebaseable)}
    if not rebaseable:
        return context
    rebased = dict(context)
    for key, value in rebaseable.items():
        sliced = value.slice(start, end)
        if sliced is None:
            del rebased[key]
        else:
            rebased[key] = sliced.offset(-start)
    return rebased


def _apply_with_context(op: Operation, video: Video, context: dict[str, Any] | None) -> Video:
    """Apply ``op`` to ``video``, threading ``op.requires`` keys from ``context``."""
    if op.requires and context:
        kwargs = {k: context[k] for k in op.requires if k in context}
        return op.apply(video, **kwargs)
    return op.apply(video)


def _predict_with_context(
    op: Operation,
    meta: VideoMetadata,
    context: dict[str, Any] | None,
) -> VideoMetadata:
    """Run ``op.predict_metadata``, threading requires-keys from ``context``."""
    if op.requires and context:
        kwargs = {k: context[k] for k in op.requires if k in context}
        return op.predict_metadata(meta, **kwargs)
    return op.predict_metadata(meta)


def _validate_effect_window(op: Operation, duration: float, location: str | None = None) -> None:
    """Bounds-check :attr:`Effect.window` against the predicted duration."""
    if not isinstance(op, Effect) or op.window is None:
        return
    eps = DURATION_EPS
    if op.window.start is not None and op.window.start > duration + eps:
        message = f"Effect '{op.op}' window.start ({op.window.start}) exceeds duration ({duration}s)"
        raise PlanValidationError(
            message,
            [
                PlanError(
                    code=PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION,
                    location=location,
                    op=op.op,
                    field="window.start",
                    value=op.window.start,
                    limit=duration,
                    predicted_duration=duration,
                )
            ],
        )
    if op.window.stop is not None and op.window.stop > duration + eps:
        message = f"Effect '{op.op}' window.stop ({op.window.stop}) exceeds duration ({duration}s)"
        raise PlanValidationError(
            message,
            [
                PlanError(
                    code=PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION,
                    location=location,
                    op=op.op,
                    field="window.stop",
                    value=op.window.stop,
                    limit=duration,
                    predicted_duration=duration,
                )
            ],
        )


def _clamp_effect_window(op: Operation, duration: float) -> Operation:
    """Return ``op`` with its ``Effect.window.stop`` clamped to ``duration``.

    Mirrors :meth:`Effect._resolved_window`'s run-time ``min(stop, total_seconds)``
    so a stop overrunning a duration-shrunk chain validates instead of raising.
    ``TimeRange`` is ``frozen=True``, so the repair builds replacement copies via
    ``model_copy(update=...)`` rather than mutating in place. ``window.start`` is
    deliberately left untouched -- an out-of-range start still hard-raises.
    """
    if not isinstance(op, Effect) or op.window is None or op.window.stop is None:
        return op
    if op.window.stop <= duration:
        return op
    clamped_window = op.window.model_copy(update={"stop": duration})
    return op.model_copy(update={"window": clamped_window})


class SegmentConfig(BaseModel):
    """A single source segment with its operation chain."""

    model_config = ConfigDict(extra="forbid")

    source: Path = Field(description="Path to the source video file.")
    start: float = Field(ge=0, description="Segment start time in seconds.")
    end: float = Field(ge=0, description="Segment end time in seconds.")
    operations: list[OperationInput] = Field(
        default_factory=list,
        description=(
            "Ordered list of operations to run against this segment. "
            "Each item is an Operation discriminated by its `op` field."
        ),
    )

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
        """Apply every operation in this segment to ``video`` in order.

        Time-based context (e.g. ``transcription``) is re-based onto this
        segment's 0-based local timeline before any operation sees it.
        """
        seg_context = _segment_context(context, self.start, self.end)
        for op in self.operations:
            video = _apply_with_context(op, video, seg_context)
        return video


class VideoEdit(BaseModel):
    """A multi-segment editing plan."""

    model_config = ConfigDict(extra="forbid")

    segments: list[SegmentConfig] = Field(
        min_length=1,
        description=(
            "Ordered list of segments. Each segment selects a time range from a "
            "source video and applies its `operations` to it; results are "
            "concatenated in order."
        ),
    )
    post_operations: list[OperationInput] = Field(
        default_factory=list,
        description="Operations applied to the concatenated output after all segments are joined.",
    )
    match_to_lowest_fps: bool = Field(
        True,
        description=(
            "When concatenating multiple segments with different fps, resample "
            "all of them to the lowest source fps. If false, mismatched fps "
            "raises during validation."
        ),
    )
    match_to_lowest_resolution: bool = Field(
        True,
        description=(
            "When concatenating multiple segments with different resolutions, "
            "resize all of them to the lowest source resolution. If false, "
            "mismatched dimensions raise during validation."
        ),
    )

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
        """LLM-facing schema: a discriminated union of operations per slot.

        A thin transform over the Pydantic models, so it cannot drift from
        them: the operations union is :meth:`Operation.json_schema` (LLM-exposed
        ops by default, per the ``llm_exposed`` ClassVar), and every other field
        shape and ``description`` is derived from ``SegmentConfig``/``VideoEdit``
        ``model_json_schema()`` rather than hand-typed. Adding a model field thus
        surfaces here automatically; in particular ``source`` carries its
        ``"format": "path"``. The Draft-07 ``$schema`` envelope and the default
        ``operations`` array shape are preserved for downstream LLM tooling.
        """
        op_schema = Operation.json_schema()

        def _field(model: type[BaseModel], field_name: str) -> dict[str, Any]:
            # Pydantic emits one self-contained property block per field
            # (description, format, minimum, ...); drop the JSON-Schema `title`
            # (a humanized field name) the hand-rolled version never carried.
            prop = dict(model.model_json_schema()["properties"][field_name])
            prop.pop("title", None)
            return prop

        def _array(model: type[BaseModel], field_name: str, items: dict[str, Any]) -> dict[str, Any]:
            # An operation-list slot: keep the model's shape/description but
            # inline the embedded op union for `items` and a `default: []` so
            # the list is optional in the LLM-facing schema.
            prop = _field(model, field_name)
            prop["items"] = items
            prop["default"] = []
            return prop

        segment_schema: dict[str, Any] = {
            "type": "object",
            "description": SegmentConfig.__doc__,
            "properties": {
                "source": _field(SegmentConfig, "source"),
                "start": _field(SegmentConfig, "start"),
                "end": _field(SegmentConfig, "end"),
                "operations": _array(SegmentConfig, "operations", op_schema),
            },
            "required": ["source", "start", "end"],
            "additionalProperties": False,
        }
        segments = _field(cls, "segments")
        segments["items"] = segment_schema
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "description": cls.__doc__,
            "properties": {
                "segments": segments,
                "post_operations": _array(cls, "post_operations", op_schema),
                "match_to_lowest_fps": _field(cls, "match_to_lowest_fps"),
                "match_to_lowest_resolution": _field(cls, "match_to_lowest_resolution"),
            },
            "required": ["segments"],
            "additionalProperties": False,
        }

    # --------------------------------------------------------------- validate

    def validate(  # type: ignore[override]
        self,
        context: dict[str, Any] | None = None,
        *,
        clamp_windows: bool = False,
    ) -> VideoMetadata:
        """Dry-run the plan via metadata. Requires source files on disk.

        Shadows Pydantic v1's deprecated ``BaseModel.validate`` classmethod;
        use ``VideoEdit.from_dict``/``model_validate`` for plan parsing.

        When ``clamp_windows`` is True, an :class:`Effect`'s ``window.stop`` that
        overruns the running predicted duration (e.g. after a duration-shrinking
        op like ``speed_change``/``cut``) is clamped to that duration -- the same
        value :meth:`Effect._resolved_window` uses at run time -- instead of
        raising. Only ``window.stop`` is clamped: a ``window.start`` past the
        duration still hard-raises (a residual divergence from ``run()``, which
        degrades it to a zero-width no-op).
        """
        source_metas = [VideoMetadata.from_path(str(seg.source)) for seg in self.segments]
        return self._validate(source_metas, context, clamp_windows=clamp_windows)

    def validate_with_metadata(
        self,
        source_metadata: VideoMetadata | dict[str, VideoMetadata],
        context: dict[str, Any] | None = None,
        *,
        clamp_windows: bool = False,
    ) -> VideoMetadata:
        """Dry-run with pre-built metadata, avoiding disk access.

        See :meth:`validate` for the ``clamp_windows`` semantics.
        """
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
        return self._validate(metas, context, clamp_windows=clamp_windows)

    def repair(
        self,
        source_metadata: VideoMetadata | dict[str, VideoMetadata],
        context: dict[str, Any] | None = None,
    ) -> tuple[VideoEdit, list[WindowClamp]]:
        """Return a copy of this plan with overrunning ``window.stop``s clamped.

        Walks the chain exactly as :meth:`_validate` does -- cut, fps/resolution
        matching, then per-op metadata prediction -- clamping each :class:`Effect`'s
        ``window.stop`` to the running predicted duration (the value
        :meth:`Effect._resolved_window` uses at run time) and recording every
        change as a :class:`WindowClamp`. Only ``window.stop`` is clamped; see
        :meth:`validate` for the ``window.start`` divergence. The returned plan is
        a deep copy -- ``self`` is left unchanged.
        """
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

        cut_metas = [
            CutSeconds(start=seg.start, end=seg.end).predict_metadata(m) for seg, m in zip(self.segments, metas)
        ]
        matched = self._apply_matching(cut_metas)

        records: list[WindowClamp] = []
        repaired_segments: list[SegmentConfig] = []
        for index, (segment, meta) in enumerate(zip(self.segments, matched)):
            seg_context = _segment_context(context, segment.start, segment.end)
            new_ops: list[Operation] = []
            for op_index, op in enumerate(segment.operations):
                clamped = _clamp_effect_window(op, meta.total_seconds)
                if clamped is not op:
                    assert isinstance(op, Effect) and op.window is not None and op.window.stop is not None
                    records.append(
                        WindowClamp(
                            location=f"segments[{index}].operations[{op_index}]",
                            field="window.stop",
                            old=op.window.stop,
                            new=meta.total_seconds,
                        )
                    )
                new_ops.append(clamped)
                meta = _predict_with_context(clamped, meta, seg_context)
            repaired_segments.append(segment.model_copy(update={"operations": new_ops}))

        repaired = self.model_copy(update={"segments": repaired_segments}, deep=True)
        return repaired, records

    def _assert_post_ops_supported(self, context: dict[str, Any] | None) -> None:
        """Reject post_operations needing time-based context on a multi-segment plan.

        ``post_operations`` run on the assembled, concatenated timeline. A
        source-absolute context value (e.g. a ``Transcription``) cannot be
        re-based across a multi-segment concat, and passing the raw value would
        silently mis-time the op (subtitles/silence-removal against the wrong
        timeline). Fail fast with an actionable message instead of producing a
        wrong render. Single-segment plans are unaffected -- their concatenated
        timeline is just the one segment's, handled by ``_segment_context``.
        """
        if len(self.segments) <= 1 or not self.post_operations:
            return
        rebaseable = _rebaseable_keys(context)
        if not rebaseable:
            return
        for op in self.post_operations:
            clash = sorted(set(op.requires) & rebaseable)
            if clash:
                raise ValueError(
                    f"post_operation '{op.op}' requires time-based context {clash}, but the plan "
                    f"has {len(self.segments)} segments. post_operations run on the concatenated "
                    "timeline and time-based context is not re-based across a multi-segment concat. "
                    f"Move '{op.op}' into a segment, or use a single-segment plan."
                )

    def _validate(
        self,
        source_metas: list[VideoMetadata],
        context: dict[str, Any] | None,
        *,
        clamp_windows: bool = False,
    ) -> VideoMetadata:
        self._assert_post_ops_supported(context)
        cut_metas: list[VideoMetadata] = []
        for i, (seg, meta) in enumerate(zip(self.segments, source_metas)):
            if seg.end > meta.total_seconds + DURATION_EPS:
                message = f"Segment {i}: end ({seg.end}) exceeds source duration ({meta.total_seconds}s)"
                raise PlanValidationError(
                    message,
                    [
                        PlanError(
                            code=PlanErrorCode.SEGMENT_END_EXCEEDS_SOURCE,
                            location=f"segments[{i}]",
                            field="end",
                            value=seg.end,
                            limit=meta.total_seconds,
                            predicted_duration=meta.total_seconds,
                        )
                    ],
                )
            try:
                cut_metas.append(CutSeconds(start=seg.start, end=seg.end).predict_metadata(meta))
            except PlanValidationError as e:
                # CutSeconds shares DURATION_EPS with the guard above, so the
                # dead-band can't fire here -- but if it ever does, carry the
                # segment index (via #4 location) instead of an index-less message.
                for err in e.errors:
                    err.location = f"segments[{i}]" if err.location is None else f"segments[{i}].{err.location}"
                raise

        matched = self._apply_matching(cut_metas)
        segment_outputs = [
            self._predict_segment(i, seg, meta, context, clamp_windows=clamp_windows)
            for i, (seg, meta) in enumerate(zip(self.segments, matched))
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
        for j, op in enumerate(self.post_operations):
            _validate_effect_window(op, assembled.total_seconds, location=f"post_operations[{j}]")
            assembled = _predict_with_context(op, assembled, context)
        return assembled

    def _predict_segment(
        self,
        index: int,
        segment: SegmentConfig,
        meta: VideoMetadata,
        context: dict[str, Any] | None,
        *,
        clamp_windows: bool = False,
    ) -> VideoMetadata:
        seg_context = _segment_context(context, segment.start, segment.end)
        for op_index, op in enumerate(segment.operations):
            location = f"segments[{index}].operations[{op_index}]"
            if clamp_windows:
                op = _clamp_effect_window(op, meta.total_seconds)
            _validate_effect_window(op, meta.total_seconds, location=location)
            try:
                meta = _predict_with_context(op, meta, seg_context)
            except PlanValidationError as e:
                # A typed error IS a ValueError, so the generic branch below would
                # flatten it back to prose and drop `.errors`. Instead enrich each
                # contained location with this segment+op index and re-raise the
                # SAME typed error, preserving its message and structure.
                for err in e.errors:
                    err.location = location if err.location is None else f"{location}.{err.location}"
                raise
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
                message = (
                    f"Segment 0 fps ({first.fps}) != segment {j} fps ({other.fps}); "
                    "all segments must share fps for concatenation."
                )
                raise PlanValidationError(
                    message,
                    [
                        PlanError(
                            code=PlanErrorCode.CONCAT_MISMATCH,
                            location=f"segments[{j}]",
                            field="fps",
                            value=other.fps,
                            limit=first.fps,
                        )
                    ],
                )
            if (first.width, first.height) != (other.width, other.height):
                message = (
                    f"Segment 0 dimensions ({first.width}x{first.height}) != "
                    f"segment {j} ({other.width}x{other.height}); all segments must share dimensions."
                )
                raise PlanValidationError(
                    message,
                    [
                        PlanError(
                            code=PlanErrorCode.CONCAT_MISMATCH,
                            location=f"segments[{j}]",
                            field="dimensions",
                        )
                    ],
                )

    # -------------------------------------------------------------------- run

    def run(self, context: dict[str, Any] | None = None) -> Video:
        """Execute the plan in memory and return the final ``Video``."""
        self._assert_post_ops_supported(context)
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
        self._assert_post_ops_supported(context)
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
                if op.requires:
                    # Same reason as the per-segment guard: no runtime context
                    # in the streaming path. (Multi-segment + requires already
                    # raised by _assert_post_ops_supported.)
                    return self._run_to_file_eager(output_path, format, preset, crf, context)
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
            if op.requires:
                # Streaming schedules effects by frame range with no runtime
                # context, so it can't supply -- let alone re-base onto the
                # segment's local timeline -- anything an op `requires`. Defer
                # to the eager path, where _segment_context handles re-basing.
                return None
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

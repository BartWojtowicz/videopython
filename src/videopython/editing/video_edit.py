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
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SerializeAsAny

from videopython.audio import Audio, AudioLoadError
from videopython.base.exceptions import PlanError, PlanErrorCode, PlanRepair, PlanValidationError
from videopython.base.video import ALLOWED_VIDEO_FORMATS, ALLOWED_VIDEO_PRESETS, Video, VideoMetadata
from videopython.editing.effects import Effect, Fade, VolumeAdjust
from videopython.editing.operation import FilterCtx, Operation, _to_strict_schema
from videopython.editing.streaming import EffectScheduleEntry, StreamingSegmentPlan, concat_files, stream_segment
from videopython.editing.transforms import DURATION_EPS, CutSeconds, Resize

__all__ = [
    "SegmentConfig",
    "VideoEdit",
]

# A located, human-readable validation failure: (message, structured PlanError).
# Helpers return these so the validation walker can either raise the first
# (``validate``, byte-stable prose) or accumulate them all (``check``).
_LocatedError = tuple[str, PlanError]


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


def _segment_end_exceeds_source(index: int, seg: SegmentConfig, meta: VideoMetadata) -> _LocatedError:
    """The located ``SEGMENT_END_EXCEEDS_SOURCE`` error for ``seg.end`` past source."""
    message = f"Segment {index}: end ({seg.end}) exceeds source duration ({meta.total_seconds}s)"
    return message, PlanError(
        code=PlanErrorCode.SEGMENT_END_EXCEEDS_SOURCE,
        location=f"segments[{index}]",
        field="end",
        value=seg.end,
        limit=meta.total_seconds,
        predicted_duration=meta.total_seconds,
    )


def _segment_bounds_errors(index: int, seg: SegmentConfig, meta: VideoMetadata) -> list[_LocatedError]:
    """Every numeric-bound failure of a segment's ``start``/``end``, in order.

    These checks moved off the (now permissive) ``SegmentConfig`` model so a
    bad range becomes a collectable, repairable :class:`PlanError` instead of a
    hard ``from_dict`` failure. Order is the raise order for ``validate``:
    negative ``start``, negative ``end``, ``end <= start``, then ``end`` past
    the source. A segment with any of these cannot be cut, so the caller skips
    its op chain.
    """
    out: list[_LocatedError] = []
    loc = f"segments[{index}]"
    if seg.start < 0:
        out.append(
            (
                f"Segment {index}: start ({seg.start}) must be >= 0",
                PlanError(code=PlanErrorCode.SEGMENT_NEGATIVE, location=loc, field="start", value=seg.start, limit=0.0),
            )
        )
    if seg.end < 0:
        out.append(
            (
                f"Segment {index}: end ({seg.end}) must be >= 0",
                PlanError(code=PlanErrorCode.SEGMENT_NEGATIVE, location=loc, field="end", value=seg.end, limit=0.0),
            )
        )
    if seg.end <= seg.start:
        out.append(
            (
                f"Segment {index}: end ({seg.end}) must be greater than start ({seg.start})",
                PlanError(code=PlanErrorCode.SEGMENT_RANGE, location=loc, field="end", value=seg.end, limit=seg.start),
            )
        )
    if seg.end > meta.total_seconds + DURATION_EPS:
        out.append(_segment_end_exceeds_source(index, seg, meta))
    return out


def _window_errors(op: Operation, duration: float, location: str) -> list[_LocatedError]:
    """Every bound failure of an :attr:`Effect.window`, in raise order.

    Subsumes the old window-vs-duration check and the negative/order checks that
    moved off the (now permissive) ``TimeRange`` model: negative ``start``,
    negative ``stop``, ``stop < start``, ``start`` past duration, ``stop`` past
    duration. A ``stop`` clamped to ``duration`` by ``clamp_windows`` no longer
    overruns, so the final check stays silent for it (matching ``run()``).
    """
    if not isinstance(op, Effect) or op.window is None:
        return []
    out: list[_LocatedError] = []
    start, stop = op.window.start, op.window.stop
    eps = DURATION_EPS
    if start is not None and start < 0:
        out.append(
            (
                f"Effect '{op.op}' window.start ({start}) must be >= 0",
                PlanError(PlanErrorCode.WINDOW_NEGATIVE, location, op.op, "window.start", start, 0.0),
            )
        )
    if stop is not None and stop < 0:
        out.append(
            (
                f"Effect '{op.op}' window.stop ({stop}) must be >= 0",
                PlanError(PlanErrorCode.WINDOW_NEGATIVE, location, op.op, "window.stop", stop, 0.0),
            )
        )
    if start is not None and stop is not None and stop < start:
        out.append(
            (
                f"Effect '{op.op}' window.stop ({stop}) must be >= start ({start})",
                PlanError(PlanErrorCode.WINDOW_ORDER, location, op.op, "window.stop", stop, start),
            )
        )
    if start is not None and start > duration + eps:
        out.append(
            (
                f"Effect '{op.op}' window.start ({start}) exceeds duration ({duration}s)",
                PlanError(
                    PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION,
                    location,
                    op.op,
                    "window.start",
                    start,
                    duration,
                    duration,
                ),
            )
        )
    if stop is not None and stop > duration + eps:
        out.append(
            (
                f"Effect '{op.op}' window.stop ({stop}) exceeds duration ({duration}s)",
                PlanError(
                    PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION,
                    location,
                    op.op,
                    "window.stop",
                    stop,
                    duration,
                    duration,
                ),
            )
        )
    return out


def _concat_errors(metas: list[VideoMetadata]) -> list[_LocatedError]:
    """``CONCAT_MISMATCH`` errors for fps/dimension divergence across segments."""
    if len(metas) <= 1:
        return []
    out: list[_LocatedError] = []
    first = metas[0]
    for j, other in enumerate(metas[1:], start=1):
        if first.fps != other.fps:
            out.append(
                (
                    f"Segment 0 fps ({first.fps}) != segment {j} fps ({other.fps}); "
                    "all segments must share fps for concatenation.",
                    PlanError(
                        PlanErrorCode.CONCAT_MISMATCH,
                        location=f"segments[{j}]",
                        field="fps",
                        value=other.fps,
                        limit=first.fps,
                    ),
                )
            )
        if (first.width, first.height) != (other.width, other.height):
            out.append(
                (
                    f"Segment 0 dimensions ({first.width}x{first.height}) != "
                    f"segment {j} ({other.width}x{other.height}); all segments must share dimensions.",
                    PlanError(PlanErrorCode.CONCAT_MISMATCH, location=f"segments[{j}]", field="dimensions"),
                )
            )
    return out


def _clamp_effect_window(op: Operation, duration: float) -> Operation:
    """Return ``op`` with its ``Effect.window.stop`` clamped to ``duration``.

    Mirrors :meth:`Effect._resolved_window`'s run-time ``min(stop, total_seconds)``
    so a stop overrunning a duration-shrunk chain validates instead of raising.
    This is the narrow ``clamp_windows`` repair: ``window.start`` and negative
    bounds are deliberately left untouched (still reported by ``_window_errors``).
    :meth:`VideoEdit.repair` uses the broader :func:`_repair_effect_window`.
    """
    if not isinstance(op, Effect) or op.window is None or op.window.stop is None:
        return op
    if op.window.stop <= duration:
        return op
    clamped_window = op.window.model_copy(update={"stop": duration})
    return op.model_copy(update={"window": clamped_window})


def _repair_effect_window(op: Operation, duration: float) -> tuple[Operation, list[PlanRepair]]:
    """Clamp an :attr:`Effect.window`'s ``start``/``stop`` into ``[0, duration]``.

    The repair counterpart to :func:`_window_errors`: a negative endpoint snaps
    to ``0`` (``WINDOW_NEGATIVE``); one past the duration snaps to ``duration``
    (``EFFECT_WINDOW_EXCEEDS_DURATION``). Returns the (possibly new) op and a
    :class:`PlanRepair` per change with an empty ``location`` the caller fills
    in. A ``stop < start`` left after clamping is deliberately not invented away
    -- it stays for ``check``/``validate`` to report.
    """
    if not isinstance(op, Effect) or op.window is None:
        return op, []
    start, stop = op.window.start, op.window.stop
    new_start, new_stop = start, stop
    changes: list[PlanRepair] = []
    if start is not None and start < 0:
        new_start = 0.0
        changes.append(PlanRepair("", "window.start", start, 0.0, PlanErrorCode.WINDOW_NEGATIVE))
    elif start is not None and start > duration:
        new_start = duration
        changes.append(PlanRepair("", "window.start", start, duration, PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION))
    if stop is not None and stop < 0:
        new_stop = 0.0
        changes.append(PlanRepair("", "window.stop", stop, 0.0, PlanErrorCode.WINDOW_NEGATIVE))
    elif stop is not None and stop > duration:
        new_stop = duration
        changes.append(PlanRepair("", "window.stop", stop, duration, PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION))
    if not changes:
        return op, []
    new_window = op.window.model_copy(update={"start": new_start, "stop": new_stop})
    return op.model_copy(update={"window": new_window}), changes


def _clamp_time_fields(op: Operation, meta: VideoMetadata) -> tuple[Operation, list[PlanRepair]]:
    """Clamp every declared :attr:`Operation.time_fields` value into range.

    Generic over the op's :class:`BoundedTimeField` declarations: the lower
    bound is ``0``; the upper is the clip duration, or -- for ``exclusive_end``
    fields that index a frame (``freeze_frame.timestamp``) -- the last
    addressable frame ``(frame_count - 1) / fps``. Only an *out-of-range* value
    is touched (and recorded as an ``OP_TIMESTAMP_OUT_OF_RANGE``
    :class:`PlanRepair` with a caller-filled ``location``); an in-range value is
    left exactly as written -- no rounding, no phantom changelog entry.
    """
    changes: list[PlanRepair] = []
    new_op = op
    for tf in op.time_fields:
        value = getattr(new_op, tf.name)
        if value is None:
            continue
        upper = max(0.0, (meta.frame_count - 1) / meta.fps if tf.exclusive_end else meta.total_seconds)
        if value < 0 or value > upper:
            clamped = round(min(max(value, 0.0), upper), 4)
            new_op = new_op.model_copy(update={tf.name: clamped})
            changes.append(PlanRepair("", tf.name, value, clamped, PlanErrorCode.OP_TIMESTAMP_OUT_OF_RANGE))
    return new_op, changes


def _repair_op_chain(
    ops: list[Operation],
    meta: VideoMetadata,
    context: dict[str, Any] | None,
    location_prefix: str,
    *,
    clamp: bool,
) -> tuple[list[Operation], list[PlanRepair], VideoMetadata | None]:
    """Clamp an op chain's windows / time-fields, predicting forward to track duration.

    Shared by :meth:`VideoEdit.repair` for both a segment's ``operations``
    (``location_prefix='segments[i].operations'``) and the plan's
    ``post_operations`` (``location_prefix='post_operations'``). Each op (while
    the running duration is still known) gets :func:`_repair_effect_window` +
    :func:`_clamp_time_fields` applied, with ``f'{location_prefix}[op_index]'``
    stamped onto every :class:`PlanRepair`. The first op that cannot be
    predicted ends clamping -- the rest are kept verbatim (we can't know the
    duration past it) and the returned running meta is ``None`` so the caller
    knows the chain broke.
    """
    new_ops: list[Operation] = []
    repairs: list[PlanRepair] = []
    running: VideoMetadata | None = meta
    for op_index, op in enumerate(ops):
        if clamp and running is not None:
            location = f"{location_prefix}[{op_index}]"
            op, win_changes = _repair_effect_window(op, running.total_seconds)
            op, time_changes = _clamp_time_fields(op, running)
            for change in (*win_changes, *time_changes):
                change.location = location
                repairs.append(change)
        new_ops.append(op)
        if running is None:
            continue
        try:
            running = _predict_with_context(op, running, context)
        except (ValueError, TypeError):
            running = None
    return new_ops, repairs, running


class SegmentConfig(BaseModel):
    """A single source segment with its operation chain."""

    model_config = ConfigDict(extra="forbid")

    source: Path = Field(description="Path to the source video file.")
    # Parsing is permissive (plain floats, no ge=0 / end>start): the numeric
    # bounds are owned by validate/check/repair as structured PlanErrors. See
    # `_segment_bounds_errors` and the VideoEdit class docstring.
    start: float = Field(description="Segment start time in seconds.")
    end: float = Field(description="Segment end time in seconds.")
    operations: list[OperationInput] = Field(
        default_factory=list,
        description=(
            "Ordered list of operations to run against this segment. "
            "Each item is an Operation discriminated by its `op` field."
        ),
    )

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
    """A multi-segment editing plan.

    **Parse vs. validate.** Parsing (``from_dict``/``model_validate``) owns the
    *shape*: field types, required fields, unknown-op/extra-field rejection, and
    op-local structural rules (e.g. ``resize`` needs a dimension) surface as a
    Pydantic ``ValidationError``. The numeric *bounds* of the plan skeleton --
    segment ``start``/``end`` and effect ``window`` ranges -- are deliberately
    **not** enforced at parse; they are owned by :meth:`validate` / :meth:`check`
    / :meth:`repair`, which report them as structured :class:`PlanError`s. This
    keeps one code path for the LLM refine loop: ``from_dict`` (permissive) ->
    :meth:`repair` (clamp the mechanical ones) -> :meth:`check` (collect whatever
    remains) -> re-prompt with the full structured error list.
    """

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
    def json_schema(cls, *, strict: bool = False) -> dict[str, Any]:
        """LLM-facing schema: a discriminated union of operations per slot.

        A thin transform over the Pydantic models, so it cannot drift from
        them: the operations union is :meth:`Operation.json_schema` (LLM-exposed
        ops by default, per the ``llm_exposed`` ClassVar), and every other field
        shape and ``description`` is derived from ``SegmentConfig``/``VideoEdit``
        ``model_json_schema()`` rather than hand-typed. Adding a model field thus
        surfaces here automatically; in particular ``source`` carries its
        ``"format": "path"``. The Draft-07 ``$schema`` envelope and the default
        ``operations`` array shape are preserved for downstream LLM tooling.

        With ``strict=True`` the result is a submittable provider strict-mode
        grammar: a closed object root, all properties ``required`` (optionality
        kept as Pydantic emitted it -- no synthesized nulls), the op union as
        ``anyOf`` of closed variants, and the union's ``$defs`` hoisted to the
        document root so every ``$ref`` resolves. See :meth:`Operation.json_schema`
        for the strict-mode contract. Use it as a ``response_format: json_schema``
        grammar so simple bound violations (``window.start >= 0``, enums, required
        fields) become impossible at decode time. Cross-field constraints
        (``timestamp < duration``, segment-dim equality) cannot live in a grammar
        and stay with :meth:`check` / :meth:`repair` / :meth:`normalize_dimensions`.
        """
        # Build the permissive envelope, then (if strict) run one closing pass
        # over the whole thing -- including the embedded op union -- so the
        # hand-built segment/top-level objects are closed and required too.
        # `op_schema` is a self-contained union carrying its own root `$defs`;
        # the same object is inlined as the `items` of both operation slots.
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
        schema = {
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
        if not strict:
            return schema
        # Inlining the op union as `items` buried its `$defs` while its `$ref`s
        # stayed root-relative (`#/$defs/X`), so they dangled. Hoist the (shared)
        # defs to this schema's root before the strict pass, so every ref
        # resolves and the result is a submittable provider grammar.
        op_defs = op_schema.pop("$defs", None)
        if op_defs:
            schema["$defs"] = op_defs
        return _to_strict_schema(schema)

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
        metas = self._resolve_source_metas(source_metadata)
        return self._validate(metas, context, clamp_windows=clamp_windows)

    def check(
        self,
        source_metadata: VideoMetadata | dict[str, VideoMetadata],
        context: dict[str, Any] | None = None,
        *,
        clamp_windows: bool = False,
    ) -> list[PlanError]:
        """Collect **every** plan error in one pass; ``[]`` means valid.

        The non-raising sibling of :meth:`validate_with_metadata`: it runs the
        same dry-run but accumulates instead of aborting on the first failure,
        so an LLM refine loop can fix all problems in a single re-prompt instead
        of playing whack-a-mole across a retry budget. Best-effort: each segment
        is checked against its own source metadata, per-op and per-segment errors
        collected; a check that cannot run because an earlier one failed (a
        segment's op chain past a bad cut, the cross-segment concat check when a
        segment did not produce an output) is skipped rather than aborting.

        Returns the same :class:`PlanError` list that
        :attr:`PlanValidationError.errors` carries -- every failure is structured
        (no bare ``ValueError`` escapes the walk), so a consumer branches on
        ``code`` rather than substring-matching prose. ``clamp_windows`` matches
        :meth:`validate`: a clampable ``window.stop`` overrun is not reported.
        """
        metas = self._resolve_source_metas(source_metadata)
        _, errors = self._collect(metas, context, clamp_windows=clamp_windows, stop_first=False)
        return errors

    def repair(
        self,
        source_metadata: VideoMetadata | dict[str, VideoMetadata],
        context: dict[str, Any] | None = None,
        *,
        clamp_op_params: bool = True,
        clamp_segment_end: bool = False,
    ) -> tuple[VideoEdit, list[PlanRepair]]:
        """Return a copy of this plan with the *unambiguous* violations clamped.

        Walks the chain (cut, fps/resolution matching, per-op prediction) and
        clamps only the mechanical faults whose fix is not a judgement call,
        recording each as a :class:`PlanRepair`. The returned plan is a deep copy
        (``self`` is untouched); the changelog is meant to be surfaced to the
        user ("we trimmed your effect to fit"). ``repair`` never *invents* intent
        -- genuinely semantic problems (a concat dimension mismatch, an
        ``end <= start`` range) are left for :meth:`check` / re-prompting.

        With ``clamp_op_params`` (default ``True``) it clamps each effect
        ``window.start``/``window.stop`` into ``[0, duration]`` and each declared
        :attr:`Operation.time_fields` value (e.g. ``freeze_frame.timestamp`` past
        the clip end) into range, plus a negative segment ``start`` to ``0``.
        With ``clamp_segment_end`` (default ``False``, since it changes editorial
        intent) it also clamps a segment ``end`` past the source to the source
        end; left ``False``, that case hard-raises as before. Always
        :meth:`check` / :meth:`validate` the returned plan before running it.
        """
        metas = self._resolve_source_metas(source_metadata)
        repairs: list[PlanRepair] = []

        # Pass 1: segment-level start/end repairs, and the cut metadata each
        # surviving segment's op clamps run against.
        fixed_segments: list[SegmentConfig] = []
        cut_metas: list[VideoMetadata | None] = []
        for i, (seg, meta) in enumerate(zip(self.segments, metas)):
            start, end = seg.start, seg.end
            if clamp_op_params and start < 0:
                repairs.append(PlanRepair(f"segments[{i}]", "start", start, 0.0, PlanErrorCode.SEGMENT_NEGATIVE))
                start = 0.0
            if end > meta.total_seconds + DURATION_EPS:
                if clamp_segment_end:
                    new_end = round(meta.total_seconds, 4)
                    repairs.append(
                        PlanRepair(f"segments[{i}]", "end", end, new_end, PlanErrorCode.SEGMENT_END_EXCEEDS_SOURCE)
                    )
                    end = new_end
                else:
                    message, err = _segment_end_exceeds_source(i, seg, meta)
                    raise PlanValidationError(message, [err])
            seg = seg.model_copy(update={"start": start, "end": end})
            fixed_segments.append(seg)
            repairable = start >= 0 and end > start and end <= meta.total_seconds + DURATION_EPS
            cut_metas.append(CutSeconds(start=start, end=end).predict_metadata(meta) if repairable else None)

        # Pass 2: per-segment op clamps against the matched cut meta. Capture each
        # segment's predicted output -- needed to assemble the post-op timeline.
        matched_by_index = self._match_cuttable(cut_metas)
        final_segments: list[SegmentConfig] = []
        seg_output_metas: list[VideoMetadata | None] = []
        for i, seg in enumerate(fixed_segments):
            seg_meta = matched_by_index.get(i)
            if seg_meta is None:
                final_segments.append(seg)
                seg_output_metas.append(None)
                continue
            seg_context = _segment_context(context, seg.start, seg.end)
            new_ops, op_repairs, out_meta = _repair_op_chain(
                list(seg.operations),
                seg_meta,
                seg_context,
                f"segments[{i}].operations",
                clamp=clamp_op_params,
            )
            repairs.extend(op_repairs)
            final_segments.append(seg.model_copy(update={"operations": new_ops}))
            seg_output_metas.append(out_meta)

        # Pass 3: clamp post-op windows / time-fields against the assembled
        # timeline -- but only when every segment predicted (else its duration is
        # unknown). A negative post_operations window is the canonical attempt-3
        # failure, so this matters as much as the per-segment clamps.
        update: dict[str, Any] = {"segments": final_segments}
        outputs = [m for m in seg_output_metas if m is not None]
        if self.post_operations and len(outputs) == len(self.segments):
            first = outputs[0]
            assembled = VideoMetadata(
                height=first.height,
                width=first.width,
                fps=first.fps,
                frame_count=sum(m.frame_count for m in outputs),
                total_seconds=round(sum(m.total_seconds for m in outputs), 4),
            )
            new_post, post_repairs, _ = _repair_op_chain(
                list(self.post_operations),
                assembled,
                context,
                "post_operations",
                clamp=clamp_op_params,
            )
            repairs.extend(post_repairs)
            update["post_operations"] = new_post

        repaired = self.model_copy(update=update, deep=True)
        return repaired, repairs

    def normalize_dimensions(
        self,
        source_metadata: VideoMetadata | dict[str, VideoMetadata],
        target: tuple[int, int] | Literal["first", "largest"],
        context: dict[str, Any] | None = None,
    ) -> tuple[VideoEdit, list[PlanRepair]]:
        """Make every segment concat-compatible by resizing to a common canvas.

        ``CONCAT_MISMATCH`` is the one class a consumer cannot cleanly repair in
        its own layer: detecting it needs each segment's *predicted post-op*
        dimensions, and fixing it needs a per-segment resize inserted **before**
        concat. videopython owns both, so it does it here: predict each segment's
        output dimensions, pick the ``target`` -- an explicit ``(width, height)``,
        ``"first"`` (the first predictable segment's output), or ``"largest"``
        (greatest area) -- and append a ``resize`` op to every segment whose
        output differs, recording a :class:`PlanRepair` per insertion. The
        returned plan satisfies the "all segments share dimensions" invariant for
        every segment it could predict.

        Best-effort and non-raising, matching :meth:`repair` / :meth:`check`: a
        segment that cannot be cut (bad range) or whose op chain fails prediction
        is left untouched and its fault deferred to :meth:`check`, rather than
        aborting the whole call. This keeps the documented refine flow
        (``repair -> normalize_dimensions -> check``) a single non-raising path.
        When no segment is predictable the plan is returned unchanged with an
        empty changelog.

        Expressed purely as appended ``resize`` ops, so the normal
        validate/run/stream paths need no special casing. Resizing to an exact
        canvas can distort aspect when segments genuinely differ -- intended for
        a plan whose segments already share a target aspect (resolve that
        upstream).
        """
        metas = self._resolve_source_metas(source_metadata)
        # Predict each segment's post-op output dims; None when it can't be cut
        # or an op fails prediction (best-effort: deferred to check, never raised).
        cut_metas: list[VideoMetadata | None] = []
        for i, (seg, meta) in enumerate(zip(self.segments, metas)):
            if _segment_bounds_errors(i, seg, meta):
                cut_metas.append(None)
            else:
                cut_metas.append(CutSeconds(start=seg.start, end=seg.end).predict_metadata(meta))
        matched_by_index = self._match_cuttable(cut_metas)
        outputs: list[VideoMetadata | None] = []
        for i, seg in enumerate(self.segments):
            seg_meta = matched_by_index.get(i)
            if seg_meta is None:
                outputs.append(None)
                continue
            try:
                outputs.append(self._predict_segment(i, seg, seg_meta, context))
            except (ValueError, TypeError):
                outputs.append(None)

        known = [m for m in outputs if m is not None]
        if not known:
            return self.model_copy(deep=True), []
        if target == "first":
            tw, th = known[0].width, known[0].height
        elif target == "largest":
            biggest = max(known, key=lambda m: m.width * m.height)
            tw, th = biggest.width, biggest.height
        else:
            tw, th = target

        repairs: list[PlanRepair] = []
        new_segments: list[SegmentConfig] = []
        for i, (seg, out) in enumerate(zip(self.segments, outputs)):
            if out is None or (out.width, out.height) == (tw, th):
                new_segments.append(seg)
                continue
            new_ops = [*seg.operations, Resize(width=tw, height=th)]
            repairs.append(
                PlanRepair(
                    location=f"segments[{i}]",
                    field="dimensions",
                    old=f"{out.width}x{out.height}",
                    new=f"{tw}x{th}",
                    code=PlanErrorCode.CONCAT_MISMATCH,
                )
            )
            new_segments.append(seg.model_copy(update={"operations": new_ops}))

        normalized = self.model_copy(update={"segments": new_segments}, deep=True)
        return normalized, repairs

    def _resolve_source_metas(self, source_metadata: VideoMetadata | dict[str, VideoMetadata]) -> list[VideoMetadata]:
        """One ``VideoMetadata`` per segment from a single meta or a by-source dict."""
        if isinstance(source_metadata, VideoMetadata):
            return [source_metadata for _ in self.segments]
        metas: list[VideoMetadata] = []
        for i, seg in enumerate(self.segments):
            key = str(seg.source)
            if key not in source_metadata:
                available = sorted(source_metadata)
                raise ValueError(f"Segment {i}: no metadata for '{key}'. Available: {available}")
            metas.append(source_metadata[key])
        return metas

    def _post_op_context_errors(self, context: dict[str, Any] | None) -> list[_LocatedError]:
        """``POST_OP_REQUIRES_CONTEXT`` errors for time-context post-ops on a multi-segment plan.

        ``post_operations`` run on the assembled, concatenated timeline. A
        source-absolute context value (e.g. a ``Transcription``) cannot be
        re-based across a multi-segment concat, and passing the raw value would
        silently mis-time the op (subtitles/silence-removal against the wrong
        timeline). Single-segment plans are unaffected -- their concatenated
        timeline is just the one segment's, handled by ``_segment_context``.
        """
        if len(self.segments) <= 1 or not self.post_operations:
            return []
        rebaseable = _rebaseable_keys(context)
        if not rebaseable:
            return []
        out: list[_LocatedError] = []
        for j, op in enumerate(self.post_operations):
            clash = sorted(set(op.requires) & rebaseable)
            if clash:
                message = (
                    f"post_operation '{op.op}' requires time-based context {clash}, but the plan "
                    f"has {len(self.segments)} segments. post_operations run on the concatenated "
                    "timeline and time-based context is not re-based across a multi-segment concat. "
                    f"Move '{op.op}' into a segment, or use a single-segment plan."
                )
                out.append(
                    (
                        message,
                        PlanError(PlanErrorCode.POST_OP_REQUIRES_CONTEXT, location=f"post_operations[{j}]", op=op.op),
                    )
                )
        return out

    def _assert_post_ops_supported(self, context: dict[str, Any] | None) -> None:
        """Raising guard used by ``run``/``run_to_file`` (see :meth:`_post_op_context_errors`)."""
        errs = self._post_op_context_errors(context)
        if errs:
            message, err = errs[0]
            raise PlanValidationError(message, [err])

    def _validate(
        self,
        source_metas: list[VideoMetadata],
        context: dict[str, Any] | None,
        *,
        clamp_windows: bool = False,
    ) -> VideoMetadata:
        assembled, _ = self._collect(source_metas, context, clamp_windows=clamp_windows, stop_first=True)
        assert assembled is not None  # stop_first raised on any error, so a result is guaranteed
        return assembled

    def _collect(
        self,
        source_metas: list[VideoMetadata],
        context: dict[str, Any] | None,
        *,
        clamp_windows: bool,
        stop_first: bool,
    ) -> tuple[VideoMetadata | None, list[PlanError]]:
        """The single dry-run walker behind both :meth:`validate` and :meth:`check`.

        ``stop_first=True`` raises ``PlanValidationError`` on the first failure
        (byte-stable prose for existing callers); ``stop_first=False``
        accumulates every structured error and returns them. The check *order* is
        identical in both modes -- post-op context guard, per-segment cut, fps/res
        matching, per-segment op prediction, cross-segment concat, post-ops -- so
        the first collected error always matches what ``validate`` would raise.
        """
        errors: list[PlanError] = []

        def emit(message: str, err: PlanError) -> None:
            if stop_first:
                raise PlanValidationError(message, [err])
            errors.append(err)

        def raise_or_collect(exc: PlanValidationError) -> None:
            if stop_first:
                raise exc
            errors.extend(exc.errors)

        for message, err in self._post_op_context_errors(context):
            emit(message, err)

        # Cut each segment against its own source metadata. A bad range isolates
        # that segment (no cut meta -> its op chain is skipped below).
        cut_metas: list[VideoMetadata | None] = []
        for i, (seg, meta) in enumerate(zip(self.segments, source_metas)):
            bounds = _segment_bounds_errors(i, seg, meta)
            if bounds:
                for message, err in bounds:
                    emit(message, err)
                cut_metas.append(None)
                continue
            cut_metas.append(CutSeconds(start=seg.start, end=seg.end).predict_metadata(meta))

        matched_by_index = self._match_cuttable(cut_metas)

        # `_apply_matching` runs over the cuttable subset only; an uncuttable
        # segment isolates and is already reported, so the (now-invalid) plan's
        # matched dims may differ from run()'s all-segments matching -- harmless,
        # since that plan can't run until the bad segment is fixed.
        seg_outputs: dict[int, VideoMetadata] = {}
        for i, seg in enumerate(self.segments):
            seg_meta = matched_by_index.get(i)
            if seg_meta is None:
                continue
            seg_context = _segment_context(context, seg.start, seg.end)
            failed = False
            for op_index, op in enumerate(seg.operations):
                location = f"segments[{i}].operations[{op_index}]"
                if clamp_windows:
                    op = _clamp_effect_window(op, seg_meta.total_seconds)
                for message, err in _window_errors(op, seg_meta.total_seconds, location):
                    emit(message, err)
                try:
                    seg_meta = _predict_with_context(op, seg_meta, seg_context)
                except PlanValidationError as e:
                    for perr in e.errors:
                        perr.location = location if perr.location is None else f"{location}.{perr.location}"
                    raise_or_collect(e)
                    failed = True
                    break
                except (ValueError, TypeError) as e:
                    message = f"Segment {i}: metadata prediction failed for '{op.op}': {e}"
                    emit(message, PlanError(PlanErrorCode.OP_PREDICTION_FAILED, location=location, op=op.op))
                    failed = True
                    break
            if not failed:
                seg_outputs[i] = seg_meta

        # Concat + post-ops need every segment's predicted output. When a segment
        # failed, those dependent checks are skipped (best-effort): the consumer
        # fixes the isolated segment and they surface on the next pass.
        if len(seg_outputs) != len(self.segments):
            return None, errors
        outputs = [seg_outputs[i] for i in range(len(self.segments))]
        for message, err in _concat_errors(outputs):
            emit(message, err)

        first = outputs[0]
        assembled = VideoMetadata(
            height=first.height,
            width=first.width,
            fps=first.fps,
            frame_count=sum(m.frame_count for m in outputs),
            total_seconds=round(sum(m.total_seconds for m in outputs), 4),
        )
        for j, op in enumerate(self.post_operations):
            location = f"post_operations[{j}]"
            for message, err in _window_errors(op, assembled.total_seconds, location):
                emit(message, err)
            try:
                assembled = _predict_with_context(op, assembled, context)
            except PlanValidationError as e:
                for perr in e.errors:
                    perr.location = location if perr.location is None else f"{location}.{perr.location}"
                raise_or_collect(e)
                return None, errors
            except (ValueError, TypeError) as e:
                message = f"post_operations[{j}]: metadata prediction failed for '{op.op}': {e}"
                emit(message, PlanError(PlanErrorCode.OP_PREDICTION_FAILED, location=location, op=op.op))
                return None, errors
        return assembled, errors

    def _predict_segment(
        self,
        index: int,
        segment: SegmentConfig,
        meta: VideoMetadata,
        context: dict[str, Any] | None,
    ) -> VideoMetadata:
        """Predict one segment's post-op metadata, raising located plan errors.

        The always-raising single-segment predictor used by
        :meth:`normalize_dimensions` (which needs each segment's predicted output
        dimensions). :meth:`_collect` has its own per-op loop because it must
        also accumulate, clamp windows, and isolate failures per segment.
        """
        seg_context = _segment_context(context, segment.start, segment.end)
        for op_index, op in enumerate(segment.operations):
            location = f"segments[{index}].operations[{op_index}]"
            for message, err in _window_errors(op, meta.total_seconds, location):
                raise PlanValidationError(message, [err])
            try:
                meta = _predict_with_context(op, meta, seg_context)
            except PlanValidationError as e:
                for err in e.errors:
                    err.location = location if err.location is None else f"{location}.{err.location}"
                raise
            except (ValueError, TypeError) as e:
                raise ValueError(f"Segment {index}: metadata prediction failed for '{op.op}': {e}") from e
        return meta

    def _match_cuttable(self, cut_metas: list[VideoMetadata | None]) -> dict[int, VideoMetadata]:
        """Run fps/resolution matching over the cuttable segments only, by original index.

        An uncuttable segment (a ``None`` cut meta -- a bad range, isolated and
        reported elsewhere) is excluded so it cannot perturb the ``min`` fps/dims
        that :meth:`_apply_matching` derives. Returns ``{original_index: matched
        meta}`` so callers stay index-aligned with ``self.segments``. Shared by
        :meth:`_collect`, :meth:`repair`, and :meth:`normalize_dimensions`.
        """
        present = [(i, m) for i, m in enumerate(cut_metas) if m is not None]
        return dict(zip((i for i, _ in present), self._apply_matching([m for _, m in present])))

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

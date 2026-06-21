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

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, get_args, runtime_checkable

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SerializeAsAny

from videopython.base import _ffmpeg
from videopython.base.exceptions import PlanError, PlanErrorCode, PlanRepair, PlanValidationError
from videopython.base.video import ALLOWED_VIDEO_FORMATS, ALLOWED_VIDEO_PRESETS, Video, VideoMetadata
from videopython.editing._schema import array_field_schema, field_schema, optional_model_field_schema
from videopython.editing.audio_ops import MusicBed, build_music_bed_filter_complex
from videopython.editing.effects import Effect
from videopython.editing.operation import FilterCtx, Operation, _to_strict_schema
from videopython.editing.streaming import (
    TRANSITION_TYPES,
    EffectScheduleEntry,
    StreamabilityReport,
    StreamingSegmentPlan,
    analyze_streamability,
    concat_files,
    source_has_audio_stream,
    stream_segment,
    stream_transition_pair,
)
from videopython.editing.transforms import DURATION_EPS, CutSeconds, Resize, speech_windows

__all__ = [
    "MusicBed",
    "SegmentConfig",
    "TransitionSpec",
    "VideoEdit",
]

# A Literal mirroring streaming.TRANSITION_TYPES so the LLM/strict grammar is
# constrained to the curated catalog. Asserted equal to TRANSITION_TYPES below
# so the two cannot drift.
TransitionType = Literal[
    "fade",
    "dissolve",
    "wipeleft",
    "wiperight",
    "wipeup",
    "wipedown",
    "slideleft",
    "slideright",
]
assert set(get_args(TransitionType)) == set(TRANSITION_TYPES), "TransitionType Literal drifted from TRANSITION_TYPES"

# A located, human-readable validation failure: (message, structured PlanError).
# Helpers return these so the validation walker can either raise the first
# (``validate``, byte-stable prose) or accumulate them all (``check``).
_LocatedError = tuple[str, PlanError]

# Sentinel marking "caller did not pass decode_filters" so a FilterCtx builder
# can fall back to FilterCtx's own default instead of forcing a value.
_DECODE_FILTERS_DEFAULT = object()


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
    """Context keys whose value carries a re-baseable source-absolute timeline.

    Recognizes both a bare rebaseable value and a per-source ``dict`` map (see
    :func:`_resolve_source_context`) whose values are all rebaseable.
    """
    if not context:
        return set()
    keys: set[str] = set()
    for k, v in context.items():
        if isinstance(v, SegmentRebaseable):
            keys.add(k)
        elif isinstance(v, dict) and v and all(isinstance(sv, SegmentRebaseable) for sv in v.values()):
            keys.add(k)
    return keys


def _resolve_source_context(context: dict[str, Any] | None, source: str) -> dict[str, Any] | None:
    """Collapse per-source context maps to a single segment source's values.

    A context value that is a plain ``dict`` is treated as a per-source map
    keyed by ``str(segment.source)`` -- mirroring
    :meth:`VideoEdit._resolve_source_metas`, so runtime metadata and runtime
    context share one mental model. Its entry for ``source`` is selected; when
    ``source`` is absent the key is dropped so the consuming op raises its own
    clear "requires ..." error (or surfaces as ``CONTEXT_SOURCE_MISSING`` in
    :meth:`VideoEdit.check`). Any non-dict value is a broadcast value shared by
    every segment and passes through unchanged (the pre-0.43 behavior).
    """
    if not context:
        return context
    if not any(isinstance(v, dict) for v in context.values()):
        return context  # no per-source maps -> nothing to resolve, keep identity
    resolved: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(value, dict):
            if source in value:
                resolved[key] = value[source]
            # else: drop -- this segment's source is not in the per-source map.
        else:
            resolved[key] = value
    return resolved


def _segment_context(
    context: dict[str, Any] | None,
    source: str,
    start: float,
    end: float,
) -> dict[str, Any] | None:
    """Resolve per-source context, then re-base it onto a cut segment's local timeline.

    Two stages, run at the single chokepoint every per-segment site funnels
    through:

    1. **Per-source resolution** (:func:`_resolve_source_context`): a context
       value may be a per-source ``dict`` keyed by ``str(segment.source)``;
       it collapses to this segment's source. A bare value broadcasts to all
       segments (the pre-0.43 behavior). This lets a multi-clip plan carry
       ``{"transcription": {"a.mp4": tx_a, "b.mp4": tx_b}}`` and feed each
       segment its OWN transcription.
    2. **Re-basing**: a cut segment is decoded 0-based -- its first frame is
       ``t=0`` -- but context values may carry source-absolute timestamps.
       Every value implementing :class:`SegmentRebaseable` (e.g. a
       ``Transcription``) is sliced to ``[start, end)`` and shifted by
       ``-start`` so segment operations (``add_subtitles``,
       ``silence_removal``) see segment-local time. Without this, subtitles on
       a segment cut from the middle of a video render blank. Values that don't
       implement the protocol pass through untouched.

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
    resolved = _resolve_source_context(context, source)
    if not resolved:
        return resolved
    rebaseable = {k: v for k, v in resolved.items() if isinstance(v, SegmentRebaseable)}
    if not rebaseable:
        return resolved
    rebased = dict(resolved)
    for key, value in rebaseable.items():
        sliced = value.slice(start, end)
        if sliced is None:
            del rebased[key]
        else:
            rebased[key] = sliced.offset(-start)
    return rebased


def _missing_source_context_errors(
    context: dict[str, Any] | None,
    index: int,
    segment: SegmentConfig,
) -> list[_LocatedError]:
    """``CONTEXT_SOURCE_MISSING`` errors for ops needing a per-source key that omits this source.

    Fires only for the new failure mode P1.9 introduces: a per-source context
    map (a plain ``dict``) is supplied for a key an op ``requires``, but it has
    no entry for this segment's source. A bare broadcast value or a fully
    absent key is left to the existing missing-context path. Surfacing it in
    :meth:`VideoEdit.check` matters because an ``add_subtitles`` plan would
    otherwise pass ``check`` (subtitles do not change predicted metadata) and
    only fail at decode.
    """
    if not context:
        return []
    source = str(segment.source)
    out: list[_LocatedError] = []
    for op_index, op in enumerate(segment.operations):
        for key in op.requires:
            value = context.get(key)
            if isinstance(value, dict) and source not in value:
                available = sorted(str(k) for k in value)
                message = (
                    f"Segment {index}: operation '{op.op}' requires context '{key}' for source "
                    f"'{source}', but the per-source map has no entry for it. Available: {available}."
                )
                out.append(
                    (
                        message,
                        PlanError(
                            PlanErrorCode.CONTEXT_SOURCE_MISSING,
                            location=f"segments[{index}].operations[{op_index}]",
                            op=op.op,
                        ),
                    )
                )
    return out


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
    overruns, so the final check stays silent for it (matching ``run_to_file()``).
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
                PlanError(
                    code=PlanErrorCode.WINDOW_NEGATIVE,
                    location=location,
                    op=op.op,
                    field="window.start",
                    value=start,
                    limit=0.0,
                ),
            )
        )
    if stop is not None and stop < 0:
        out.append(
            (
                f"Effect '{op.op}' window.stop ({stop}) must be >= 0",
                PlanError(
                    code=PlanErrorCode.WINDOW_NEGATIVE,
                    location=location,
                    op=op.op,
                    field="window.stop",
                    value=stop,
                    limit=0.0,
                ),
            )
        )
    if start is not None and stop is not None and stop < start:
        out.append(
            (
                f"Effect '{op.op}' window.stop ({stop}) must be >= start ({start})",
                PlanError(
                    code=PlanErrorCode.WINDOW_ORDER,
                    location=location,
                    op=op.op,
                    field="window.stop",
                    value=stop,
                    limit=start,
                ),
            )
        )
    if start is not None and start > duration + eps:
        out.append(
            (
                f"Effect '{op.op}' window.start ({start}) exceeds duration ({duration}s)",
                PlanError(
                    code=PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION,
                    location=location,
                    op=op.op,
                    field="window.start",
                    value=start,
                    limit=duration,
                ),
            )
        )
    if stop is not None and stop > duration + eps:
        out.append(
            (
                f"Effect '{op.op}' window.stop ({stop}) exceeds duration ({duration}s)",
                PlanError(
                    code=PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION,
                    location=location,
                    op=op.op,
                    field="window.stop",
                    value=stop,
                    limit=duration,
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


def _transition_structure_errors(segments: list[SegmentConfig]) -> list[_LocatedError]:
    """``TRANSITION_TOO_LONG`` for a transition on the first segment.

    A transition describes how a segment enters from the PREVIOUS one, so
    ``segments[0].transition_in`` has no predecessor and is rejected up front
    (independent of any duration). Per-boundary duration overruns are reported
    separately by :func:`_transition_duration_errors`, which needs the
    predicted post-op durations.
    """
    out: list[_LocatedError] = []
    if segments and segments[0].transition_in is not None:
        out.append(
            (
                "Segment 0: transition_in is not allowed on the first segment (no previous segment to "
                "transition from).",
                PlanError(
                    code=PlanErrorCode.TRANSITION_TOO_LONG,
                    location="segments[0]",
                    field="transition_in",
                ),
            )
        )
    return out


def _transition_too_long_error(
    index: int,
    spec: TransitionSpec,
    overlap: int,
    limit_frames: int,
    limit_seconds: float,
    remedy: str,
) -> _LocatedError:
    """The located ``TRANSITION_TOO_LONG`` error for an overlap that fills a segment.

    Shared by the validation pass (:func:`_transition_duration_errors`) and the
    pre-decode guard (:meth:`VideoEdit._assert_transitions_runnable`); each call
    site resolves its own ``overlap``/``limit_frames``/``limit_seconds`` from its
    own fps source, then supplies the trailing ``remedy`` prose.
    """
    message = (
        f"Segment {index}: transition_in overlaps {overlap} frames ({spec.duration}s) but must overlap "
        f"fewer frames than the shorter adjacent segment ({limit_frames} frames, {limit_seconds}s); "
        f"{remedy}"
    )
    return message, PlanError(
        code=PlanErrorCode.TRANSITION_TOO_LONG,
        location=f"segments[{index}]",
        field="transition_in.duration",
        value=spec.duration,
        limit=limit_seconds,
    )


def _transition_duration_errors(
    segments: list[SegmentConfig],
    outputs: list[VideoMetadata],
) -> list[_LocatedError]:
    """``TRANSITION_TOO_LONG`` for an overlap that consumes a whole adjacent segment.

    A transition consumes its ``round(duration * fps)`` overlap from the END of
    the left segment's predicted post-op output and the START of the right's,
    so the overlap must be *strictly fewer frames* than either adjacent
    segment. The constraint is frame-based to match the streaming pass exactly
    (:func:`videopython.editing.streaming.stream_transition_pair` raises on
    ``overlap >= n_frames``); a seconds comparison rounds differently and
    admits a near-full-overlap sliver that crashes ``run_to_file``. A clean ``check`` guarantees the xfade pass will
    not fail at decode. ``repair`` clamps the same boundary mechanically.
    """
    out: list[_LocatedError] = []
    for i in range(1, len(segments)):
        spec = segments[i].transition_in
        if spec is None:
            continue
        fps = outputs[i].fps
        overlap = _transition_overlap_frames(spec, fps)
        limit_frames = min(outputs[i - 1].frame_count, outputs[i].frame_count)
        if overlap >= limit_frames:
            limit_seconds = round(limit_frames / fps, 4) if fps else 0.0
            out.append(
                _transition_too_long_error(
                    i,
                    spec,
                    overlap,
                    limit_frames,
                    limit_seconds,
                    "a transition cannot consume a whole segment.",
                )
            )
    return out


def _transition_overlap_frames(spec: TransitionSpec, fps: float) -> int:
    """The number of frames a transition overlaps two segments: ``round(D*fps)``.

    The single source of the overlap frame count, used by the timeline math
    (:func:`_assemble_timeline`) and the per-boundary streaming pass, so they
    cannot disagree on how much the seam shortens.
    """
    return round(spec.duration * fps)


def _assemble_timeline(
    outputs: list[VideoMetadata],
    transitions: list[TransitionSpec | None] | None = None,
) -> VideoMetadata:
    """The concatenated-output metadata for a list of per-segment outputs.

    Segment 0 fixes height/width/fps (matching/normalization already made them
    uniform). Without transitions ``frame_count`` and ``total_seconds`` simply
    sum -- the prediction-side model of ``run_to_file()``'s ``result + video`` concat.
    With transitions (``transitions[i]`` is segment ``i``'s ``transition_in``),
    each boundary overlaps ``round(D*fps)`` frames, so the assembled total is
    ``sum(durations) - sum(overlaps)``: every post-op window, ``predict``,
    ``repair``, and the streamability report see the true shortened output.
    Shared by :meth:`_collect` (validates post-ops against it) and
    :meth:`repair` (clamps post-ops against it) so the two can never disagree.
    """
    first = outputs[0]
    frame_count = sum(m.frame_count for m in outputs)
    if transitions is not None:
        for i, spec in enumerate(transitions):
            if spec is not None and i > 0:
                frame_count -= _transition_overlap_frames(spec, first.fps)
    total_seconds = round(frame_count / first.fps, 4) if first.fps else round(sum(m.total_seconds for m in outputs), 4)
    return VideoMetadata(
        height=first.height,
        width=first.width,
        fps=first.fps,
        frame_count=frame_count,
        total_seconds=total_seconds,
    )


def _relocate(errors: list[PlanError], location: str) -> None:
    """Stamp ``location`` onto each error, preserving any deeper sub-path it carries.

    A typed op error may already carry a field-level ``location`` (e.g. a nested
    op error); prefix it with the op's plan path, otherwise set it. Mutates in
    place. The one home for the located-error convention shared by every per-op
    walk (``_collect``'s segment and post-op loops, ``_predict_segment``).
    """
    for err in errors:
        err.location = location if err.location is None else f"{location}.{err.location}"


def _clamp_effect_window(op: Operation, duration: float) -> Operation:
    """Return ``op`` with its ``Effect.window.stop`` clamped to ``duration``.

    Mirrors the run-time ``min(stop, total_seconds)`` window clamp the streaming
    engine applies, so a stop overrunning a duration-shrunk chain validates
    instead of raising.
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


def _repair_effect_window(op: Operation, duration: float, location: str) -> tuple[Operation, list[PlanRepair]]:
    """Clamp an :attr:`Effect.window`'s ``start``/``stop`` into ``[0, duration]``.

    The repair counterpart to :func:`_window_errors`, sharing its boundaries
    exactly: a negative endpoint snaps to ``0`` (``WINDOW_NEGATIVE``); one past
    the duration *by more than* ``DURATION_EPS`` snaps to ``duration``
    (``EFFECT_WINDOW_EXCEEDS_DURATION``). The eps tolerance matches the check, so
    a within-eps overrun ``check``/``validate`` accept is left untouched (no
    phantom repair). Each change is recorded as a :class:`PlanRepair` at
    ``location``. A ``stop < start`` left after clamping is deliberately not
    invented away -- it stays for ``check``/``validate`` to report.
    """
    if not isinstance(op, Effect) or op.window is None:
        return op, []
    start, stop = op.window.start, op.window.stop
    new_start, new_stop = start, stop
    changes: list[PlanRepair] = []
    eps = DURATION_EPS
    if start is not None and start < 0:
        new_start = 0.0
        changes.append(PlanRepair(location, "window.start", start, 0.0, PlanErrorCode.WINDOW_NEGATIVE))
    elif start is not None and start > duration + eps:
        new_start = duration
        changes.append(
            PlanRepair(location, "window.start", start, duration, PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION)
        )
    if stop is not None and stop < 0:
        new_stop = 0.0
        changes.append(PlanRepair(location, "window.stop", stop, 0.0, PlanErrorCode.WINDOW_NEGATIVE))
    elif stop is not None and stop > duration + eps:
        new_stop = duration
        changes.append(
            PlanRepair(location, "window.stop", stop, duration, PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION)
        )
    if not changes:
        return op, []
    new_window = op.window.model_copy(update={"start": new_start, "stop": new_stop})
    return op.model_copy(update={"window": new_window}), changes


def _repair_time_fields(op: Operation, meta: VideoMetadata, location: str) -> tuple[Operation, list[PlanRepair]]:
    """Clamp every declared :attr:`Operation.time_fields` value into range.

    Generic over the op's :class:`BoundedTimeField` declarations, and gated on
    *exactly what validation rejects* so an in-range value is never touched (no
    rounding, no phantom changelog entry). An ``exclusive_end`` field indexes a
    frame, so validation rejects ``value >= total_seconds`` (e.g.
    ``freeze_frame.timestamp``); an out-of-range one is clamped to the last
    addressable frame ``(frame_count - 1) / fps`` (strictly ``< duration``). An
    inclusive field rejects ``value > total_seconds + DURATION_EPS`` and clamps
    to the duration. Each change is recorded as an ``OP_TIMESTAMP_OUT_OF_RANGE``
    :class:`PlanRepair` at ``location``.
    """
    changes: list[PlanRepair] = []
    new_op = op
    for tf in op.time_fields:
        value = getattr(new_op, tf.name)
        if value is None:
            continue
        if tf.exclusive_end:
            over = value >= meta.total_seconds
            upper = max(0.0, (meta.frame_count - 1) / meta.fps)
        else:
            over = value > meta.total_seconds + DURATION_EPS
            upper = max(0.0, meta.total_seconds)
        if value < 0 or over:
            clamped = round(min(max(value, 0.0), upper), 4)
            new_op = new_op.model_copy(update={tf.name: clamped})
            changes.append(PlanRepair(location, tf.name, value, clamped, PlanErrorCode.OP_TIMESTAMP_OUT_OF_RANGE))
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
    :func:`_repair_time_fields` applied at ``f'{location_prefix}[op_index]'``.
    The first op that cannot be predicted ends clamping -- the rest are kept
    verbatim (we can't know the duration past it) and the returned running meta
    is ``None`` so the caller knows the chain broke.
    """
    new_ops: list[Operation] = []
    repairs: list[PlanRepair] = []
    running: VideoMetadata | None = meta
    for op_index, op in enumerate(ops):
        if clamp and running is not None:
            location = f"{location_prefix}[{op_index}]"
            op, win_changes = _repair_effect_window(op, running.total_seconds, location)
            op, time_changes = _repair_time_fields(op, running, location)
            repairs.extend(win_changes)
            repairs.extend(time_changes)
        new_ops.append(op)
        if running is None:
            continue
        try:
            running = _predict_with_context(op, running, context)
        except (ValueError, TypeError):
            running = None
    return new_ops, repairs, running


@dataclasses.dataclass(frozen=True)
class _MatchTarget:
    """The resolved concat-compatibility target for a set of segment metas.

    Produced by :meth:`VideoEdit._resolve_matching_target` and consumed by BOTH
    the prediction pass (:meth:`VideoEdit._apply_matching`) and the execution
    pass (:meth:`VideoEdit._matching_targets_from_disk`), so the
    min-fps/min-resolution policy lives in one place and the two cannot drift. A
    ``None`` axis means that axis is not matched (its ``match_to_lowest_*`` flag
    is off, or there is nothing to match): callers leave that axis untouched.
    """

    fps: float | None
    width: int | None
    height: int | None

    def apply(self, meta: VideoMetadata) -> VideoMetadata:
        """Return ``meta`` conformed to this target on every non-``None`` axis."""
        out = meta
        if self.width is not None and self.height is not None and (out.width, out.height) != (self.width, self.height):
            out = out.with_dimensions(self.width, self.height)
        if self.fps is not None and out.fps != self.fps:
            out = out.with_fps(self.fps)
        return out


class TransitionSpec(BaseModel):
    """How one segment enters from the previous one: a native ffmpeg crossfade.

    A transition describes the boundary on its INCOMING side -- it lives on
    ``SegmentConfig.transition_in`` of segment ``i`` and overlaps the last
    ``duration`` seconds of segment ``i-1`` with the first ``duration`` of
    segment ``i``. ``type`` is a literal ffmpeg ``xfade`` ``transition=`` mode
    from a curated catalog; ``duration`` is the overlap in seconds (the
    assembled timeline shortens by it); ``audio`` ``acrossfade``-s the audio
    across the same overlap when both adjacent segments carry audio, else the
    audio hard butt-joins. Frozen and closed so it surfaces as a constrained
    object in :meth:`VideoEdit.json_schema`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: TransitionType = Field(
        description="Crossfade style: an ffmpeg xfade transition mode from the curated catalog."
    )
    duration: float = Field(gt=0, description="Overlap duration in seconds; the assembled timeline shortens by it.")
    audio: bool = Field(
        True,
        description=(
            "Crossfade (acrossfade) the audio across the same overlap when both adjacent "
            "segments have audio; otherwise the audio hard butt-joins at the seam."
        ),
    )


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
    transition_in: TransitionSpec | None = Field(
        default=None,
        description=(
            "Optional crossfade from the previous segment into this one. Must be null on the "
            "first segment (there is no previous segment to transition from)."
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
    music_bed: MusicBed | None = Field(
        default=None,
        description=(
            "Optional music bed mixed under the whole assembled program in a final pass "
            "(after segment concat / transitions). Set `duck` on it to lower the bed under "
            "transcription-derived speech (single-segment plans only)."
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

        segment_schema: dict[str, Any] = {
            "type": "object",
            "description": SegmentConfig.__doc__,
            "properties": {
                "source": field_schema(SegmentConfig, "source"),
                "start": field_schema(SegmentConfig, "start"),
                "end": field_schema(SegmentConfig, "end"),
                "operations": array_field_schema(SegmentConfig, "operations", op_schema),
                "transition_in": optional_model_field_schema(TransitionSpec, SegmentConfig, "transition_in"),
            },
            "required": ["source", "start", "end"],
            "additionalProperties": False,
        }
        segments = field_schema(cls, "segments")
        segments["items"] = segment_schema
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "description": cls.__doc__,
            "properties": {
                "segments": segments,
                "post_operations": array_field_schema(cls, "post_operations", op_schema),
                "match_to_lowest_fps": field_schema(cls, "match_to_lowest_fps"),
                "match_to_lowest_resolution": field_schema(cls, "match_to_lowest_resolution"),
                "music_bed": optional_model_field_schema(MusicBed, cls, "music_bed"),
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
        ``min(stop, total_seconds)`` value the streaming engine applies at run
        time -- instead of raising. Only ``window.stop`` is clamped: a ``window.start`` past the
        duration still hard-raises (a residual divergence from ``run_to_file()``, which
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

        Streaming is the only engine, so ops that cannot stream at their
        plan position are real plan errors: one ``STREAMING_UNSUPPORTED`` per
        offending op is appended after the validity errors, in plan order,
        with the actionable cause in :attr:`PlanError.detail`. See
        :meth:`streamability` for the full per-op report including the ops
        that *do* stream.
        """
        metas = self._resolve_source_metas(source_metadata)
        _, errors = self._collect(metas, context, clamp_windows=clamp_windows, stop_first=False)
        errors.extend(self.streamability().errors())
        return errors

    def streamability(self) -> StreamabilityReport:
        """Classify every op by streaming class, without touching the disk.

        Streamability is purely structural -- it depends on op classes, their
        order, and the plan shape, never on source metadata or runtime context
        -- so this needs no source files and is safe to call before a job is
        admitted. ``report.streamable`` answers "will :meth:`run_to_file`
        stream this plan in O(1) memory, or is an op unstreamable at its
        plan position?"; each entry carries the op's memory class and, for
        unstreamable ops, the reason.
        """
        return analyze_streamability(
            [list(seg.operations) for seg in self.segments],
            list(self.post_operations),
        )

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
            seg_context = _segment_context(context, str(seg.source), seg.start, seg.end)
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

        # Pass 2b: clamp each transition's overlap down to one frame short of the
        # shorter adjacent segment -- the mechanical TRANSITION_TOO_LONG repair,
        # mirroring the window clamps. The clamp is frame-safe: it targets
        # `limit_frames - 1` overlap frames so the repaired plan satisfies the
        # strict `overlap < min(frames)` streaming constraint and actually runs
        # (a seconds clamp to the segment length lands on `overlap == frames`,
        # which passes check but crashes run_to_file). Only fires when both adjacent
        # segments predicted; a segment too short for any transition
        # (limit_frames <= 1) is left for check. A first-segment transition is a
        # structural fault, deliberately left for check.
        if clamp_op_params:
            for i in range(1, len(final_segments)):
                spec = final_segments[i].transition_in
                if spec is None:
                    continue
                left_meta = seg_output_metas[i - 1]
                right_meta = seg_output_metas[i]
                if left_meta is None or right_meta is None:
                    continue
                fps = right_meta.fps
                overlap = _transition_overlap_frames(spec, fps)
                limit_frames = min(left_meta.frame_count, right_meta.frame_count)
                if overlap >= limit_frames and limit_frames > 1 and fps:
                    new_dur = round((limit_frames - 1) / fps, 4)
                    repairs.append(
                        PlanRepair(
                            f"segments[{i}]",
                            "transition_in.duration",
                            spec.duration,
                            new_dur,
                            PlanErrorCode.TRANSITION_TOO_LONG,
                        )
                    )
                    new_spec = spec.model_copy(update={"duration": new_dur})
                    final_segments[i] = final_segments[i].model_copy(update={"transition_in": new_spec})

        # Pass 3: clamp post-op windows / time-fields against the assembled
        # timeline -- but only when every segment predicted (else its duration is
        # unknown). A negative post_operations window is the canonical attempt-3
        # failure, so this matters as much as the per-segment clamps.
        update: dict[str, Any] = {"segments": final_segments}
        outputs = [m for m in seg_output_metas if m is not None]
        if self.post_operations and len(outputs) == len(self.segments):
            transitions = [seg.transition_in for seg in final_segments]
            assembled = _assemble_timeline(outputs, transitions)
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
        target: tuple[int, int] | Literal["first", "largest", "match"],
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
        elif target == "match":
            # The same min-resolution policy the engine's
            # match_to_lowest_resolution applies in-stream, materialized as
            # resize ops so it survives serialization. Degrades to "first" when
            # the flag is off (resolved.width is None), keeping the
            # always-produce-a-compatible-plan contract.
            resolved = self._resolve_matching_target(known)
            tw = resolved.width if resolved.width is not None else known[0].width
            th = resolved.height if resolved.height is not None else known[0].height
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
        """Raising guard used by ``run_to_file`` (see :meth:`_post_op_context_errors`)."""
        errs = self._post_op_context_errors(context)
        if errs:
            message, err = errs[0]
            raise PlanValidationError(message, [err])

    def _music_bed_errors(self) -> list[_LocatedError]:
        """Validate :attr:`music_bed`: readable source, and duck on a single segment.

        Two checks, both fail-fast (before any decode): the bed ``source`` must
        be a readable audio file (``SOURCE_UNREADABLE``, a cheap ffprobe header
        probe like :class:`ImageOverlay`'s); and transcription-derived ducking is
        only well-defined when the assembled timeline is a single segment's
        timeline -- a multi-segment plan with ``duck`` set is rejected
        (``MUSIC_BED_DUCK_MULTISEGMENT``), since the assembled-timeline
        transcription mapping across cuts is out of scope. A non-ducked bed on a
        multi-segment plan is fine.
        """
        bed = self.music_bed
        if bed is None:
            return []
        out: list[_LocatedError] = []
        if bed.duck is not None and len(self.segments) > 1:
            message = (
                f"music_bed.duck is set, but the plan has {len(self.segments)} segments. "
                "Transcription-derived ducking is only supported on a single-segment plan "
                "(the assembled-timeline transcription mapping across cuts is out of scope); "
                "drop the duck or use a single-segment plan."
            )
            out.append(
                (
                    message,
                    PlanError(PlanErrorCode.MUSIC_BED_DUCK_MULTISEGMENT, location="music_bed", field="duck"),
                )
            )
        try:
            bed.validate_source()
        except PlanValidationError as e:
            for err in e.errors:
                err.location = "music_bed"
            out.append((str(e), e.errors[0]))
        return out

    def _assert_music_bed_supported(self) -> None:
        """Raising guard used by ``run_to_file`` (see :meth:`_music_bed_errors`)."""
        errs = self._music_bed_errors()
        if errs:
            message, err = errs[0]
            raise PlanValidationError(message, [err])

    def _music_bed_speech_windows(self, context: dict[str, Any] | None) -> list[tuple[float, float]] | None:
        """Speech windows for the music-bed duck, on the single segment's timeline.

        Only reached for a single-segment plan (the duck-multisegment guard
        rejects the rest), so the assembled timeline == the segment's timeline:
        the rebased context transcription maps directly. Resolves the
        segment-local transcription exactly as the per-segment ops do
        (:func:`_segment_context`), then derives padded speech windows via the
        shared :func:`speech_windows` helper. ``None`` when the bed is not
        ducked or there is no transcription to derive windows from -- a non-ducked
        (or context-less) bed mixes flat.
        """
        bed = self.music_bed
        if bed is None or bed.duck is None:
            return None
        from videopython.base.transcription import Transcription

        segment = self.segments[0]
        seg_context = _segment_context(context, str(segment.source), segment.start, segment.end)
        transcription = (seg_context or {}).get("transcription")
        if not isinstance(transcription, Transcription):
            return None
        # The bed's padding mirrors silence_removal's default breathing room so
        # the duck eases around speech rather than clipping each word tightly.
        total = segment.end - segment.start
        return speech_windows(transcription.words, 0.15, total)

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

        for message, err in self._music_bed_errors():
            emit(message, err)

        for message, err in _transition_structure_errors(list(self.segments)):
            emit(message, err)

        for i, seg in enumerate(self.segments):
            for message, err in _missing_source_context_errors(context, i, seg):
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
        # matched dims may differ from run_to_file()'s all-segments matching -- harmless,
        # since that plan can't run until the bad segment is fixed.
        seg_outputs: dict[int, VideoMetadata] = {}
        for i, seg in enumerate(self.segments):
            seg_meta = matched_by_index.get(i)
            if seg_meta is None:
                continue
            seg_context = _segment_context(context, str(seg.source), seg.start, seg.end)
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
                    _relocate(e.errors, location)
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
        for message, err in _transition_duration_errors(list(self.segments), outputs):
            emit(message, err)

        transitions = [seg.transition_in for seg in self.segments]
        assembled = _assemble_timeline(outputs, transitions)
        for j, op in enumerate(self.post_operations):
            location = f"post_operations[{j}]"
            for message, err in _window_errors(op, assembled.total_seconds, location):
                emit(message, err)
            try:
                assembled = _predict_with_context(op, assembled, context)
            except PlanValidationError as e:
                _relocate(e.errors, location)
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
        seg_context = _segment_context(context, str(segment.source), segment.start, segment.end)
        for op_index, op in enumerate(segment.operations):
            location = f"segments[{index}].operations[{op_index}]"
            for message, err in _window_errors(op, meta.total_seconds, location):
                raise PlanValidationError(message, [err])
            try:
                meta = _predict_with_context(op, meta, seg_context)
            except PlanValidationError as e:
                _relocate(e.errors, location)
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

    def _resolve_matching_target(self, metas: list[VideoMetadata]) -> _MatchTarget:
        """The one place the min-fps/min-resolution concat policy is decided.

        Reduces per-segment metas to a single :class:`_MatchTarget` honoring
        ``match_to_lowest_fps`` / ``match_to_lowest_resolution``. An axis is
        ``None`` (left untouched) when its flag is off or there is nothing to
        match (``<= 1`` meta). Shared verbatim by the prediction pass
        (:meth:`_apply_matching`) and the execution pass
        (:meth:`_matching_targets_from_disk`), and reused by
        :meth:`normalize_dimensions` for its ``"match"`` target, so all three
        agree by construction. Pure: no disk, no mutation of the inputs.
        """
        if len(metas) <= 1:
            return _MatchTarget(fps=None, width=None, height=None)
        fps = min(m.fps for m in metas) if self.match_to_lowest_fps else None
        width = min(m.width for m in metas) if self.match_to_lowest_resolution else None
        height = min(m.height for m in metas) if self.match_to_lowest_resolution else None
        return _MatchTarget(fps=fps, width=width, height=height)

    def _apply_matching(self, metas: list[VideoMetadata]) -> list[VideoMetadata]:
        if len(metas) <= 1:
            return metas
        target = self._resolve_matching_target(metas)
        return [target.apply(m) for m in metas]

    def run_to_file(
        self,
        output_path: str | Path,
        format: ALLOWED_VIDEO_FORMATS = "mp4",
        preset: ALLOWED_VIDEO_PRESETS = "medium",
        crf: int = 23,
        context: dict[str, Any] | None = None,
    ) -> Path:
        """Execute the plan, streaming directly to a file.

        Memory usage is O(1) w.r.t. video length (video; segment audio is
        in-memory). Streaming is the only engine: a plan with an unstreamable
        shape raises :class:`PlanValidationError` carrying one
        ``STREAMING_UNSUPPORTED`` :class:`PlanError` per offending op -- before
        any decode. Gate plans early with :meth:`check` or
        :meth:`streamability`, which report the same errors without running
        anything.
        """
        output_path = Path(output_path).with_suffix(f".{format}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plans = self._compile_streaming_plans(context)
        self._assert_music_bed_supported()

        # The program flows through up to three stages, each writing to its own
        # temp file: assemble segments -> apply post_operations -> mix music bed.
        # Only the final active stage writes straight to `output_path`.
        has_post = bool(self.post_operations)
        has_bed = self.music_bed is not None
        intermediates: list[Path] = []

        def _stage_target(*, is_last: bool) -> Path:
            if is_last:
                return output_path
            tmp = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
            tmp.close()
            p = Path(tmp.name)
            intermediates.append(p)
            return p

        try:
            self._assert_transitions_runnable(plans)
            # Stage 1: assemble the segments.
            target = _stage_target(is_last=not (has_post or has_bed))
            if len(plans) == 1:
                assembled = stream_segment(plans[0], target, format=format, preset=preset, crf=crf)
            else:
                # Realize each segment to its own temp file (effects + audio
                # already baked by stream_segment's filter_complex), capturing
                # per-segment audibility for the crossfade-vs-butt-join decision
                # at any transition seam. Audibility is the source's audio-stream
                # presence: a source with no audio gets a native silent track
                # (anullsrc), so it never crossfades -- the same decision the old
                # materialized is_silent made for the no-audio case, without
                # decoding audio.
                temp_files: list[Path] = []
                audible: list[bool] = []
                try:
                    for plan in plans:
                        tmp = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
                        tmp.close()
                        audible.append(source_has_audio_stream(plan.source_path))
                        stream_segment(plan, Path(tmp.name), with_audio=True, format=format, preset=preset, crf=crf)
                        temp_files.append(Path(tmp.name))

                    if all(seg.transition_in is None for seg in self.segments):
                        assembled = concat_files(temp_files, target)
                    else:
                        assembled = self._assemble_with_transitions(
                            temp_files, audible, target, format=format, preset=preset, crf=crf
                        )
                finally:
                    for f in temp_files:
                        f.unlink(missing_ok=True)

            # Stage 2: apply post_operations as ONE pass over the assembled
            # program (a synthetic single segment), so filter-class effects,
            # frame effects, and transforms all apply on the assembled timeline.
            if has_post:
                target = _stage_target(is_last=not has_bed)
                assembled = self._apply_post_operations(
                    assembled, target, context, format=format, preset=preset, crf=crf
                )

            # Stage 3: mix the music bed under the assembled program.
            if has_bed:
                speech = self._music_bed_speech_windows(context)
                assembled = self._mix_music_bed_to_file(
                    assembled, output_path, speech, format=format, preset=preset, crf=crf
                )

            return assembled
        finally:
            for p in intermediates:
                p.unlink(missing_ok=True)
            for plan in plans:
                for f in plan.owned_temp_files:
                    f.unlink(missing_ok=True)

    def _apply_post_operations(
        self,
        assembled: Path,
        output_path: Path,
        context: dict[str, Any] | None,
        *,
        format: str,
        preset: str,
        crf: int,
    ) -> Path:
        """Apply ``post_operations`` as one pass over the assembled program.

        The assembled file is treated as a single synthetic segment whose
        ``operations`` are the ``post_operations``, then run through the same
        streaming engine as any segment -- so filter-class effects (vignette,
        color_adjust, ...), frame effects, and transforms all apply uniformly to
        the whole program on the assembled timeline, with each effect ``window``
        resolving against the assembled file's own frame count. No per-segment
        fold and no cross-boundary re-basing.

        Runtime context: a single-segment plan's assembled timeline IS that
        segment's, so a rebaseable value (e.g. a ``Transcription``) is re-based
        onto it exactly as a per-segment op would see it. Multi-segment context
        post-ops are rejected up front by :meth:`_assert_post_ops_supported`
        (source-absolute context cannot be re-based across a concat), so only
        broadcast (non-time) context reaches a multi-segment post-op pass.
        """
        meta = VideoMetadata.from_path(str(assembled))
        seg = SegmentConfig(
            source=assembled,
            start=0.0,
            end=round(meta.total_seconds, 6),
            operations=list(self.post_operations),
        )
        if len(self.segments) == 1:
            src = self.segments[0]
            post_context = _segment_context(context, str(src.source), src.start, src.end)
        else:
            post_context = context
        plan = self._build_streaming_plan(seg, None, None, None, post_context)
        if plan is None:
            raise PlanValidationError(
                "post_operations did not compile to a streaming plan",
                [PlanError(code=PlanErrorCode.STREAMING_UNSUPPORTED, location="post_operations")],
            )
        try:
            return stream_segment(plan, output_path, format=format, preset=preset, crf=crf)
        finally:
            for f in plan.owned_temp_files:
                f.unlink(missing_ok=True)

    def _mix_music_bed_to_file(
        self,
        assembled: Path,
        output_path: Path,
        speech: list[tuple[float, float]] | None,
        *,
        format: str,
        preset: str,
        crf: int,
    ) -> Path:
        """Mix the music bed under an assembled program file in one ffmpeg pass.

        The bed mix: the assembled program is input 0, the bed
        a second ``-i`` input, and :func:`build_music_bed_filter_complex`
        compiles the ``amix`` graph. The program duration is the
        assembled file's PROBED length so the bed's loop/trim pin matches the
        realized timeline exactly. The video stream is copied (``-c:v copy`` --
        the bed touches only audio); the mixed audio is re-encoded to AAC.
        """
        bed = self.music_bed
        assert bed is not None
        meta = VideoMetadata.from_path(str(assembled))
        program_seconds = meta.total_seconds
        bed_inputs, graph, out_label = build_music_bed_filter_complex(
            bed, program_seconds, speech=speech, bed_input_index=1, prog_label="0:a"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(assembled),
            *bed_inputs,
            "-filter_complex",
            ";".join(graph),
            "-map",
            "0:v",
            "-map",
            out_label,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        try:
            _ffmpeg.run(cmd)
        except Exception:
            if output_path.exists():
                output_path.unlink()
            raise
        return output_path

    def _assemble_with_transitions(
        self,
        seg_files: list[Path],
        audible: list[bool],
        output_path: Path,
        *,
        format: str,
        preset: str,
        crf: int,
    ) -> Path:
        """Boundary-aware assembly: concat-copy hard-cut runs, xfade transitions.

        Walks segments left to right. Maximal runs of hard-cut (``transition_in
        is None``) boundaries are grouped and joined with ``concat_files``
        (``-c copy``, no re-encode), so only transition seams pay an encode.
        Each transition then xfade-joins the running tail with the next group:
        ``offset`` is derived from the realized TAIL's PROBED duration (not the
        prediction) so the seam is frame-aligned, the video blend goes through
        the shared :func:`xfade_filter` builder (matching ``run_to_file()``), and the
        audio is ``acrossfade``-d when ``spec.audio`` and both boundary
        segments are audible, else hard butt-joined.

        Note xfade re-encodes the whole left tail at each seam, so a
        transition-heavy reel re-encodes its cumulative tail repeatedly; a
        mixed reel only re-encodes at the seams (the hard-cut runs stay copies).
        """
        # Partition into hard-cut groups; a new group starts at each segment
        # whose transition_in is set. Each group concat-copies to one file.
        groups: list[list[int]] = [[0]]
        for i in range(1, len(self.segments)):
            if self.segments[i].transition_in is None:
                groups[-1].append(i)
            else:
                groups.append([i])

        owned: list[Path] = []

        def _group_file(indices: list[int]) -> Path:
            if len(indices) == 1:
                return seg_files[indices[0]]
            tmp = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
            tmp.close()
            owned.append(Path(tmp.name))
            return concat_files([seg_files[i] for i in indices], Path(tmp.name))

        try:
            tail = _group_file(groups[0])
            tail_left_audible = audible[groups[0][-1]]
            for g in range(1, len(groups)):
                right_indices = groups[g]
                right_file = _group_file(right_indices)
                first_idx = right_indices[0]
                spec = self.segments[first_idx].transition_in
                assert spec is not None  # a group (after the first) always opens on a transition
                meta = VideoMetadata.from_path(str(tail))
                right_meta = VideoMetadata.from_path(str(right_file))
                # Offset from the realized tail's PROBED frame count (matching
                # the xfade input trim), not the prediction, so the seam is
                # frame-aligned -- ``(n_left - overlap) / fps``.
                overlap = _transition_overlap_frames(spec, meta.fps)
                offset = round((meta.frame_count - overlap) / meta.fps, 6)
                crossfade_audio = bool(spec.audio and tail_left_audible and audible[first_idx])

                is_last = g == len(groups) - 1
                if is_last:
                    target = output_path
                else:
                    tmp = tempfile.NamedTemporaryFile(suffix=f".{format}", delete=False)
                    tmp.close()
                    target = Path(tmp.name)
                    owned.append(target)
                new_tail = stream_transition_pair(
                    tail,
                    right_file,
                    spec.type,
                    spec.duration,
                    offset,
                    target,
                    width=meta.width,
                    height=meta.height,
                    fps=meta.fps,
                    left_frame_count=meta.frame_count,
                    right_frame_count=right_meta.frame_count,
                    left_has_audio=True,
                    right_has_audio=True,
                    crossfade_audio=crossfade_audio,
                    format=format,
                    preset=preset,
                    crf=crf,
                )
                # Once we xfade past a seam the new tail's trailing content is
                # the right group's, so the next seam's audibility is that
                # group's last segment.
                tail = new_tail
                tail_left_audible = audible[right_indices[-1]]
            return output_path
        finally:
            for f in owned:
                if f != output_path:
                    f.unlink(missing_ok=True)

    def _compile_streaming_plans(self, context: dict[str, Any] | None) -> list[StreamingSegmentPlan]:
        """Compile every per-segment streaming plan, or raise.

        Post-operations are NOT compiled here -- they run as a separate pass over
        the assembled program (:meth:`_apply_post_operations`).

        The single admission point for :meth:`run_to_file`. Unstreamable shapes raise
        :class:`PlanValidationError` with the streamability report's
        structured errors; a builder/report drift (e.g. an op whose
        ``to_ffmpeg_filter`` returns ``None`` at its plan position despite
        overriding it) raises with a generic ``STREAMING_UNSUPPORTED``.
        On raise, any compile-time temp files already created are deleted;
        on success the caller owns ``plan.owned_temp_files``.
        """
        self._assert_post_ops_supported(context)
        report = self.streamability()
        if not report.streamable:
            causes = "; ".join(f"{e.location} '{e.op}': {e.reason}" for e in report.unstreamable)
            message = (
                f"Plan cannot stream: {len(report.unstreamable)} op(s) have no streaming "
                f"strategy at their plan position -- {causes}"
            )
            raise PlanValidationError(message, report.errors())

        def drift(detail: str) -> PlanValidationError:
            return PlanValidationError(
                f"plan stopped streaming despite a clean streamability report ({detail})",
                [PlanError(code=PlanErrorCode.STREAMING_UNSUPPORTED, detail=detail)],
            )

        target_fps, target_w, target_h = self._matching_targets_from_disk()
        plans: list[StreamingSegmentPlan] = []
        try:
            for i, segment in enumerate(self.segments):
                plan = self._build_streaming_plan(segment, target_fps, target_w, target_h, context)
                if plan is None:
                    raise drift(f"segments[{i}] did not compile to a streaming plan")
                plans.append(plan)
        except BaseException:
            for plan in plans:
                for f in plan.owned_temp_files:
                    f.unlink(missing_ok=True)
            raise
        return plans

    # ----------------------------------------------------------------- helpers

    def _matching_targets_from_disk(self) -> tuple[float | None, int | None, int | None]:
        if len(self.segments) <= 1 or (not self.match_to_lowest_fps and not self.match_to_lowest_resolution):
            return None, None, None
        metas = [VideoMetadata.from_path(str(seg.source)) for seg in self.segments]
        target = self._resolve_matching_target(metas)
        return target.fps, target.width, target.height

    def _build_streaming_plan(
        self,
        segment: SegmentConfig,
        target_fps: float | None,
        target_w: int | None,
        target_h: int | None,
        context: dict[str, Any] | None = None,
    ) -> StreamingSegmentPlan | None:
        source_meta = VideoMetadata.from_path(str(segment.source))
        # Fold real metadata through the chain -- the same walk validation
        # does -- so duration-changing transforms (speed, freeze) keep frame
        # counts, effect ranges, and audio in sync with the actual output.
        running = CutSeconds(start=segment.start, end=segment.end).predict_metadata(source_meta)
        if running.frame_count <= 0:
            message = (
                f"Segment [{segment.start}, {segment.end}) is shorter than one frame "
                f"at {running.fps} fps and cannot be rendered"
            )
            raise PlanValidationError(
                message,
                [PlanError(code=PlanErrorCode.DEGENERATE_DURATION, op=None, field="end", value=segment.end)],
            )

        vf_filters: list[str] = []
        if target_w and target_h and (target_w != source_meta.width or target_h != source_meta.height):
            vf_filters.append(f"scale={target_w}:{target_h}")
            running = running.with_dimensions(target_w, target_h)
        if target_fps and target_fps != source_meta.fps:
            vf_filters.append(f"fps={target_fps}")
            running = running.with_fps(target_fps)

        # Resolve requires-context onto the segment's local timeline once; each
        # context-requiring effect gets its keys on the schedule entry, which
        # stream_segment forwards to streaming_init. A missing/empty key is
        # passed as absent so the effect raises its own clear "requires ..."
        # error -- before that segment's decode.
        seg_context = _segment_context(context, str(segment.source), segment.start, segment.end)

        owned_files: list[Path] = []

        def abandon() -> None:
            """Delete compile-time temp files of a plan that won't run."""
            for f in owned_files:
                f.unlink(missing_ok=True)

        def make_ctx(decode_filters: tuple[str, ...] | None | object = _DECODE_FILTERS_DEFAULT) -> FilterCtx:
            """A :class:`FilterCtx` snapshot of the current ``running`` state.

            Captures the per-op pipeline geometry (dims/fps/frame-count) and the
            segment's location/context every op compiles against. ``decode_filters``
            is the decode-stage filter prefix ahead of this op; left unset it keeps
            :class:`FilterCtx`'s own default (the op consumes no decode pass).
            """
            kwargs: dict[str, Any] = {
                "width": running.width,
                "height": running.height,
                "fps": running.fps,
                "frame_count": running.frame_count,
                "context": seg_context or {},
                "source_path": segment.source,
                "start_second": segment.start,
                "end_second": segment.end,
            }
            if decode_filters is not _DECODE_FILTERS_DEFAULT:
                kwargs["decode_filters"] = decode_filters
            return FilterCtx(**kwargs)

        effect_schedule: list[EffectScheduleEntry] = []
        post_vf_filters: list[str] = []
        af_filters: list[str] = []
        post_af_filters: list[str] = []
        audio_idx = 0
        duration_changed = False

        def compile_audio_twin(op: Operation, ctx: FilterCtx, encode_stage: bool) -> None:
            """Append the op's audio-domain filter at the same stage the video
            filter landed, keeping audio and video stage placement coupled.

            ``ctx`` is the SAME FilterCtx the video side compiled this op with
            (pre-op ``running``), plus a per-op ``audio_label`` so a
            multi-statement fragment (freeze_frame's splice) cannot collide with
            another op's internal labels."""
            nonlocal audio_idx
            audio_ctx = dataclasses.replace(ctx, audio_label=f"f{audio_idx}")
            af = op.to_ffmpeg_audio_filter(audio_ctx)
            audio_idx += 1
            if af is None:
                return
            (post_af_filters if encode_stage else af_filters).append(af)

        # The pipe stage: dims/fps/frame-count of the frames flowing through
        # process_frame and into the encoder's rawvideo stdin. Frozen the
        # moment encode-stage content appears (a scheduled effect or any
        # post_vf entry); transforms after that fold `running` further (for
        # audio durations and the final output), but the pipe stays put.
        pipe_meta: VideoMetadata | None = None
        try:
            for op in segment.operations:
                if isinstance(op, Effect):
                    if not op.streams():
                        abandon()
                        return None
                    if op.requires and duration_changed:
                        # A duration-changing transform earlier in the chain
                        # moved the timeline; segment-local context (e.g. the
                        # transcription) is not re-mapped through the warp
                        # yet. Not streamable here -- rejected as
                        # UNSTREAMABLE by the streamability report.
                        abandon()
                        return None
                    if op.compiles_to_filter:
                        # Filter-class effect (add_subtitles): consumes its
                        # context at compile time and joins the filter chain at
                        # this op's plan position -- the decode chain when no
                        # frame effect precedes it, else the encode chain
                        # (FrameEncoder -vf), which runs after every
                        # process_frame. Either way plan order is preserved. A
                        # None compile falls through to the frame-effect path.
                        encode_stage_effect = bool(effect_schedule or post_vf_filters)
                        ctx = make_ctx(decode_filters=None if encode_stage_effect else tuple(vf_filters))
                        filter_expr = op.to_ffmpeg_filter(ctx)
                        owned_files.extend(ctx.owned_files)
                        if filter_expr is not None:
                            if encode_stage_effect:
                                pipe_meta = pipe_meta or running
                                post_vf_filters.append(filter_expr)
                            else:
                                vf_filters.append(filter_expr)
                            # Audio twin at the same stage (add_subtitles has
                            # none today; kept coupled for extensibility).
                            compile_audio_twin(op, ctx, encode_stage_effect)
                            continue
                    if post_vf_filters:
                        # A frame effect after an encode-stage filter would run
                        # before it (process_frame precedes the encoder), so
                        # plan order cannot be preserved -- not streamable
                        # (rejected as UNSTREAMABLE by the streamability report).
                        abandon()
                        return None
                    op_context: dict[str, Any] = {}
                    if op.requires and seg_context:
                        op_context = {k: seg_context[k] for k in op.requires if k in seg_context}
                    pipe_meta = pipe_meta or running
                    start_f, end_f = _effect_frame_range(op, running.fps, running.frame_count)
                    effect_schedule.append(EffectScheduleEntry(op, start_f, end_f, op_context))
                    # Audio-coupled frame effects (fade/volume_adjust) express
                    # their gain on the audio graph. A frame effect always sits
                    # at the decode stage (one after an encode-stage filter is
                    # rejected above), so its audio twin joins af_filters; its
                    # window resolves against this position's running metadata.
                    effect_ctx = make_ctx()
                    compile_audio_twin(op, effect_ctx, encode_stage=False)
                    continue
                if op.requires and (duration_changed or not op.streams()):
                    # A context-requiring transform can stream only when its
                    # filter compile can consume the context (silence_removal
                    # does, via FilterCtx.context) -- but not after a
                    # duration-changing transform (no time-warp re-mapping
                    # yet), and not without a streaming strategy. Not
                    # streamable here -- rejected as UNSTREAMABLE by the
                    # streamability report.
                    abandon()
                    return None
                # Non-effect transform: streams iff it compiles a filter. Checked
                # structurally (op.streams() == overrides to_ffmpeg_filter) so the
                # streamability report and the builder cannot disagree; the
                # remaining drift class (overrides to_ffmpeg_filter but it compiles
                # to None here) is caught by the STREAMING_UNSUPPORTED raise in
                # _compile_streaming_plans.
                if not op.streams():
                    abandon()
                    return None
                encode_stage = bool(effect_schedule or post_vf_filters)
                if encode_stage and op.compiles_from_source:
                    # The op's compile decodes the source to see its input
                    # frames (face_crop's detection pass); frames behind
                    # per-frame Python effects are not reproducible at
                    # compile time -- not streamable (rejected as UNSTREAMABLE).
                    abandon()
                    return None
                ctx = make_ctx(decode_filters=None if encode_stage else tuple(vf_filters))
                filter_expr = op.to_ffmpeg_filter(ctx)
                owned_files.extend(ctx.owned_files)
                if filter_expr is None:
                    abandon()
                    return None
                if encode_stage:
                    # A transform following frame effects joins the encode
                    # chain (FrameEncoder -vf), which runs after every
                    # process_frame -- plan order is preserved because this
                    # shape places the filter without the whole-plan fallback.
                    pipe_meta = pipe_meta or running
                    post_vf_filters.append(filter_expr)
                else:
                    vf_filters.append(filter_expr)
                # Audio twin compiled from the SAME pre-op ctx, appended to the
                # same stage the video filter landed -- so audio and video
                # stage placement cannot drift (speed->atempo, freeze->silence
                # splice, silence_removal->aselect keep windows).
                compile_audio_twin(op, ctx, encode_stage)
                running = _predict_with_context(op, running, seg_context)
                if op.changes_duration:
                    duration_changed = True
        except BaseException:
            # A raising compile or prediction (e.g. missing subtitle context,
            # crop exceeding source) must not leak earlier ops' temp files.
            abandon()
            raise

        pipe = pipe_meta or running
        # Final output duration after every transform (decode + encode stage),
        # used to pin the audio graph length to the video timeline. `running`
        # is folded through the whole chain, so this is the encoded output's
        # duration even when an encode-stage transform (post_vf speed) shortens
        # it below the pipe frame count.
        output_total_seconds = running.frame_count / running.fps if running.fps else 0.0
        return StreamingSegmentPlan(
            source_path=segment.source,
            start_second=segment.start,
            end_second=segment.end,
            output_fps=pipe.fps,
            output_width=pipe.width,
            output_height=pipe.height,
            vf_filters=vf_filters,
            effect_schedule=effect_schedule,
            post_vf_filters=post_vf_filters,
            owned_temp_files=owned_files,
            output_total_frames=pipe.frame_count,
            af_filters=af_filters,
            post_af_filters=post_af_filters,
            output_total_seconds=output_total_seconds,
            final_fps=running.fps,
            final_width=running.width,
            final_height=running.height,
        )

    def _assert_transitions_runnable(self, plans: list[StreamingSegmentPlan]) -> None:
        """Raise before decode if any transition is structurally invalid or too long.

        Mirrors :func:`_transition_structure_errors` /
        :func:`_transition_duration_errors` but measures each segment against
        its compiled plan's predicted post-op duration -- the exact length the
        per-segment realize pass produces -- so a plan that passes here cannot
        fail the xfade pass at decode. Called by ``run_to_file`` after plan
        compilation (no frames decoded yet).
        ``repair`` clamps these mechanically; here they hard-raise.
        """
        for message, err in _transition_structure_errors(list(self.segments)):
            raise PlanValidationError(message, [err])
        for i in range(1, len(self.segments)):
            spec = self.segments[i].transition_in
            if spec is None:
                continue
            left_frames, left_fps = _plan_output_frames(plans[i - 1])
            right_frames, right_fps = _plan_output_frames(plans[i])
            fps = right_fps or left_fps
            overlap = _transition_overlap_frames(spec, fps)
            limit_frames = min(left_frames, right_frames)
            if overlap >= limit_frames:
                limit_seconds = round(limit_frames / fps, 4) if fps else 0.0
                message, err = _transition_too_long_error(
                    i,
                    spec,
                    overlap,
                    limit_frames,
                    limit_seconds,
                    "repair the plan or shorten the transition.",
                )
                raise PlanValidationError(message, [err])


def _plan_output_frames(plan: StreamingSegmentPlan) -> tuple[int, float]:
    """A compiled segment plan's predicted post-op ``(frame_count, fps)``.

    ``output_total_frames`` is the prediction folded through every transform,
    over ``final_fps`` (the rate after encode-stage resamplers; falls back to
    ``output_fps``). The transition guard measures overlap in *frames* against
    this -- the exact length the per-segment realize pass produces -- so a plan
    that passes the guard cannot fail the frame-based xfade constraint.
    """
    fps = plan.final_fps or plan.output_fps
    frames = plan.output_total_frames
    if frames <= 0:
        frames = round((plan.end_second - plan.start_second) * fps)
    return frames, fps


def _effect_frame_range(op: Effect, fps: float, total_frames: int) -> tuple[int, int]:
    """Resolve an effect's ``window`` to a ``(start_frame, end_frame)`` pair."""
    if op.window is None:
        return 0, total_frames
    start_s = op.window.start
    stop_s = op.window.stop
    start_f = round(start_s * fps) if start_s is not None else 0
    end_f = round(stop_s * fps) if stop_s is not None else total_frames
    return start_f, end_f

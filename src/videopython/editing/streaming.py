"""Streaming video processing pipeline.

Connects ffmpeg decode -> per-frame effect chain -> ffmpeg encode, keeping
only one frame in memory at a time. Memory usage is O(1) with respect to
video length for streamable pipelines.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, get_args

import numpy as np
from tqdm import tqdm

from videopython.base import _ffmpeg
from videopython.base._dimensions import require_even
from videopython.base.exceptions import PlanError, PlanErrorCode
from videopython.base.video import ALLOWED_VIDEO_FORMATS, ALLOWED_VIDEO_PRESETS, FrameIterator
from videopython.editing.effects import Effect
from videopython.editing.operation import Operation

logger = logging.getLogger(__name__)


class StreamingClass(str, Enum):
    """How an op executes on the streaming engine -- its memory class.

    ``FILTER`` and ``FRAME_EFFECT`` stream in O(1) memory w.r.t. video
    length. ``UNSTREAMABLE`` means the op (or its plan position) has no
    streaming strategy; since streaming is the only engine, such plans are
    rejected with structured ``STREAMING_FALLBACK`` errors instead of run.
    """

    FILTER = "filter"
    """Compiles to an ffmpeg filter -- the decode chain, or the encode chain
    when ordered after frame effects."""
    FRAME_EFFECT = "frame_effect"
    """Shape-preserving per-frame Python (``streaming_init`` + ``process_frame``)."""
    UNSTREAMABLE = "unstreamable"
    """No streaming strategy at this plan position; the plan is rejected."""


@dataclass(frozen=True)
class OpStreamability:
    """Streaming classification for a single op within a plan."""

    location: str
    """Path into the plan, e.g. ``'segments[1].operations[0]'``."""
    op: str
    """The op discriminator, e.g. ``'add_subtitles'``."""
    streaming_class: StreamingClass
    reason: str | None = None
    """Why the op cannot stream; ``None`` unless ``UNSTREAMABLE``."""

    @property
    def streams(self) -> bool:
        return self.streaming_class is not StreamingClass.UNSTREAMABLE


@dataclass(frozen=True)
class StreamabilityReport:
    """Per-op streaming classification for a whole plan.

    Built by :meth:`VideoEdit.streamability` from the plan structure alone --
    no source files, metadata, or runtime context needed -- so a consumer can
    gate job admission on it before downloading anything. ``streamable`` is
    the plan-level verdict: one ``UNSTREAMABLE`` op rejects the entire plan
    (streaming is the only engine).
    """

    entries: tuple[OpStreamability, ...]

    @property
    def streamable(self) -> bool:
        """True when the plan runs (no op is unstreamable at its position)."""
        return all(e.streams for e in self.entries)

    @property
    def fallbacks(self) -> tuple[OpStreamability, ...]:
        """The unstreamable ops, in plan order."""
        return tuple(e for e in self.entries if not e.streams)

    def errors(self) -> list[PlanError]:
        """The fallbacks as structured ``STREAMING_FALLBACK`` plan errors.

        The same shape :meth:`VideoEdit.check` returns, so an LLM refine loop
        can treat "would not stream" exactly like any other plan violation.
        """
        return [
            PlanError(code=PlanErrorCode.STREAMING_FALLBACK, location=e.location, op=e.op, detail=e.reason)
            for e in self.fallbacks
        ]


def analyze_streamability(
    segment_operations: Sequence[Sequence[Operation]],
    post_operations: Sequence[Operation],
    *,
    has_transitions: bool = False,
) -> StreamabilityReport:
    """Classify every op in a plan by streaming class, in plan order.

    Mirrors the decision points of ``VideoEdit._build_streaming_plan`` and the
    post-op folding in ``VideoEdit.run_to_file`` exactly -- this function is
    the single documented source of those rules. Both deciders treat the
    ``streamable`` ClassVar as the authoritative declaration (a flag-False
    transform never streams, even with a working ``to_ffmpeg_filter``); the
    one divergence the flag cannot express -- flag True but the filter
    compiles to ``None`` -- is caught at runtime by the strict-mode drift
    guard, and a registry test pins flag-True transforms to an actual
    ``to_ffmpeg_filter`` override. Purely structural: no disk access and no
    runtime context.

    Segment transitions (``SegmentConfig.transition_in``) do not appear here:
    a transition is a native ffmpeg ``xfade``/``acrossfade`` pass over two
    realized per-segment files (each segment streams to a temp file exactly as
    it would without a transition), so it is FILTER-class by construction and
    never makes a plan unstreamable or changes any op's class. The one transition
    fault that can reject a plan -- an overlap longer than an adjacent
    segment's predicted post-op duration -- is duration-dependent, so it is
    reported as ``TRANSITION_TOO_LONG`` by :meth:`VideoEdit.check` (which has
    the predicted durations) and raised by ``run_to_file`` before
    decode, not classified structurally here.
    """
    entries: list[OpStreamability] = []
    segment_has_encode_stage: list[bool] = []
    for i, ops in enumerate(segment_operations):
        seen_effect = False
        seen_encode_stage = False
        seen_duration_change = False
        for j, op in enumerate(ops):
            location = f"segments[{i}].operations[{j}]"
            if isinstance(op, Effect):
                if op.streamable and op.requires and seen_duration_change:
                    # The builder rejects context-consuming ops behind a
                    # duration-changing transform: segment-local context is
                    # not re-mapped through the time warp yet.
                    entries.append(
                        OpStreamability(
                            location,
                            op.op,
                            StreamingClass.UNSTREAMABLE,
                            reason=(
                                "context-requiring op follows a duration-changing transform "
                                "(speed_change/freeze_frame); time-based context is not re-mapped "
                                "through the warp yet -- move the op before it to stream"
                            ),
                        )
                    )
                    continue
                if op.streamable and op.compiles_to_filter:
                    # Filter-class effect (add_subtitles): joins the decode
                    # filter chain at this plan position -- or the encode
                    # chain when frame effects precede it -- so it streams in
                    # plan order either way and does not block later
                    # transforms the way a scheduled frame effect does. The
                    # builder mirrors this exactly.
                    entries.append(OpStreamability(location, op.op, StreamingClass.FILTER))
                    if seen_effect:
                        seen_encode_stage = True
                    continue
                if not op.streamable:
                    entries.append(
                        OpStreamability(
                            location,
                            op.op,
                            StreamingClass.UNSTREAMABLE,
                            reason="effect has no streaming implementation (streamable=False)",
                        )
                    )
                elif seen_encode_stage:
                    entries.append(
                        OpStreamability(
                            location,
                            op.op,
                            StreamingClass.UNSTREAMABLE,
                            reason=(
                                "frame effect follows encode-stage filters (subtitles or transforms "
                                "ordered after effects) in plan order; those filters run after every "
                                "frame effect, so plan order cannot be preserved -- move the effect "
                                "earlier to stream"
                            ),
                        )
                    )
                else:
                    entries.append(OpStreamability(location, op.op, StreamingClass.FRAME_EFFECT))
                seen_effect = True
            elif op.requires and seen_duration_change:
                entries.append(
                    OpStreamability(
                        location,
                        op.op,
                        StreamingClass.UNSTREAMABLE,
                        reason=(
                            "context-requiring transform follows a duration-changing transform; "
                            "time-based context is not re-mapped through the warp yet -- move it "
                            "before the duration change to stream"
                        ),
                    )
                )
            elif op.requires and not op.streamable:
                entries.append(
                    OpStreamability(
                        location,
                        op.op,
                        StreamingClass.UNSTREAMABLE,
                        reason=(
                            f"transform requires runtime context {sorted(op.requires)} and has no streaming strategy"
                        ),
                    )
                )
            elif op.streamable and op.compiles_from_source and (seen_effect or seen_encode_stage):
                entries.append(
                    OpStreamability(
                        location,
                        op.op,
                        StreamingClass.UNSTREAMABLE,
                        reason=(
                            "the op's compile-time detection pass cannot reproduce frames behind "
                            "per-frame Python effects -- move it before the effects to stream"
                        ),
                    )
                )
            elif op.streamable:
                # Transforms compile to filters at any plan position: the
                # decode chain normally, the encode chain (after every
                # process_frame) when frame effects precede them.
                entries.append(OpStreamability(location, op.op, StreamingClass.FILTER))
                if seen_effect or seen_encode_stage:
                    seen_encode_stage = True
                if op.changes_duration:
                    seen_duration_change = True
            else:
                entries.append(
                    OpStreamability(
                        location,
                        op.op,
                        StreamingClass.UNSTREAMABLE,
                        reason="transform has no ffmpeg filter compilation",
                    )
                )

        segment_has_encode_stage.append(seen_encode_stage)

    multi_segment = len(segment_operations) > 1
    any_encode_stage = any(segment_has_encode_stage)
    for j, op in enumerate(post_operations):
        location = f"post_operations[{j}]"
        if has_transitions:
            # Post-operations fold into the per-segment frame schedules, but a
            # transition re-encodes the overlapped seam in a SEPARATE xfade
            # pass after those schedules run -- so a post-op envelope spanning
            # the seam cannot be represented (it would be applied to the
            # pre-blend frames and then doubled at the overlap). Reject the
            # combination rather than mis-time it; move the op into a segment.
            entries.append(
                OpStreamability(
                    location,
                    op.op,
                    StreamingClass.UNSTREAMABLE,
                    reason=(
                        "post-operations cannot fold across a plan that has segment transitions: the "
                        "overlapped seam is re-encoded in a separate xfade pass, so an envelope across "
                        "it cannot be represented -- move the op into a segment or drop the transition"
                    ),
                )
            )
        elif op.requires:
            entries.append(
                OpStreamability(
                    location,
                    op.op,
                    StreamingClass.UNSTREAMABLE,
                    reason=(
                        f"post-operation requires runtime context {sorted(op.requires)}, which is not "
                        "re-based onto the assembled timeline -- move it into the segment to stream"
                    ),
                )
            )
        elif isinstance(op, Effect) and op.streamable and op.compiles_to_filter:
            entries.append(
                OpStreamability(
                    location,
                    op.op,
                    StreamingClass.UNSTREAMABLE,
                    reason=(
                        "filter-class post-operations cannot fold into the segment schedule -- "
                        "move the op into the segment to stream"
                    ),
                )
            )
        elif isinstance(op, Effect) and op.streamable and any_encode_stage:
            entries.append(
                OpStreamability(
                    location,
                    op.op,
                    StreamingClass.UNSTREAMABLE,
                    reason=(
                        "post-operations fold into the per-segment frame-effect schedules, which run "
                        "before a segment's encode-stage filters -- move the op into the segments "
                        "(before the encode-stage ops) to stream"
                    ),
                )
            )
        elif isinstance(op, Effect) and op.streamable and op.audio_coupled and multi_segment:
            entries.append(
                OpStreamability(
                    location,
                    op.op,
                    StreamingClass.UNSTREAMABLE,
                    reason=(
                        "audio-coupled post-operations (fade/volume_adjust) cannot fold across a "
                        "multi-segment concat: each segment's audio is processed independently, so "
                        "the gain envelope would restart at every boundary -- apply it per segment "
                        "or use a single-segment plan"
                    ),
                )
            )
        elif isinstance(op, Effect) and op.streamable:
            # Folds into the per-segment schedules with globally-rebased
            # frame offsets (multi-segment included since 0.42.0).
            entries.append(OpStreamability(location, op.op, StreamingClass.FRAME_EFFECT))
        elif isinstance(op, Effect):
            entries.append(
                OpStreamability(
                    location,
                    op.op,
                    StreamingClass.UNSTREAMABLE,
                    reason="effect has no streaming implementation (streamable=False)",
                )
            )
        else:
            entries.append(
                OpStreamability(
                    location,
                    op.op,
                    StreamingClass.UNSTREAMABLE,
                    reason="transforms cannot stream as post-operations -- move the transform into the segment",
                )
            )

    return StreamabilityReport(entries=tuple(entries))


@dataclass
class EffectScheduleEntry:
    """An effect with its active frame range for streaming.

    ``context`` carries the effect's resolved ``requires`` values (already
    re-based onto the segment-local timeline by the plan builder), forwarded
    as keyword arguments to ``streaming_init``. Empty for context-free
    effects.

    For a post-operation folded across a multi-segment plan, the effect's
    window lives on the assembled (concatenated) timeline: ``index_offset``
    is the number of window frames consumed by previous segments (so
    ``process_frame`` indices continue across the concat boundary) and
    ``total_effect_frames`` is the global window length ``streaming_init``
    must size envelopes to. Defaults describe the ordinary segment-local
    case.
    """

    effect: Effect
    start_frame: int
    end_frame: int  # exclusive
    context: dict[str, Any] = field(default_factory=dict)
    index_offset: int = 0
    total_effect_frames: int | None = None


@dataclass
class StreamingSegmentPlan:
    """Describes how to stream-process one video segment.

    ``vf_filters`` run at decode time, before every scheduled effect;
    ``post_vf_filters`` run at encode time (``FrameEncoder``'s ``-vf``),
    after every scheduled effect -- where a filter-class op that follows
    frame effects in plan order lands (e.g. ``[fade, add_subtitles]``).

    ``owned_temp_files`` are compile-time artifacts referenced by the filter
    chains (e.g. the ``.ass`` file behind a ``subtitles=`` entry); the runner
    deletes them once streaming finishes or the plan is abandoned.
    """

    source_path: Path
    start_second: float
    end_second: float
    output_fps: float
    output_width: int
    output_height: int
    vf_filters: list[str] = field(default_factory=list)
    effect_schedule: list[EffectScheduleEntry] = field(default_factory=list)
    post_vf_filters: list[str] = field(default_factory=list)
    owned_temp_files: list[Path] = field(default_factory=list)
    final_fps: float = 0.0
    """Frame rate after the encode-stage filters (== ``output_fps`` unless a
    ``post_vf`` transform resamples). ``0`` means "same as output_fps"."""
    final_width: int = 0
    final_height: int = 0
    """Dimensions after the encode-stage filters; ``0`` means "same as
    output_width/output_height"."""
    output_total_frames: int = 0
    """Predicted output frame count, folded through every compiled transform.

    Authoritative for effect scheduling, envelope sizing, and progress --
    duration-changing filters (speed, freeze) make ``(end_second -
    start_second) * output_fps`` wrong. ``0`` means "no transform changed
    duration; derive from the cut".
    """
    af_filters: list[str] = field(default_factory=list)
    """Decode-stage audio filter expressions (the audio twin of
    ``vf_filters``): each a single-in/single-out ``filter_complex`` fragment
    compiled from an op's :meth:`Operation.to_ffmpeg_audio_filter`, in plan
    order. Run on the segment's source audio (a second ``-i`` input) before the
    effect-stage audio filters. Built at exactly the points ``vf_filters`` is,
    so audio and video stage placement cannot drift."""
    post_af_filters: list[str] = field(default_factory=list)
    """Encode-stage audio filters (twins of transforms/effects ordered after
    the frame effects), the audio twin of ``post_vf_filters``; appended to the
    same single audio graph after ``af_filters``."""
    output_total_seconds: float = 0.0
    """Predicted output duration in seconds (``output_total_frames /
    final_fps``), used to pin the audio graph's length to the video timeline
    (``atrim``+``apad``) so a few-sample codec/PTS drift cannot exceed the A/V
    tolerance. ``0`` means "derive from the frame count"."""


@dataclass(frozen=True)
class SegmentAudio:
    """The segment's source-audio input + compiled audio filter graph.

    Replaces the old pre-rendered WAV ``audio_path``: the source file is added
    as a second ffmpeg input (input-side ``-ss``/``-t`` trimming it to exactly
    the segment the video decode reads), and its audio stream is routed through
    a labeled ``filter_complex`` built from the plan's ``af_filters`` /
    ``post_af_filters``. When the source has no audio stream, silence is
    synthesised natively via an ``anullsrc`` input (preserving the
    "no audio -> silent track" contract without a Python round-trip).
    """

    source_path: Path
    start_second: float
    duration: float
    af_filters: tuple[str, ...]
    post_af_filters: tuple[str, ...]
    has_audio_stream: bool
    output_seconds: float = 0.0
    """Predicted output duration; pins the audio graph length (``atrim``+``apad``)
    to the video timeline so AAC/PTS drift cannot exceed the A/V tolerance.
    ``0`` -> no explicit pin."""


def source_has_audio_stream(source_path: Path) -> bool:
    """True when ``source_path`` carries at least one audio stream.

    Drives the native silent-track contract: a source with no audio stream
    gets an ``anullsrc`` silence input instead of ``-i source`` audio.
    """
    try:
        info = _ffmpeg.probe(source_path)
    except Exception:
        return False
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def build_audio_filter_complex(
    audio: SegmentAudio, *, input_index: int = 1, sample_rate: int = 44100
) -> tuple[list[str], list[str], str]:
    """Compile the per-segment audio inputs + ``filter_complex`` graph.

    Returns ``(input_args, graph_statements, out_label)``:

    - ``input_args`` are the ffmpeg ``-i`` argv for the audio source -- either
      ``-ss start -t dur -i source`` (real audio) or ``-f lavfi -i anullsrc``
      (synthesised silence). ``input_index`` is the ffmpeg input index this
      stream lands at: ``1`` for :class:`FrameEncoder` (after the ``pipe:0``
      video), ``0`` when the audio stream is the sole input.
    - ``graph_statements`` are the ``;``-joined ``filter_complex`` statements
      wiring input ``[<i>:a]`` through ``af_filters`` then ``post_af_filters``,
      a length pin, and an ``aresample`` to a final ``[aout]``.
    - ``out_label`` is the label to ``-map`` (always ``"[aout]"`` -- the graph
      always carries at least the length pin + ``aresample``).

    Built as a LABELED graph (not a one-off ``-af`` string) so P1.11 can splice
    additional inputs/labels (``amix`` music bed, sidechain ducking) onto the
    same graph: each op fragment is wrapped ``[prev]<fragment>[next]``, so a new
    input slots in as another labeled source feeding an ``amix`` node.
    """
    if audio.has_audio_stream:
        input_args = [
            "-ss",
            f"{audio.start_second:.6f}",
            "-t",
            f"{audio.duration:.6f}",
            "-i",
            str(audio.source_path),
        ]
    else:
        # Native silent track: an anullsrc of exactly the trimmed duration.
        input_args = [
            "-f",
            "lavfi",
            "-t",
            f"{audio.duration:.6f}",
            "-i",
            f"anullsrc=channel_layout=stereo:sample_rate={sample_rate}",
        ]

    stages = list(audio.af_filters) + list(audio.post_af_filters)
    # Pin the graph to the predicted output length so a few-sample AAC/PTS drift
    # cannot exceed the A/V tolerance: trim to, then pad up to, output_seconds.
    if audio.output_seconds > 0:
        stages.append(f"atrim=end={audio.output_seconds:.6f}")
        stages.append(f"apad=whole_dur={audio.output_seconds:.6f}")
    # aresample=async=1 absorbs the input-side -ss / pipe PTS mismatch.
    stages.append("aresample=async=1")

    statements: list[str] = []
    prev = f"{input_index}:a"
    for i, stage in enumerate(stages):
        out = "aout" if i == len(stages) - 1 else f"a{i}"
        statements.append(f"[{prev}]{stage}[{out}]")
        prev = out
    return input_args, statements, "[aout]"


class FrameEncoder:
    """Writes raw RGB frames to an ffmpeg encode process via stdin pipe.

    Use as a context manager to ensure proper cleanup of the ffmpeg process.
    Segment audio rides the SAME ffmpeg invocation: a second ``-i`` input
    (the original source, input-side trimmed, or an ``anullsrc`` silence)
    routed through a labeled ``filter_complex`` built from the plan's compiled
    audio filters. There is no second pass and no pre-rendered WAV.
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        audio: SegmentAudio | None = None,
        format: str = "mp4",
        preset: str = "medium",
        crf: int = 23,
        vf_filters: list[str] | None = None,
    ):
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._audio = audio
        self._format = format
        self._preset = preset
        self._crf = crf
        self._vf_filters = vf_filters or []
        self._stack: ExitStack | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def _build_command(self) -> list[str]:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            # Raw video input via stdin (input index 0)
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self._width}x{self._height}",
            "-framerate",
            str(self._fps),
            "-i",
            "pipe:0",
        ]
        audio_graph: list[str] = []
        audio_map = ""
        if self._audio is not None:
            audio_inputs, audio_graph, audio_map = build_audio_filter_complex(self._audio)
            # Audio source is input index 1.
            cmd.extend(audio_inputs)

        # The video graph is the encode-stage -vf (frames arrive pre-processed
        # on pipe:0). The audio graph is a separate filter_complex chain over
        # input 1; the two never share labels, so -vf and -filter_complex
        # coexist. The rawvideo pipe's PTS start at zero, so time-based video
        # filters (subtitles=) see the same segment-local timeline the decode
        # side uses.
        if self._vf_filters:
            cmd.extend(["-vf", ",".join(self._vf_filters)])
        if self._audio is not None:
            cmd.extend(["-filter_complex", ";".join(audio_graph)])
            cmd.extend(["-map", "0:v", "-map", audio_map])

        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                self._preset,
                "-crf",
                str(self._crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-vsync",
                "cfr",
            ]
        )
        if self._audio is not None:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-an"])

        cmd.append(str(self._output_path))
        return cmd

    def __enter__(self) -> FrameEncoder:
        require_even(self._width, self._height)
        self._stack = ExitStack()
        self._process = self._stack.enter_context(_ffmpeg.popen_encode(self._build_command()))
        return self

    def write_frame(self, frame: np.ndarray) -> None:
        """Write a single RGB frame to the encoder."""
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("FrameEncoder not started -- use as context manager")
        self._process.stdin.write(frame.tobytes())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        stack = self._stack
        self._stack = None
        self._process = None
        if stack is None:
            return None
        return stack.__exit__(exc_type, exc_val, exc_tb)


def segment_audio_from_plan(plan: StreamingSegmentPlan) -> SegmentAudio:
    """Build the segment's :class:`SegmentAudio` from its compiled plan.

    Trims the source on the input side to exactly the cut the video decode
    reads (``-ss start_second -t (end-start)``), so the source audio and the
    rawvideo pipe see the same segment, and probes the source once for the
    silent-track decision. The compiled ``af_filters`` / ``post_af_filters``
    ride this graph in the single FrameEncoder invocation -- no WAV round-trip.
    """
    out_seconds = plan.output_total_seconds
    if out_seconds <= 0 and plan.output_total_frames > 0:
        out_seconds = plan.output_total_frames / (plan.final_fps or plan.output_fps)
    return SegmentAudio(
        source_path=plan.source_path,
        start_second=plan.start_second,
        duration=plan.end_second - plan.start_second,
        af_filters=tuple(plan.af_filters),
        post_af_filters=tuple(plan.post_af_filters),
        has_audio_stream=source_has_audio_stream(plan.source_path),
        output_seconds=out_seconds,
    )


def stream_segment(
    plan: StreamingSegmentPlan,
    output_path: Path,
    with_audio: bool = True,
    format: str = "mp4",
    preset: str = "medium",
    crf: int = 23,
) -> Path:
    """Execute a streaming pipeline for a single segment.

    Reads frames from source via ffmpeg, applies effects per-frame, and
    writes directly to output via ffmpeg encode. Peak memory is ~2 frames.
    Segment audio is compiled into the SAME ffmpeg invocation (a second
    ``-i`` input through ``-filter_complex``) -- no pre-rendered WAV.

    Args:
        plan: Streaming segment plan describing source, effects, audio filters, and output params.
        output_path: Destination file path.
        with_audio: When True (default) the output carries the compiled audio
            track; False writes a video-only file.
        format: Output container format.
        preset: x264 encoding preset.
        crf: Constant rate factor.

    Returns:
        Path to the output file.
    """
    if format not in get_args(ALLOWED_VIDEO_FORMATS):
        raise ValueError(f"Unsupported format: {format}")
    if preset not in get_args(ALLOWED_VIDEO_PRESETS):
        raise ValueError(f"Unsupported preset: {preset}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Total output frames for progress, streaming_init sizing, and the
    # open-ended schedule logic. The plan's folded prediction is
    # authoritative (duration-changing filters break the cut-length
    # derivation); the derivation remains as a fallback for hand-built plans.
    total_frames = plan.output_total_frames
    if total_frames <= 0:
        duration = plan.end_second - plan.start_second
        total_frames = round(duration * plan.output_fps)

    # Initialize effects for streaming. Runs before this segment's decode,
    # so a context-requiring effect with missing context fails before any
    # frame work on this segment.
    for entry in plan.effect_schedule:
        n_effect_frames = (
            entry.total_effect_frames if entry.total_effect_frames is not None else entry.end_frame - entry.start_frame
        )
        entry.effect.streaming_init(
            n_effect_frames, plan.output_fps, plan.output_width, plan.output_height, **entry.context
        )

    audio = segment_audio_from_plan(plan) if with_audio else None

    try:
        with FrameIterator(
            plan.source_path,
            start_second=plan.start_second,
            end_second=plan.end_second,
            vf_filters=plan.vf_filters,
            output_fps=plan.output_fps,
            output_width=plan.output_width,
            output_height=plan.output_height,
        ) as decoder:
            with FrameEncoder(
                output_path,
                width=plan.output_width,
                height=plan.output_height,
                fps=plan.output_fps,
                audio=audio,
                format=format,
                preset=preset,
                crf=crf,
                vf_filters=plan.post_vf_filters,
            ) as encoder:
                logger.info("Streaming frames...")
                frame_count = 0
                for _, frame in tqdm(decoder, desc="Streaming", total=total_frames):
                    for entry in plan.effect_schedule:
                        # total_frames is an estimate (round(duration * fps));
                        # ffmpeg can emit one more frame on rounding ties. An
                        # entry covering the full estimated range is treated
                        # as open-ended so trailing frames don't escape it
                        # (an unfaded frame popping after a fade-out), with
                        # the window-local index clamped so envelope arrays
                        # sized to the estimate stay in bounds.
                        open_ended = entry.end_frame >= total_frames
                        if entry.start_frame <= frame_count and (frame_count < entry.end_frame or open_ended):
                            local_index = entry.index_offset + min(frame_count, entry.end_frame - 1) - entry.start_frame
                            frame = entry.effect.process_frame(frame, local_index)

                    encoder.write_frame(frame)
                    frame_count += 1

        return output_path
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise


def concat_files(segment_files: list[Path], output_path: Path) -> Path:
    """Concatenate encoded video files using ffmpeg concat demuxer (no re-encode).

    All input files must have identical codec parameters.
    """
    output_path = Path(output_path)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for seg_file in segment_files:
            f.write(f"file '{seg_file}'\n")
        list_path = Path(f.name)

    try:
        _ffmpeg.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_path),
            ]
        )
        return output_path
    finally:
        list_path.unlink(missing_ok=True)


# ----------------------------------------------------------------- transitions

# Curated subset of ffmpeg ``xfade`` ``transition=`` modes. The single source
# of truth for both the ``TransitionSpec.type`` Literal (so the strict LLM
# grammar stays tight) and the filter builder. Every name here is a literal
# ffmpeg ``xfade=transition=<name>`` value -- no translation layer.
TRANSITION_TYPES: tuple[str, ...] = (
    "fade",
    "dissolve",
    "wipeleft",
    "wiperight",
    "wipeup",
    "wipedown",
    "slideleft",
    "slideright",
)


def xfade_filter(transition_type: str, duration: float, offset: float, *, in_a: str = "0:v", in_b: str = "1:v") -> str:
    """Build the one ``xfade`` filter expression for the transition path.

    The single source of the seam pixels: ``run_to_file``'s file pass
    (:func:`stream_transition_pair`) routes through this.
    ``offset`` is in seconds against the LEFT input's exact (probed)
    duration -- the caller derives it from the realized file, never the
    prediction, to avoid a one-frame seam.
    """
    return f"[{in_a}][{in_b}]xfade=transition={transition_type}:duration={duration}:offset={offset}"


def acrossfade_filter(duration: float, *, in_a: str = "0:a", in_b: str = "1:a") -> str:
    """The ``acrossfade`` companion to :func:`xfade_filter` for the audio seam."""
    return f"[{in_a}][{in_b}]acrossfade=d={duration}"


def stream_transition_pair(
    left_file: Path,
    right_file: Path,
    transition_type: str,
    duration: float,
    offset: float,
    output_path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    left_frame_count: int,
    right_frame_count: int,
    left_has_audio: bool,
    right_has_audio: bool,
    crossfade_audio: bool,
    format: str = "mp4",
    preset: str = "medium",
    crf: int = 23,
) -> Path:
    """xfade/acrossfade two realized segment files into one output file.

    The file-path half of the transition mechanism: each input is a finished
    per-segment render (effects + audio already baked by ``stream_segment``),
    so this pass is a pure ffmpeg ``filter_complex`` over two decode inputs --
    no two-decoder-with-Python-effects machinery. ``offset`` is seconds against
    ``left_file``'s exact duration (``left_total - duration``); the video seam
    is produced by the shared :func:`xfade_filter` builder. When both inputs
    carry audio and ``crossfade_audio`` is set the audio is ``acrossfade``-d
    across the same overlap; otherwise a hard audio butt-join (``concat``)
    keeps the streams aligned to the shortened video timeline.

    Each video input is normalized to CFR ``fps`` and trimmed to its PROBED
    ``*_frame_count`` before xfade. A concat-demuxer (``-c copy``) tail can
    decode one duplicated frame at the join beyond what ffprobe reports, which
    would inflate the xfade output by that frame; trimming to the probed count
    pins the output to ``left + right - round(duration*fps)`` so the realized
    seam matches the predicted shortened timeline exactly.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cap each input to its probed frame count, then re-stamp CFR. A
    # concat-copy duplicate cannot then leak past the prediction (see
    # docstring). The trailing ``fps`` re-establishes the constant frame rate
    # xfade requires (``trim`` alone leaves the rate undefined).
    filters = [
        f"[0:v]trim=end_frame={left_frame_count},setpts=PTS-STARTPTS,fps={fps}[lv]",
        f"[1:v]trim=end_frame={right_frame_count},setpts=PTS-STARTPTS,fps={fps}[rv]",
        f"{xfade_filter(transition_type, duration, offset, in_a='lv', in_b='rv')}[vout]",
    ]
    maps = ["-map", "[vout]"]
    both_audio = left_has_audio and right_has_audio
    if both_audio and crossfade_audio:
        filters.append(f"{acrossfade_filter(duration)}[aout]")
        maps.extend(["-map", "[aout]"])
    elif both_audio:
        # Hard audio butt-join over the same overlap: drop the left's last
        # ``duration`` seconds and concat, so the audio length tracks the
        # shortened (xfade) video timeline rather than summing.
        filters.append(f"[0:a]atrim=end={offset},asetpts=PTS-STARTPTS[la]")
        filters.append("[la][1:a]concat=n=2:v=0:a=1[aout]")
        maps.extend(["-map", "[aout]"])

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(left_file),
        "-i",
        str(right_file),
        "-filter_complex",
        ";".join(filters),
        *maps,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-vsync",
        "cfr",
        "-r",
        str(fps),
    ]
    if "-map" in maps and "[aout]" in maps:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.append("-an")
    cmd.append(str(output_path))

    require_even(width, height)
    try:
        _ffmpeg.run(cmd)
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise
    return output_path

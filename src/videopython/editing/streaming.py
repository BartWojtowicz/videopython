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

from videopython.audio import Audio
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
        if op.requires:
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
    audio_ops: list[tuple[Operation, float, float, dict[str, Any]]] = field(default_factory=list)
    """Transforms with an audio-domain twin: ``(op, predicted post-op
    duration, fps at the op's chain position, resolved requires-context)``;
    ``_load_segment_audio`` replays them in plan order."""
    post_audio_ops: list[tuple[Operation, float, float, dict[str, Any]]] = field(default_factory=list)
    """Encode-stage audio twins (transforms ordered after the frame
    effects), replayed after the effect envelopes."""


class FrameEncoder:
    """Writes raw RGB frames to an ffmpeg encode process via stdin pipe.

    Use as a context manager to ensure proper cleanup of the ffmpeg process.
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        audio_path: Path | None = None,
        format: str = "mp4",
        preset: str = "medium",
        crf: int = 23,
        vf_filters: list[str] | None = None,
    ):
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._audio_path = audio_path
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
            # Raw video input via stdin
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
        if self._audio_path is not None:
            cmd.extend(["-i", str(self._audio_path)])

        if self._vf_filters:
            # Encode-stage filters: run after every per-frame effect (frames
            # arrive via stdin already processed). The rawvideo pipe's PTS
            # start at zero, so time-based filters (subtitles=) see the same
            # segment-local timeline the decode side uses.
            cmd.extend(["-vf", ",".join(self._vf_filters)])
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
        if self._audio_path is not None:
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


def stream_segment(
    plan: StreamingSegmentPlan,
    output_path: Path,
    audio: Audio | None = None,
    format: str = "mp4",
    preset: str = "medium",
    crf: int = 23,
) -> Path:
    """Execute a streaming pipeline for a single segment.

    Reads frames from source via ffmpeg, applies effects per-frame, and
    writes directly to output via ffmpeg encode. Peak memory is ~2 frames.

    Args:
        plan: Streaming segment plan describing source, effects, and output params.
        output_path: Destination file path.
        audio: Pre-processed audio to mux with the output. If None, output has no audio.
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

    # Save audio to temp file for muxing
    audio_path = None
    temp_audio_file = None
    if audio is not None:
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.save(temp_audio_file.name, format="wav")
        audio_path = Path(temp_audio_file.name)

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
                audio_path=audio_path,
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
    finally:
        if temp_audio_file is not None:
            Path(temp_audio_file.name).unlink(missing_ok=True)


def stream_segment_to_frames(plan: StreamingSegmentPlan) -> np.ndarray:
    """Run a segment's streaming pipeline into memory instead of a file.

    The same scheduler ``stream_segment`` uses -- decode through the plan's
    vf chain, per-frame effects in plan order, encode-stage filters via a
    lossless rawvideo pass -- collecting RGB frames instead of encoding.
    This is what makes ``VideoEdit.run`` a view over the streaming engine
    rather than a second execution path.
    """
    total_frames = plan.output_total_frames
    if total_frames <= 0:
        total_frames = round((plan.end_second - plan.start_second) * plan.output_fps)

    for entry in plan.effect_schedule:
        n_effect_frames = (
            entry.total_effect_frames if entry.total_effect_frames is not None else entry.end_frame - entry.start_frame
        )
        entry.effect.streaming_init(
            n_effect_frames, plan.output_fps, plan.output_width, plan.output_height, **entry.context
        )

    frames: list[np.ndarray] = []
    with FrameIterator(
        plan.source_path,
        start_second=plan.start_second,
        end_second=plan.end_second,
        vf_filters=plan.vf_filters,
        output_fps=plan.output_fps,
        output_width=plan.output_width,
        output_height=plan.output_height,
    ) as decoder:
        frame_count = 0
        for _, frame in tqdm(decoder, desc="Streaming (in-memory)", total=total_frames):
            for entry in plan.effect_schedule:
                open_ended = entry.end_frame >= total_frames
                if entry.start_frame <= frame_count and (frame_count < entry.end_frame or open_ended):
                    local_index = entry.index_offset + min(frame_count, entry.end_frame - 1) - entry.start_frame
                    frame = entry.effect.process_frame(frame, local_index)
            frames.append(frame)
            frame_count += 1

    stacked = np.stack(frames) if frames else np.empty((0, plan.output_height, plan.output_width, 3), dtype=np.uint8)
    if not plan.post_vf_filters:
        return stacked

    # Encode-stage filters: a lossless rawvideo->rawvideo pass, the in-memory
    # twin of FrameEncoder's -vf.
    n, height, width = stacked.shape[:3]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(plan.output_fps),
        "-i",
        "pipe:0",
        "-vf",
        ",".join(plan.post_vf_filters),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    out = _ffmpeg.run(cmd, stdin=stacked.tobytes())
    return out_frames_reshape(out, plan, n)


def out_frames_reshape(raw: bytes, plan: StreamingSegmentPlan, n_in: int) -> np.ndarray:
    """Reshape a rawvideo byte stream using the plan's final geometry.

    Encode-stage transforms may change dims/fps; the builder records the
    folded final geometry on the plan (``final_width``/``final_height``).
    """
    final_w = plan.final_width or plan.output_width
    final_h = plan.final_height or plan.output_height
    frame_size = final_w * final_h * 3
    n_frames = len(raw) // frame_size
    return np.frombuffer(raw, dtype=np.uint8)[: n_frames * frame_size].reshape(n_frames, final_h, final_w, 3).copy()


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

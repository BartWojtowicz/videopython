"""Streaming video processing pipeline.

Connects ffmpeg decode -> per-frame effect chain -> ffmpeg encode, keeping
only one frame in memory at a time. Memory usage is O(1) with respect to
video length for streamable pipelines.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, get_args

import numpy as np
from tqdm import tqdm

from videopython.audio import Audio
from videopython.base import _ffmpeg
from videopython.base._dimensions import require_even
from videopython.base.video import ALLOWED_VIDEO_FORMATS, ALLOWED_VIDEO_PRESETS, FrameIterator
from videopython.editing.effects import Effect

logger = logging.getLogger(__name__)


@dataclass
class EffectScheduleEntry:
    """An effect with its active frame range for streaming.

    ``context`` carries the effect's resolved ``requires`` values (already
    re-based onto the segment-local timeline by the plan builder), forwarded
    as keyword arguments to ``streaming_init``. Empty for context-free
    effects.
    """

    effect: Effect
    start_frame: int
    end_frame: int  # exclusive
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamingSegmentPlan:
    """Describes how to stream-process one video segment."""

    source_path: Path
    start_second: float
    end_second: float
    output_fps: float
    output_width: int
    output_height: int
    vf_filters: list[str] = field(default_factory=list)
    effect_schedule: list[EffectScheduleEntry] = field(default_factory=list)


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
    ):
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._audio_path = audio_path
        self._format = format
        self._preset = preset
        self._crf = crf
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

    # Estimate total frames for progress and streaming_init
    duration = plan.end_second - plan.start_second
    total_frames = round(duration * plan.output_fps)

    # Initialize effects for streaming. Runs before this segment's decode,
    # so a context-requiring effect with missing context fails before any
    # frame work on this segment.
    for entry in plan.effect_schedule:
        n_effect_frames = entry.end_frame - entry.start_frame
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
                            local_index = min(frame_count, entry.end_frame - 1) - entry.start_frame
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

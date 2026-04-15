"""ffmpeg helper for replacing a video file's audio track without re-encoding video."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class RemuxError(RuntimeError):
    """ffmpeg failed while replacing an audio stream."""


def replace_audio_stream(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> None:
    """Copy ``video_path``'s video stream and mux in ``audio_path`` as the audio track.

    Uses ffmpeg stream-copy for video (no re-encode) and encodes audio to AAC.
    ``-shortest`` trims to the shorter of the two streams so the output duration
    matches the source video when the dubbed audio is slightly longer.

    Args:
        video_path: Source video file (video stream is copied unchanged).
        audio_path: Audio file to use as the new audio track.
        output_path: Destination file. Overwritten if it exists.
        audio_codec: ffmpeg audio codec name. Defaults to ``aac`` (MP4-compatible).
        audio_bitrate: Audio bitrate passed to ffmpeg (``-b:a``).

    Raises:
        FileNotFoundError: If ``video_path`` or ``audio_path`` does not exist.
        RemuxError: If ffmpeg returns a non-zero exit code.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
        "-shortest",
        str(output_path),
    ]

    logger.info("replace_audio_stream: %s + %s -> %s", video_path, audio_path, output_path)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RemuxError(f"ffmpeg failed (exit {result.returncode}): {result.stderr.decode(errors='replace')}")

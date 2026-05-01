"""ffmpeg helper for replacing a video file's audio track without re-encoding video."""

from __future__ import annotations

import io
import logging
import subprocess
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from videopython.base.audio import Audio

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


def replace_audio_stream_from_audio(
    video_path: str | Path,
    audio: Audio,
    output_path: str | Path,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> None:
    """Like ``replace_audio_stream`` but takes an in-memory ``Audio`` and pipes WAV to ffmpeg.

    Avoids the ``Audio.save -> read-from-disk -> ffmpeg`` round-trip used by
    the path-based variant: we serialize the WAV in memory and feed it to
    ffmpeg via stdin. For long dubs this saves a full WAV write+read of the
    output audio (~10 GB for a 2h source).

    Args:
        video_path: Source video file (video stream is copied unchanged).
        audio: ``Audio`` instance to mux in as the new audio track.
        output_path: Destination file. Overwritten if it exists.
        audio_codec: ffmpeg audio codec name. Defaults to ``aac``.
        audio_bitrate: Audio bitrate passed to ffmpeg (``-b:a``).

    Raises:
        FileNotFoundError: If ``video_path`` does not exist.
        RemuxError: If ffmpeg returns a non-zero exit code.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Serialize Audio to WAV bytes in memory. Mirrors Audio.save's WAV writer:
    # int16 samples, header from metadata. We stream these bytes to ffmpeg's
    # stdin as the second input (the first is the video file on disk).
    int_data = (audio.data * np.iinfo(np.int16).max).astype(np.int16)
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(audio.metadata.channels)
        wav_file.setsampwidth(audio.metadata.sample_width)
        wav_file.setframerate(audio.metadata.sample_rate)
        wav_file.writeframes(int_data.tobytes())
    wav_bytes = wav_io.getvalue()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-f",
        "wav",
        "-i",
        "-",
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

    logger.info(
        "replace_audio_stream_from_audio: %s + <stdin wav %d bytes> -> %s",
        video_path,
        len(wav_bytes),
        output_path,
    )
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate(wav_bytes)
    if process.returncode != 0:
        raise RemuxError(f"ffmpeg failed (exit {process.returncode}): {stderr.decode(errors='replace')}")

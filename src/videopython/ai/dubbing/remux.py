"""ffmpeg helper for replacing a video file's audio track without re-encoding video."""

from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from videopython.base import _ffmpeg
from videopython.base.exceptions import FFmpegRunError

if TYPE_CHECKING:
    from videopython.audio import Audio

logger = logging.getLogger(__name__)


class RemuxError(RuntimeError):
    """ffmpeg failed while replacing an audio stream."""


def _build_stream_maps(keep_original_audio: bool) -> list[str]:
    """ffmpeg ``-map`` flags for the video + audio + subtitle streams.

    Convention: dubbed audio (input 1) is the *first* audio track so default
    playback uses it; original audio (input 0) tags onto the back when
    ``keep_original_audio=True`` for editorial A/B. Subtitles from input 0
    are carried with ``?`` so sources without subs don't fail the mux.
    """
    maps = ["-map", "0:v:0", "-map", "1:a:0"]
    if keep_original_audio:
        maps += ["-map", "0:a?"]
    maps += ["-map", "0:s?"]
    return maps


def replace_audio_stream_from_audio(
    video_path: str | Path,
    audio: Audio,
    output_path: str | Path,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    keep_original_audio: bool = False,
) -> None:
    """Copy ``video_path``'s video stream and mux in an in-memory ``Audio`` as the audio track.

    Serializes the WAV in memory and feeds it to ffmpeg via stdin, avoiding an
    ``Audio.save -> read-from-disk -> ffmpeg`` round-trip. For long dubs this
    saves a full WAV write+read of the output audio (~10 GB for a 2h source).
    Video is stream-copied (no re-encode); subtitle streams from ``video_path``
    are carried through unchanged. ``-shortest`` trims to the shorter stream.

    Args:
        video_path: Source video file (video + subtitle streams are copied unchanged).
        audio: ``Audio`` instance to mux in as the new (default) audio track.
        output_path: Destination file. Overwritten if it exists.
        audio_codec: ffmpeg audio codec name. Defaults to ``aac``.
        audio_bitrate: Audio bitrate passed to ffmpeg (``-b:a``).
        keep_original_audio: If True, retain the source audio as a secondary
            track behind the dubbed one. Useful for editorial A/B.

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
        *_build_stream_maps(keep_original_audio),
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
        "-c:s",
        "copy",
        "-shortest",
        str(output_path),
    ]

    logger.info(
        "replace_audio_stream_from_audio: %s + <stdin wav %d bytes> -> %s",
        video_path,
        len(wav_bytes),
        output_path,
    )
    try:
        _ffmpeg.run(cmd, stdin=wav_bytes)
    except FFmpegRunError as e:
        raise RemuxError(str(e)) from e

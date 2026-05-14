"""Internal ffmpeg decode/encode helpers for ``Video``.

Holds the subprocess-heavy bodies of ``Video.from_path`` (decode an
ffmpeg pipe into a frame array) and ``Video.save`` (stream a frame
array to an ffmpeg encode). Keeping these out of ``base/video.py``
lets the data class stay focused on the in-memory frame/audio
container.

Public callers should keep using ``Video.from_path`` and
``Video.save``; this module is internal scaffolding.
"""

from __future__ import annotations

import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Literal, get_args

import numpy as np

from videopython.base import _ffmpeg
from videopython.base._dimensions import require_even
from videopython.base.audio import Audio
from videopython.base.exceptions import (
    AudioLoadError,
    FFmpegRunError,
    VideoLoadError,
    VideoMetadataError,
)

ALLOWED_VIDEO_FORMATS = Literal["mp4", "avi", "mov", "mkv", "webm"]
ALLOWED_VIDEO_PRESETS = Literal[
    "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"
]

# Pre-allocation safety margin for the decode frame array.
FRAME_BUFFER_MULTIPLIER = 1.1
FRAME_BUFFER_PADDING = 10


def decode_video(
    path: str,
    *,
    read_batch_size: int = 100,
    start_second: float | None = None,
    end_second: float | None = None,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray, float, Audio]:
    """Decode a video file into an RGB frame array plus its audio track.

    Returns ``(frames, fps, audio)`` ready to feed straight into the
    ``Video`` constructor. Silent audio is substituted when the source
    has no usable audio stream.

    Raises:
        FileNotFoundError: If ``path`` does not exist (via VideoMetadata).
        VideoLoadError: On ffmpeg failure or unreadable I/O.
        VideoMetadataError: When ffprobe cannot describe the source.
    """
    from videopython.base.video import VideoMetadata

    try:
        metadata = VideoMetadata.from_path(path)

        out_width = width if width is not None else metadata.width
        out_height = height if height is not None else metadata.height
        out_fps = fps if fps is not None else metadata.fps
        total_duration = metadata.total_seconds

        if start_second is not None and start_second < 0:
            raise ValueError("start_second must be non-negative")
        if end_second is not None and end_second > total_duration:
            raise ValueError(f"end_second ({end_second}) exceeds video duration ({total_duration})")
        if start_second is not None and end_second is not None and start_second >= end_second:
            raise ValueError("start_second must be less than end_second")

        segment_duration = total_duration
        if start_second is not None and end_second is not None:
            segment_duration = end_second - start_second
        elif end_second is not None:
            segment_duration = end_second
        elif start_second is not None:
            segment_duration = total_duration - start_second

        estimated_frames = int(segment_duration * out_fps)
        estimated_bytes = estimated_frames * out_height * out_width * 3
        estimated_gb = estimated_bytes / (1024**3)
        if estimated_gb > 10:
            warnings.warn(
                f"Loading this video will use ~{estimated_gb:.1f}GB of RAM. "
                f"For large videos, consider using FrameIterator for memory-efficient streaming.",
                ResourceWarning,
                stacklevel=2,
            )

        ffmpeg_cmd = ["ffmpeg"]

        if start_second is not None:
            ffmpeg_cmd.extend(["-ss", str(start_second)])

        ffmpeg_cmd.extend(["-i", path])

        if end_second is not None and start_second is not None:
            duration = end_second - start_second
            ffmpeg_cmd.extend(["-t", str(duration)])
        elif end_second is not None:
            ffmpeg_cmd.extend(["-t", str(end_second)])

        vf_filters: list[str] = []
        if width is not None or height is not None:
            vf_filters.append(f"scale={out_width}:{out_height}")
        if fps is not None and fps != metadata.fps:
            vf_filters.append(f"fps={out_fps}")
        if vf_filters:
            ffmpeg_cmd.extend(["-vf", ",".join(vf_filters)])

        ffmpeg_cmd.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-vcodec",
                "rawvideo",
                "-avoid_negative_ts",
                "make_zero",
                "-y",
                "pipe:1",
            ]
        )

        frame_size = out_width * out_height * 3

        if start_second is not None and end_second is not None:
            estimated_duration = end_second - start_second
        elif end_second is not None:
            estimated_duration = end_second
        elif start_second is not None:
            estimated_duration = total_duration - start_second
        else:
            estimated_duration = total_duration

        estimated_frames = int(estimated_duration * out_fps * FRAME_BUFFER_MULTIPLIER) + FRAME_BUFFER_PADDING

        frames = np.empty((estimated_frames, out_height, out_width, 3), dtype=np.uint8)
        frames_read = 0

        with _ffmpeg.popen_decode(ffmpeg_cmd, bufsize=10**8) as process:
            while frames_read < estimated_frames:
                remaining_frames = estimated_frames - frames_read
                batch_size = min(read_batch_size, remaining_frames)

                batch_data = process.stdout.read(frame_size * batch_size)  # type: ignore[union-attr]
                if not batch_data:
                    break

                batch_frames = np.frombuffer(batch_data, dtype=np.uint8)
                complete_frames = len(batch_frames) // (out_height * out_width * 3)
                if complete_frames == 0:
                    break

                complete_data = batch_frames[: complete_frames * out_height * out_width * 3]
                batch_frames_array = complete_data.reshape(complete_frames, out_height, out_width, 3)

                if frames_read + complete_frames > estimated_frames:
                    new_size = max(estimated_frames * 2, frames_read + complete_frames + 100)
                    new_frames = np.empty((new_size, out_height, out_width, 3), dtype=np.uint8)
                    new_frames[:frames_read] = frames[:frames_read]
                    frames = new_frames
                    estimated_frames = new_size

                end_idx = frames_read + complete_frames
                frames[frames_read:end_idx] = batch_frames_array
                frames_read += complete_frames

        if process.returncode not in (0, None) and frames_read == 0:
            raise ValueError(f"FFmpeg failed to process video (return code: {process.returncode})")

        if frames_read == 0:
            raise ValueError("No frames were read from the video")

        frames = frames[:frames_read]  # type: ignore

        try:
            audio = Audio.from_path(path)
            if start_second is not None or end_second is not None:
                audio_start = start_second if start_second is not None else 0
                audio_end = end_second if end_second is not None else audio.metadata.duration_seconds
                audio = audio.slice(start_seconds=audio_start, end_seconds=audio_end)
        except (AudioLoadError, FileNotFoundError):
            warnings.warn(f"No audio found for `{path}`, adding silent track.")
            segment_duration = frames_read / out_fps
            audio = Audio.create_silent(duration_seconds=round(segment_duration, 2), stereo=True, sample_rate=44100)

        return frames, out_fps, audio

    except VideoMetadataError:
        raise
    except FFmpegRunError as e:
        raise VideoLoadError(f"FFmpeg failed: {e}") from e
    except (OSError, IOError) as e:
        raise VideoLoadError(f"I/O error: {e}")


def encode_video(
    frames: np.ndarray,
    fps: float,
    audio: Audio,
    frame_shape: tuple[int, int, int],
    *,
    filename: str | Path | None = None,
    format: ALLOWED_VIDEO_FORMATS = "mp4",
    preset: ALLOWED_VIDEO_PRESETS = "medium",
    crf: int = 23,
) -> Path:
    """Encode an RGB frame array + audio track to disk via ffmpeg.

    ``frame_shape`` is ``(height, width, channels)``; passed in so this
    helper stays a plain function rather than re-deriving shape from
    the array.

    Raises:
        ValueError: If ``format`` or ``preset`` is not in the allowed set.
        FFmpegRunError: If ffmpeg fails to encode.
    """
    if format.lower() not in get_args(ALLOWED_VIDEO_FORMATS):
        raise ValueError(
            f"Unsupported format: {format}. Allowed formats are: {', '.join(get_args(ALLOWED_VIDEO_FORMATS))}"
        )

    if preset not in get_args(ALLOWED_VIDEO_PRESETS):
        raise ValueError(
            f"Unsupported preset: {preset}. Allowed presets are: {', '.join(get_args(ALLOWED_VIDEO_PRESETS))}"
        )

    frame_height, frame_width = frame_shape[:2]
    require_even(frame_width, frame_height)

    if filename is None:
        filename = Path(f"{uuid.uuid4()}.{format}")
    else:
        filename = Path(filename).with_suffix(f".{format}")
        filename.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
        audio.save(temp_audio.name, format="wav")

        duration = len(frames) / fps

        ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{frame_width}x{frame_height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-i",
            temp_audio.name,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-t",
            str(duration),
            "-vsync",
            "cfr",
            str(filename),
        ]

        with _ffmpeg.popen_encode(ffmpeg_command) as process:
            if frames.dtype != np.uint8 or not frames.flags["C_CONTIGUOUS"]:
                frames = np.ascontiguousarray(frames, dtype=np.uint8)

            buffer = memoryview(frames)
            try:
                process.stdin.write(buffer)  # type: ignore[union-attr]
            except BrokenPipeError as e:
                stderr = process.stderr.read() if process.stderr is not None else b""
                raise FFmpegRunError(
                    f"ffmpeg terminated while receiving video data: {stderr.decode(errors='replace')}"
                ) from e

        return filename

from __future__ import annotations

import tempfile
import uuid
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Generator, Literal, get_args

import numpy as np

from videopython.base import _ffmpeg
from videopython.base._dimensions import require_even
from videopython.base.audio import Audio
from videopython.base.exceptions import (
    AudioLoadError,
    FFmpegProbeError,
    FFmpegRunError,
    VideoLoadError,
    VideoMetadataError,
)

__all__ = [
    "Video",
    "VideoMetadata",
    "FrameIterator",
    "extract_frames_at_indices",
    "extract_frames_at_times",
]

ALLOWED_VIDEO_FORMATS = Literal["mp4", "avi", "mov", "mkv", "webm"]
ALLOWED_VIDEO_PRESETS = Literal[
    "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"
]

# Frame buffer constants for video loading
# Used to pre-allocate frame array with safety margin for frame rate variations
FRAME_BUFFER_MULTIPLIER = 1.1  # 10% buffer for frame rate estimation errors
FRAME_BUFFER_PADDING = 10  # Additional fixed frame padding


@dataclass
class VideoMetadata:
    """Class to store video metadata."""

    height: int
    width: int
    fps: float
    frame_count: int
    total_seconds: float

    def __str__(self) -> str:
        return f"{self.width}x{self.height} @ {self.fps}fps, {self.total_seconds} seconds"

    def __repr__(self) -> str:
        return self.__str__()

    def get_frame_shape(self) -> np.ndarray:
        """Returns frame shape."""
        return np.array((self.height, self.width, 3))

    def get_video_shape(self) -> np.ndarray:
        """Returns video shape."""
        return np.array((self.frame_count, self.height, self.width, 3))

    @staticmethod
    def _run_ffprobe(video_path: str | Path) -> dict:
        """Run ffprobe and return parsed JSON output."""
        try:
            return _ffmpeg.probe(
                video_path,
                extra_args=[
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,r_frame_rate,nb_frames",
                    "-show_entries",
                    "format=duration",
                ],
            )
        except FFmpegProbeError as e:
            raise VideoMetadataError(str(e)) from e

    @classmethod
    def from_path(cls, video_path: str | Path) -> VideoMetadata:
        """Creates VideoMetadata object from video file using ffprobe."""
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        probe_data = cls._run_ffprobe(video_path)

        try:
            stream_info = probe_data["streams"][0]

            width = int(stream_info["width"])
            height = int(stream_info["height"])

            try:
                fps_fraction = Fraction(stream_info["r_frame_rate"])
                fps = float(fps_fraction)
            except (ValueError, ZeroDivisionError):
                raise VideoMetadataError(f"Invalid frame rate: {stream_info['r_frame_rate']}")

            if "nb_frames" in stream_info and stream_info["nb_frames"].isdigit():
                frame_count = int(stream_info["nb_frames"])
            else:
                duration = float(probe_data["format"]["duration"])
                frame_count = int(round(duration * fps))

            total_seconds = round(frame_count / fps, 4)

            return cls(height=height, width=width, fps=fps, frame_count=frame_count, total_seconds=total_seconds)

        except KeyError as e:
            raise VideoMetadataError(f"Missing required metadata field: {e}")
        except (TypeError, IndexError) as e:
            raise VideoMetadataError(f"Invalid metadata structure: {e}")

    @classmethod
    def from_video(cls, video: Video) -> VideoMetadata:
        """Creates VideoMetadata object from Video instance."""
        frame_count, height, width, _ = video.frames.shape
        total_seconds = round(frame_count / video.fps, 4)

        return cls(height=height, width=width, fps=video.fps, frame_count=frame_count, total_seconds=total_seconds)

    def with_duration(self, seconds: float) -> VideoMetadata:
        """Return new metadata with updated duration.

        Args:
            seconds: New duration in seconds.

        Returns:
            New VideoMetadata with updated duration and frame count.
        """
        return VideoMetadata(
            height=self.height,
            width=self.width,
            fps=self.fps,
            frame_count=round(self.fps * seconds),
            total_seconds=round(seconds, 4),
        )

    def with_dimensions(self, width: int, height: int) -> VideoMetadata:
        """Return new metadata with updated dimensions.

        Args:
            width: New width in pixels.
            height: New height in pixels.

        Returns:
            New VideoMetadata with updated dimensions.
        """
        return VideoMetadata(
            height=height,
            width=width,
            fps=self.fps,
            frame_count=self.frame_count,
            total_seconds=self.total_seconds,
        )

    def with_fps(self, fps: float) -> VideoMetadata:
        """Return new metadata with updated fps.

        Args:
            fps: New frames per second.

        Returns:
            New VideoMetadata with updated fps (duration stays same).
        """
        return VideoMetadata(
            height=self.height,
            width=self.width,
            fps=fps,
            frame_count=round(fps * self.total_seconds),
            total_seconds=self.total_seconds,
        )

    def can_be_downsampled_to(self, target_format: VideoMetadata) -> bool:
        """Checks if video can be downsampled to target_format."""
        return (
            self.height >= target_format.height
            and self.width >= target_format.width
            and round(self.fps) >= round(target_format.fps)
            and self.total_seconds >= target_format.total_seconds
        )


class FrameIterator:
    """Memory-efficient frame iterator using ffmpeg streaming.

    Yields frames one at a time, keeping memory usage constant regardless
    of video length. Supports context manager protocol for resource cleanup.

    This is useful for operations that only need to process frames sequentially,
    such as scene detection, without loading the entire video into memory.

    Example:
        >>> with FrameIterator("video.mp4") as frames:
        ...     for idx, frame in frames:
        ...         process(frame)
    """

    def __init__(
        self,
        path: str | Path,
        start_second: float | None = None,
        end_second: float | None = None,
        vf_filters: list[str] | None = None,
        output_fps: float | None = None,
        output_width: int | None = None,
        output_height: int | None = None,
    ):
        """Initialize the frame iterator.

        Args:
            path: Path to video file
            start_second: Optional start time in seconds (seek before reading)
            end_second: Optional end time in seconds (stop reading after this)
            vf_filters: Optional list of ffmpeg -vf filter expressions to apply
                during decode (e.g. ``["scale=1280:720", "fps=30"]``).
            output_fps: Override output fps (adds fps filter if not in vf_filters).
            output_width: Override output width for frame size calculation.
            output_height: Override output height for frame size calculation.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        self.metadata = VideoMetadata.from_path(path)
        self.start_second = start_second if start_second is not None else 0.0
        self.end_second = end_second
        self._iter: Generator[tuple[int, np.ndarray], None, None] | None = None

        # Build -vf filter chain
        self._vf_filters = list(vf_filters) if vf_filters else []
        if output_fps is not None and not any(f.startswith("fps=") for f in self._vf_filters):
            self._vf_filters.append(f"fps={output_fps}")

        # Output dimensions (after filters)
        self.output_width = output_width or self.metadata.width
        self.output_height = output_height or self.metadata.height
        self.output_fps = output_fps or self.metadata.fps
        self._frame_size = self.output_width * self.output_height * 3

    def _build_ffmpeg_command(self) -> list[str]:
        """Build ffmpeg command for frame streaming."""
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        if self.start_second > 0:
            cmd.extend(["-ss", str(self.start_second)])

        cmd.extend(["-i", str(self.path)])

        if self.end_second is not None:
            duration = self.end_second - self.start_second
            cmd.extend(["-t", str(duration)])

        if self._vf_filters:
            cmd.extend(["-vf", ",".join(self._vf_filters)])

        cmd.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-vcodec",
                "rawvideo",
                "-y",
                "pipe:1",
            ]
        )
        return cmd

    def __iter__(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame) tuples.

        Frame indices are absolute indices in the original video,
        accounting for any start_second offset.
        """
        self._iter = self._iter_frames()
        return self._iter

    def _iter_frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        cmd = self._build_ffmpeg_command()
        with _ffmpeg.popen_decode(cmd, bufsize=self._frame_size * 2) as proc:
            frame_idx = int(self.start_second * self.output_fps)
            while True:
                raw_frame = proc.stdout.read(self._frame_size)  # type: ignore[union-attr]
                if len(raw_frame) != self._frame_size:
                    break
                frame = (
                    np.frombuffer(raw_frame, dtype=np.uint8).copy().reshape(self.output_height, self.output_width, 3)
                )
                yield frame_idx, frame
                frame_idx += 1

    def __enter__(self) -> "FrameIterator":
        return self

    def __exit__(self, *args: object) -> None:
        if self._iter is not None:
            self._iter.close()
            self._iter = None


def extract_frames_at_indices(
    path: str | Path,
    frame_indices: list[int],
) -> np.ndarray:
    """Extract specific frames from video without loading all frames.

    Uses ffmpeg's select filter for extraction. For sparse frame selection
    (e.g., 1 frame every 100), this is much more memory-efficient than
    loading all frames.

    Args:
        path: Path to video file
        frame_indices: List of frame indices to extract (0-indexed)

    Returns:
        numpy array of shape (len(frame_indices), H, W, 3)

    Example:
        >>> frames = extract_frames_at_indices("video.mp4", [0, 100, 200])
        >>> frames.shape  # (3, 1080, 1920, 3)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if not frame_indices:
        metadata = VideoMetadata.from_path(path)
        return np.empty((0, metadata.height, metadata.width, 3), dtype=np.uint8)

    metadata = VideoMetadata.from_path(path)

    # Remove duplicates and sort for ffmpeg
    unique_sorted_indices = sorted(set(frame_indices))

    # Build select filter expression
    select_expr = "+".join([f"eq(n\\,{idx})" for idx in unique_sorted_indices])

    cmd = [
        "ffmpeg",
        "-i",
        str(path),
        "-vf",
        f"select='{select_expr}'",
        "-vsync",
        "vfr",  # Variable frame rate output
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-y",
        "pipe:1",
    ]

    frame_size = metadata.width * metadata.height * 3

    with _ffmpeg.popen_decode(cmd, bufsize=10**8) as process:
        raw_data, _ = process.communicate()

    actual_frames = len(raw_data) // frame_size
    if actual_frames == 0:
        return np.empty((0, metadata.height, metadata.width, 3), dtype=np.uint8)

    # Truncate to complete frames only
    raw_data = raw_data[: actual_frames * frame_size]

    frames = np.frombuffer(raw_data, dtype=np.uint8).copy()
    frames = frames.reshape(-1, metadata.height, metadata.width, 3)

    # Reorder to match original frame_indices order if needed
    if unique_sorted_indices != frame_indices:
        index_map = {idx: i for i, idx in enumerate(unique_sorted_indices)}
        reorder = [index_map[idx] for idx in frame_indices if idx in index_map]
        frames = frames[reorder]

    return frames


def extract_frames_at_times(
    path: str | Path,
    timestamps: list[float],
) -> np.ndarray:
    """Extract frames at specific timestamps.

    Args:
        path: Path to video file
        timestamps: List of timestamps in seconds

    Returns:
        numpy array of shape (len(timestamps), H, W, 3)

    Example:
        >>> frames = extract_frames_at_times("video.mp4", [0.0, 5.0, 10.0])
        >>> frames.shape  # (3, 1080, 1920, 3)
    """
    metadata = VideoMetadata.from_path(path)
    frame_indices = [int(t * metadata.fps) for t in timestamps]
    return extract_frames_at_indices(path, frame_indices)


class Video:
    def __init__(self, frames: np.ndarray, fps: int | float, audio: Audio | None = None):
        self.frames = frames
        self.fps = fps
        if audio:
            self.audio = audio
        else:
            self.audio = Audio.create_silent(
                duration_seconds=round(self.total_seconds, 2), stereo=True, sample_rate=44100
            )

    @classmethod
    def from_path(
        cls,
        path: str,
        read_batch_size: int = 100,
        start_second: float | None = None,
        end_second: float | None = None,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Video:
        try:
            # Get video metadata using VideoMetadata.from_path
            metadata = VideoMetadata.from_path(path)

            out_width = width if width is not None else metadata.width
            out_height = height if height is not None else metadata.height
            out_fps = fps if fps is not None else metadata.fps
            total_duration = metadata.total_seconds

            # Validate time bounds
            if start_second is not None and start_second < 0:
                raise ValueError("start_second must be non-negative")
            if end_second is not None and end_second > total_duration:
                raise ValueError(f"end_second ({end_second}) exceeds video duration ({total_duration})")
            if start_second is not None and end_second is not None and start_second >= end_second:
                raise ValueError("start_second must be less than end_second")

            # Estimate memory usage and warn for large videos
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

            # Build FFmpeg command with improved segment handling
            ffmpeg_cmd = ["ffmpeg"]

            # Add seek option BEFORE input for more efficient seeking
            if start_second is not None:
                ffmpeg_cmd.extend(["-ss", str(start_second)])

            ffmpeg_cmd.extend(["-i", path])

            # Add duration AFTER input for more precise timing
            if end_second is not None and start_second is not None:
                duration = end_second - start_second
                ffmpeg_cmd.extend(["-t", str(duration)])
            elif end_second is not None:
                ffmpeg_cmd.extend(["-t", str(end_second)])

            # Apply video filters for resize and fps resampling
            vf_filters: list[str] = []
            if width is not None or height is not None:
                vf_filters.append(f"scale={out_width}:{out_height}")
            if fps is not None and fps != metadata.fps:
                vf_filters.append(f"fps={out_fps}")
            if vf_filters:
                ffmpeg_cmd.extend(["-vf", ",".join(vf_filters)])

            # Output format settings - removed problematic -vsync 0
            ffmpeg_cmd.extend(
                [
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-vcodec",
                    "rawvideo",
                    "-avoid_negative_ts",
                    "make_zero",  # Handle timing issues
                    "-y",
                    "pipe:1",
                ]
            )

            # Calculate frame size in bytes
            frame_size = out_width * out_height * 3  # 3 bytes per pixel for RGB

            # Estimate frame count for pre-allocation
            if start_second is not None and end_second is not None:
                estimated_duration = end_second - start_second
            elif end_second is not None:
                estimated_duration = end_second
            elif start_second is not None:
                estimated_duration = total_duration - start_second
            else:
                estimated_duration = total_duration

            # Add buffer to handle frame rate variations and rounding
            estimated_frames = int(estimated_duration * out_fps * FRAME_BUFFER_MULTIPLIER) + FRAME_BUFFER_PADDING

            # Pre-allocate numpy array
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
                        # Pre-allocation undershoot — rare with the buffer.
                        new_size = max(estimated_frames * 2, frames_read + complete_frames + 100)
                        new_frames = np.empty((new_size, out_height, out_width, 3), dtype=np.uint8)
                        new_frames[:frames_read] = frames[:frames_read]
                        frames = new_frames
                        estimated_frames = new_size

                    end_idx = frames_read + complete_frames
                    frames[frames_read:end_idx] = batch_frames_array
                    frames_read += complete_frames

            # Check if FFmpeg had an error (non-zero return code)
            if process.returncode not in (0, None) and frames_read == 0:
                raise ValueError(f"FFmpeg failed to process video (return code: {process.returncode})")

            if frames_read == 0:
                raise ValueError("No frames were read from the video")

            # Trim the pre-allocated array to actual frames read
            frames = frames[:frames_read]  # type: ignore

            # Load audio for the specified segment
            try:
                audio = Audio.from_path(path)
                # Slice audio to match the video segment
                if start_second is not None or end_second is not None:
                    audio_start = start_second if start_second is not None else 0
                    audio_end = end_second if end_second is not None else audio.metadata.duration_seconds
                    audio = audio.slice(start_seconds=audio_start, end_seconds=audio_end)
            except (AudioLoadError, FileNotFoundError):
                warnings.warn(f"No audio found for `{path}`, adding silent track.")
                # Create silent audio based on actual frames read
                segment_duration = frames_read / out_fps
                audio = Audio.create_silent(duration_seconds=round(segment_duration, 2), stereo=True, sample_rate=44100)

            return cls(frames=frames, fps=out_fps, audio=audio)

        except VideoMetadataError:
            raise
        except FFmpegRunError as e:
            raise VideoLoadError(f"FFmpeg failed: {e}") from e
        except (OSError, IOError) as e:
            raise VideoLoadError(f"I/O error: {e}")

    @classmethod
    def from_frames(cls, frames: np.ndarray, fps: float) -> Video:
        if frames.ndim != 4:
            raise ValueError(f"Unsupported number of dimensions: {frames.shape}!")
        elif frames.shape[-1] == 4:
            frames = frames[:, :, :, :3]
        elif frames.shape[-1] != 3:
            raise ValueError(f"Unsupported number of dimensions: {frames.shape}!")
        return cls(frames=frames, fps=fps)

    @classmethod
    def from_image(cls, image: np.ndarray, fps: float = 24.0, length_seconds: float = 1.0) -> Video:
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        frames = np.repeat(image, round(length_seconds * fps), axis=0)
        return cls(frames=frames, fps=fps)

    def copy(self) -> Video:
        copied = Video.from_frames(self.frames.copy(), self.fps)
        copied.audio = self.audio  # Audio objects are immutable, no need to copy
        return copied

    def is_loaded(self) -> bool:
        return self.fps is not None and self.frames is not None and self.audio is not None

    def split(self, frame_index: int | None = None) -> tuple[Video, Video]:
        if frame_index:
            if not (0 <= frame_index <= len(self.frames)):
                raise ValueError(f"frame_idx must be between 0 and {len(self.frames)}, got {frame_index}")
        else:
            frame_index = len(self.frames) // 2

        split_videos = (
            self.from_frames(self.frames[:frame_index], self.fps),
            self.from_frames(self.frames[frame_index:], self.fps),
        )

        # Split audio at the corresponding time point
        split_time = frame_index / self.fps
        split_videos[0].audio = self.audio.slice(start_seconds=0, end_seconds=split_time)
        split_videos[1].audio = self.audio.slice(start_seconds=split_time)

        return split_videos

    def save(
        self,
        filename: str | Path | None = None,
        format: ALLOWED_VIDEO_FORMATS = "mp4",
        preset: ALLOWED_VIDEO_PRESETS = "medium",
        crf: int = 23,
    ) -> Path:
        """Save video to file.

        Args:
            filename: Output filename. If None, generates random name
            format: Output format (mp4, avi, mov, mkv, webm)
            preset: Encoding speed/compression tradeoff. Slower presets give smaller
                files at the same quality. Options from fastest to smallest:
                ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
            crf: Constant Rate Factor (0-51). Lower = better quality, larger file.
                Default 23 is visually lossless for most content. Range 18-28 recommended.

        Returns:
            Path to saved video file

        Raises:
            RuntimeError: If video is not loaded
            ValueError: If format or preset is not supported
        """
        if not self.is_loaded():
            raise RuntimeError("Video is not loaded, cannot save!")

        if format.lower() not in get_args(ALLOWED_VIDEO_FORMATS):
            raise ValueError(
                f"Unsupported format: {format}. Allowed formats are: {', '.join(get_args(ALLOWED_VIDEO_FORMATS))}"
            )

        if preset not in get_args(ALLOWED_VIDEO_PRESETS):
            raise ValueError(
                f"Unsupported preset: {preset}. Allowed presets are: {', '.join(get_args(ALLOWED_VIDEO_PRESETS))}"
            )

        frame_height, frame_width = self.frame_shape[:2]
        require_even(frame_width, frame_height)

        if filename is None:
            filename = Path(f"{uuid.uuid4()}.{format}")
        else:
            filename = Path(filename).with_suffix(f".{format}")
            filename.parent.mkdir(parents=True, exist_ok=True)

        # Save audio to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            self.audio.save(temp_audio.name, format="wav")

            # Calculate exact duration
            duration = len(self.frames) / self.fps

            # Construct FFmpeg command (stream raw video via stdin)
            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                # Raw video input settings
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{self.frame_shape[1]}x{self.frame_shape[0]}",
                "-framerate",
                str(self.fps),
                "-i",
                "pipe:0",
                # Audio input
                "-i",
                temp_audio.name,
                # Video encoding settings
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                # Audio settings
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                # Output settings
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",  # Enable fast start for web playback
                "-t",
                str(duration),
                "-vsync",
                "cfr",
                str(filename),
            ]

            with _ffmpeg.popen_encode(ffmpeg_command) as process:
                frames = self.frames
                if frames.dtype != np.uint8 or not frames.flags["C_CONTIGUOUS"]:
                    frames = np.ascontiguousarray(frames, dtype=np.uint8)

                buffer = memoryview(frames)
                try:
                    process.stdin.write(buffer)  # type: ignore[union-attr]
                except BrokenPipeError as e:
                    # ffmpeg has already died; surface its stderr for diagnostics.
                    stderr = process.stderr.read() if process.stderr is not None else b""
                    raise FFmpegRunError(
                        f"ffmpeg terminated while receiving video data: {stderr.decode(errors='replace')}"
                    ) from e

            return filename

    def add_audio(self, audio: Audio, overlay: bool = True) -> Video:
        """Add audio to video, returning a new Video instance.

        Args:
            audio: Audio to add
            overlay: If True, overlay on existing audio; if False, replace it

        Returns:
            New Video with the audio added
        """
        video_duration = self.total_seconds
        audio_duration = audio.metadata.duration_seconds

        if audio_duration > video_duration:
            audio = audio.slice(start_seconds=0, end_seconds=video_duration)
        elif audio_duration < video_duration:
            silence_duration = video_duration - audio_duration
            silence = Audio.create_silent(
                duration_seconds=silence_duration,
                stereo=audio.metadata.channels == 2,
                sample_rate=audio.metadata.sample_rate,
            )
            audio = audio.concat(silence)

        new_video = self.copy()
        if new_video.audio.is_silent:
            new_video.audio = audio
        elif overlay:
            new_video.audio = new_video.audio.overlay(audio, position=0.0)
        else:
            new_video.audio = audio
        return new_video

    def add_audio_from_file(self, path: str, overlay: bool = True) -> Video:
        """Add audio from file, returning a new Video instance.

        Args:
            path: Path to audio file
            overlay: If True, overlay on existing audio; if False, replace it

        Returns:
            New Video with the audio added

        Raises:
            AudioLoadError: If audio file cannot be loaded
            FileNotFoundError: If audio file does not exist
        """
        new_audio = Audio.from_path(path)
        return self.add_audio(new_audio, overlay)

    def __add__(self, other: Video) -> Video:
        if self.fps != other.fps:
            raise ValueError("FPS of videos do not match!")
        elif self.frame_shape != other.frame_shape:
            raise ValueError(f"Resolutions do not match: {self.frame_shape} vs {other.frame_shape}")
        new_video = self.from_frames(np.r_["0,2", self.frames, other.frames], fps=self.fps)
        new_video.audio = self.audio.concat(other.audio)
        return new_video

    def __str__(self) -> str:
        return str(self.metadata)

    def __getitem__(self, val: slice) -> Video:
        if not isinstance(val, slice):
            raise ValueError("Only slices are supported for video indexing!")

        # Sub-slice video frames
        sliced = self.from_frames(self.frames[val], fps=self.fps)

        # Handle slicing bounds for audio
        start = val.start if val.start else 0
        stop = val.stop if val.stop else len(self.frames)
        if start < 0:
            start = len(self.frames) + start
        if stop < 0:
            stop = len(self.frames) + stop

        # Slice audio to match video duration
        audio_start = start / self.fps
        audio_end = stop / self.fps
        sliced.audio = self.audio.slice(start_seconds=audio_start, end_seconds=audio_end)
        return sliced

    @property
    def video_shape(self) -> tuple[int, int, int, int]:
        return self.frames.shape

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return self.frames.shape[1:]

    @property
    def total_seconds(self) -> float:
        return round(self.frames.shape[0] / self.fps, 4)

    @property
    def metadata(self) -> VideoMetadata:
        return VideoMetadata.from_video(self)

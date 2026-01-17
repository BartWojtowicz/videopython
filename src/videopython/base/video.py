from __future__ import annotations

import json
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Literal, get_args

import numpy as np

from videopython.base.audio import Audio
from videopython.base.exceptions import AudioLoadError, VideoLoadError, VideoMetadataError
from videopython.base.utils import generate_random_name

if TYPE_CHECKING:
    from videopython.base.description import BoundingBox

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
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames",
            "-show_entries",
            "format=duration",
            "-print_format",
            "json",
            str(video_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise VideoMetadataError(f"FFprobe error: {e.stderr}")
        except json.JSONDecodeError as e:
            raise VideoMetadataError(f"Error parsing FFprobe output: {e}")

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

            total_seconds = round(frame_count / fps, 2)

            return cls(height=height, width=width, fps=fps, frame_count=frame_count, total_seconds=total_seconds)

        except KeyError as e:
            raise VideoMetadataError(f"Missing required metadata field: {e}")
        except (TypeError, IndexError) as e:
            raise VideoMetadataError(f"Invalid metadata structure: {e}")

    @classmethod
    def from_video(cls, video: Video) -> VideoMetadata:
        """Creates VideoMetadata object from Video instance."""
        frame_count, height, width, _ = video.frames.shape
        total_seconds = round(frame_count / video.fps, 2)

        return cls(height=height, width=width, fps=video.fps, frame_count=frame_count, total_seconds=total_seconds)

    def can_be_merged_with(self, other_format: VideoMetadata) -> bool:
        """Check if videos can be merged."""
        return (
            self.height == other_format.height
            and self.width == other_format.width
            and round(self.fps) == round(other_format.fps)
        )

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
            total_seconds=seconds,
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

    # Fluent API for operation validation
    # These methods mirror the Video fluent API but only transform metadata

    def cut(self, start: float, end: float) -> VideoMetadata:
        """Predict metadata after cutting by time range.

        Args:
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            New VideoMetadata with updated duration.
        """
        if end <= start:
            raise ValueError(f"End time ({end}) must be greater than start time ({start})")
        if start < 0:
            raise ValueError(f"Start time ({start}) cannot be negative")
        if end > self.total_seconds:
            raise ValueError(f"End time ({end}) exceeds video duration ({self.total_seconds})")
        return self.with_duration(end - start)

    def cut_frames(self, start: int, end: int) -> VideoMetadata:
        """Predict metadata after cutting by frame range.

        Args:
            start: Start frame index (inclusive).
            end: End frame index (exclusive).

        Returns:
            New VideoMetadata with updated duration.
        """
        if end <= start:
            raise ValueError(f"End frame ({end}) must be greater than start frame ({start})")
        if start < 0:
            raise ValueError(f"Start frame ({start}) cannot be negative")
        if end > self.frame_count:
            raise ValueError(f"End frame ({end}) exceeds frame count ({self.frame_count})")
        duration = (end - start) / self.fps
        return self.with_duration(duration)

    def resize(self, width: int | None = None, height: int | None = None) -> VideoMetadata:
        """Predict metadata after resizing.

        If only width or height is provided, the other dimension is calculated
        to preserve aspect ratio.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            New VideoMetadata with updated dimensions.
        """
        if width is None and height is None:
            raise ValueError("Must provide width or height")

        if width and height:
            return self.with_dimensions(width, height)
        elif width:
            ratio = width / self.width
            new_height = round(self.height * ratio)
            return self.with_dimensions(width, new_height)
        else:  # height only
            ratio = height / self.height  # type: ignore[operator]
            new_width = round(self.width * ratio)
            return self.with_dimensions(new_width, height)  # type: ignore[arg-type]

    def crop(self, width: int, height: int) -> VideoMetadata:
        """Predict metadata after cropping.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            New VideoMetadata with updated dimensions.
        """
        if width > self.width:
            raise ValueError(f"Crop width ({width}) exceeds video width ({self.width})")
        if height > self.height:
            raise ValueError(f"Crop height ({height}) exceeds video height ({self.height})")
        return self.with_dimensions(width, height)

    def resample_fps(self, fps: float) -> VideoMetadata:
        """Predict metadata after resampling frame rate.

        Args:
            fps: Target frames per second.

        Returns:
            New VideoMetadata with updated fps.
        """
        if fps <= 0:
            raise ValueError(f"FPS ({fps}) must be positive")
        return self.with_fps(fps)

    def transition_to(self, other: VideoMetadata, effect_time: float = 0.0) -> VideoMetadata:
        """Predict metadata after transition to another video.

        Args:
            other: Metadata of the video to transition to.
            effect_time: Duration of the transition effect in seconds.

        Returns:
            New VideoMetadata for the combined video.

        Raises:
            ValueError: If videos have incompatible dimensions or fps.
        """
        if not self.can_be_merged_with(other):
            raise ValueError(
                f"Cannot merge videos: {self.width}x{self.height}@{round(self.fps)}fps "
                f"vs {other.width}x{other.height}@{round(other.fps)}fps"
            )
        combined_duration = self.total_seconds + other.total_seconds - effect_time
        return self.with_duration(combined_duration)


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
    ):
        """Initialize the frame iterator.

        Args:
            path: Path to video file
            start_second: Optional start time in seconds (seek before reading)
            end_second: Optional end time in seconds (stop reading after this)
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        self.metadata = VideoMetadata.from_path(path)
        self.start_second = start_second if start_second is not None else 0.0
        self.end_second = end_second
        self._process: subprocess.Popen | None = None
        self._frame_size = self.metadata.width * self.metadata.height * 3

    def _build_ffmpeg_command(self) -> list[str]:
        """Build ffmpeg command for frame streaming."""
        cmd = ["ffmpeg"]

        if self.start_second > 0:
            cmd.extend(["-ss", str(self.start_second)])

        cmd.extend(["-i", str(self.path)])

        if self.end_second is not None:
            duration = self.end_second - self.start_second
            cmd.extend(["-t", str(duration)])

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
        cmd = self._build_ffmpeg_command()

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=self._frame_size * 2,
        )

        # Calculate starting frame index based on start_second
        start_frame = int(self.start_second * self.metadata.fps)
        frame_idx = start_frame

        try:
            while True:
                raw_frame = self._process.stdout.read(self._frame_size)  # type: ignore
                if len(raw_frame) != self._frame_size:
                    break

                frame = np.frombuffer(raw_frame, dtype=np.uint8).copy()
                frame = frame.reshape(self.metadata.height, self.metadata.width, 3)

                yield frame_idx, frame
                frame_idx += 1
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Clean up ffmpeg process."""
        if self._process is not None:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
            if self._process.stdout:
                self._process.stdout.close()
            self._process = None

    def __enter__(self) -> "FrameIterator":
        return self

    def __exit__(self, *args: object) -> None:
        self._cleanup()


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

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
    )

    frame_size = metadata.width * metadata.height * 3

    try:
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

    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


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
        cls, path: str, read_batch_size: int = 100, start_second: float | None = None, end_second: float | None = None
    ) -> Video:
        try:
            # Get video metadata using VideoMetadata.from_path
            metadata = VideoMetadata.from_path(path)

            width = metadata.width
            height = metadata.height
            fps = metadata.fps
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

            estimated_frames = int(segment_duration * fps)
            estimated_bytes = estimated_frames * height * width * 3
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

            # Start FFmpeg process with stderr redirected to avoid deadlock
            process = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # Redirect stderr to avoid deadlock
                bufsize=10**8,  # Use large buffer for efficient I/O
            )

            # Calculate frame size in bytes
            frame_size = width * height * 3  # 3 bytes per pixel for RGB

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
            estimated_frames = int(estimated_duration * fps * FRAME_BUFFER_MULTIPLIER) + FRAME_BUFFER_PADDING

            # Pre-allocate numpy array
            frames = np.empty((estimated_frames, height, width, 3), dtype=np.uint8)
            frames_read = 0

            try:
                while frames_read < estimated_frames:
                    # Calculate remaining frames to read
                    remaining_frames = estimated_frames - frames_read
                    batch_size = min(read_batch_size, remaining_frames)

                    # Read batch of data
                    batch_data = process.stdout.read(frame_size * batch_size)  # type: ignore

                    if not batch_data:
                        break

                    # Convert to numpy array
                    batch_frames = np.frombuffer(batch_data, dtype=np.uint8)

                    # Calculate how many complete frames we got
                    complete_frames = len(batch_frames) // (height * width * 3)

                    if complete_frames == 0:
                        break

                    # Only keep complete frames
                    complete_data = batch_frames[: complete_frames * height * width * 3]
                    batch_frames_array = complete_data.reshape(complete_frames, height, width, 3)

                    # Check if we have room in pre-allocated array
                    if frames_read + complete_frames > estimated_frames:
                        # Need to expand array - this should be rare with our buffer
                        new_size = max(estimated_frames * 2, frames_read + complete_frames + 100)
                        new_frames = np.empty((new_size, height, width, 3), dtype=np.uint8)
                        new_frames[:frames_read] = frames[:frames_read]
                        frames = new_frames
                        estimated_frames = new_size

                    # Store batch in pre-allocated array
                    end_idx = frames_read + complete_frames
                    frames[frames_read:end_idx] = batch_frames_array
                    frames_read += complete_frames

            finally:
                # Ensure process is properly terminated
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

                # Clean up pipes
                if process.stdout:
                    process.stdout.close()

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
            except (AudioLoadError, FileNotFoundError, subprocess.CalledProcessError):
                warnings.warn(f"No audio found for `{path}`, adding silent track.")
                # Create silent audio based on actual frames read
                segment_duration = frames_read / fps
                audio = Audio.create_silent(duration_seconds=round(segment_duration, 2), stereo=True, sample_rate=44100)

            return cls(frames=frames, fps=fps, audio=audio)

        except VideoMetadataError:
            raise
        except subprocess.CalledProcessError as e:
            raise VideoLoadError(f"FFmpeg failed: {e}")
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

        if filename is None:
            filename = Path(generate_random_name(suffix=f".{format}"))
        else:
            filename = Path(filename).with_suffix(f".{format}")
            filename.parent.mkdir(parents=True, exist_ok=True)

        # Create a temporary raw video file
        with tempfile.NamedTemporaryFile(suffix=".raw") as raw_video:
            # Convert frames to raw video data
            raw_data = self.frames.astype(np.uint8).tobytes()
            raw_video.write(raw_data)
            raw_video.flush()

            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                self.audio.save(temp_audio.name, format="wav")

                # Calculate exact duration
                duration = len(self.frames) / self.fps

                # Construct FFmpeg command
                ffmpeg_command = [
                    "ffmpeg",
                    "-y",
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
                    raw_video.name,
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

                try:
                    subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                    return filename
                except subprocess.CalledProcessError as e:
                    print(f"Error saving video: {e}")
                    print(f"FFmpeg stderr: {e.stderr}")
                    raise

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

    # Fluent API for video transformations
    # These methods mirror the VideoMetadata fluent API

    def cut(self, start: float, end: float) -> Video:
        """Cut video to a time range.

        Args:
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            New Video with the specified time range.
        """
        from videopython.base.transforms import CutSeconds

        return CutSeconds(start, end).apply(self)

    def cut_frames(self, start: int, end: int) -> Video:
        """Cut video to a frame range.

        Args:
            start: Start frame index (inclusive).
            end: End frame index (exclusive).

        Returns:
            New Video with the specified frame range.
        """
        from videopython.base.transforms import CutFrames

        return CutFrames(start, end).apply(self)

    def resize(self, width: int | None = None, height: int | None = None) -> Video:
        """Resize video.

        If only width or height is provided, the other dimension is calculated
        to preserve aspect ratio.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            New Video with the specified dimensions.
        """
        from videopython.base.transforms import Resize

        return Resize(width=width, height=height).apply(self)

    def crop(self, width: int, height: int) -> Video:
        """Crop video to specified dimensions (center crop).

        Args:
            width: Target width in pixels.
            height: Target height in pixels.

        Returns:
            New Video with the specified dimensions.
        """
        from videopython.base.transforms import Crop

        return Crop(width=width, height=height).apply(self)

    def resample_fps(self, fps: float) -> Video:
        """Resample video to a different frame rate.

        Args:
            fps: Target frames per second.

        Returns:
            New Video with the specified frame rate.
        """
        from videopython.base.transforms import ResampleFPS

        return ResampleFPS(fps=fps).apply(self)

    def transition_to(self, other: Video, transition: object) -> Video:
        """Combine with another video using a transition.

        Args:
            other: Video to transition to.
            transition: Transition to apply (e.g., FadeTransition, BlurTransition).

        Returns:
            New Video combining both videos with the transition effect.
        """
        from videopython.base.transitions import Transition

        if not isinstance(transition, Transition):
            raise TypeError(f"Expected Transition, got {type(transition).__name__}")
        return transition.apply((self, other))

    def ken_burns(
        self,
        start_region: "BoundingBox",
        end_region: "BoundingBox",
        easing: Literal["linear", "ease_in", "ease_out", "ease_in_out"] = "linear",
        start: float | None = None,
        stop: float | None = None,
    ) -> Video:
        """Apply Ken Burns pan-and-zoom effect.

        Creates cinematic movement by smoothly transitioning between two regions.

        Args:
            start_region: Starting crop region (BoundingBox with normalized 0-1 coordinates).
            end_region: Ending crop region (BoundingBox with normalized 0-1 coordinates).
            easing: Animation easing - "linear", "ease_in", "ease_out", or "ease_in_out".
            start: Optional start time in seconds for the effect.
            stop: Optional stop time in seconds for the effect.

        Returns:
            New Video with Ken Burns effect applied.
        """
        from videopython.base.effects import KenBurns

        return KenBurns(start_region=start_region, end_region=end_region, easing=easing).apply(
            self, start=start, stop=stop
        )

    def picture_in_picture(
        self,
        overlay: Video,
        position: tuple[float, float] = (0.7, 0.7),
        scale: float = 0.25,
        border_width: int = 0,
        border_color: tuple[int, int, int] = (255, 255, 255),
        corner_radius: int = 0,
        opacity: float = 1.0,
        audio_mode: Literal["main", "overlay", "mix"] = "main",
        audio_mix: tuple[float, float] = (1.0, 1.0),
    ) -> Video:
        """Overlay another video as picture-in-picture.

        Args:
            overlay: Video to overlay on this video.
            position: Normalized (x, y) center position, (0,0)=top-left, (1,1)=bottom-right.
            scale: Overlay size relative to main video width (0.25 = 25%).
            border_width: Border width in pixels (default 0).
            border_color: Border color as RGB tuple (default white).
            corner_radius: Rounded corner radius in pixels (default 0).
            opacity: Overlay transparency from 0 to 1 (default 1.0).
            audio_mode: Audio handling - "main" (default), "overlay", or "mix".
            audio_mix: Volume factors (main, overlay) for mix mode, default (1.0, 1.0).

        Returns:
            New Video with picture-in-picture overlay.
        """
        from videopython.base.transforms import PictureInPicture

        return PictureInPicture(
            overlay=overlay,
            position=position,
            scale=scale,
            border_width=border_width,
            border_color=border_color,
            corner_radius=corner_radius,
            opacity=opacity,
            audio_mode=audio_mode,
            audio_mix=audio_mix,
        ).apply(self)

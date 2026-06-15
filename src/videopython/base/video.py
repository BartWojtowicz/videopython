from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Generator

import numpy as np

from videopython.audio import Audio
from videopython.base import _ffmpeg, _video_io
from videopython.base._video_io import ALLOWED_VIDEO_FORMATS, ALLOWED_VIDEO_PRESETS
from videopython.base.exceptions import FFmpegProbeError, VideoMetadataError

__all__ = [
    "Video",
    "VideoMetadata",
    "FrameIterator",
    "extract_frames_at_indices",
    "extract_frames_at_times",
    "ALLOWED_VIDEO_FORMATS",
    "ALLOWED_VIDEO_PRESETS",
]

# Cache of probed VideoMetadata keyed by (resolved path, mtime_ns, size). Every
# plan traversal (repair -> check -> validate -> run) re-probes the same files
# per segment; ffprobe is a ~10-50ms subprocess while os.stat is microseconds,
# so we re-stat on every call and invalidate on mtime_ns/size change. Bounded
# LRU so a long-lived worker touching many files stays memory-stable.
_METADATA_CACHE: OrderedDict[tuple[str, int, int], VideoMetadata] = OrderedDict()
_METADATA_CACHE_LOCK = threading.Lock()
_METADATA_CACHE_MAXSIZE = 128


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
    def _run_ffprobe(video_path: str | Path) -> dict[str, Any]:
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
    def _probe_uncached(cls, video_path: str | Path) -> VideoMetadata:
        """Probe a video file with ffprobe and parse it into VideoMetadata, bypassing the cache."""
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
    def from_path(cls, video_path: str | Path) -> VideoMetadata:
        """Creates VideoMetadata object from video file using ffprobe.

        Results are cached per ``(resolved path, mtime_ns, size)`` so repeated
        probes of the same file in one process collapse to a single ffprobe
        call. A file modified in place is re-probed automatically (the stat key
        changes); call :meth:`clear_cache` to force a re-probe after an in-place
        overwrite that somehow preserved both mtime_ns and size.
        """
        try:
            stat_result = os.stat(video_path)
        except OSError:
            raise FileNotFoundError(f"Video file not found: {video_path}")

        key = (os.fspath(Path(video_path).resolve()), stat_result.st_mtime_ns, stat_result.st_size)

        with _METADATA_CACHE_LOCK:
            cached = _METADATA_CACHE.get(key)
            if cached is not None:
                _METADATA_CACHE.move_to_end(key)
                return cached

        # Probe outside the lock so concurrent probes of different files do not serialize on it.
        metadata = cls._probe_uncached(video_path)

        with _METADATA_CACHE_LOCK:
            _METADATA_CACHE[key] = metadata
            _METADATA_CACHE.move_to_end(key)
            while len(_METADATA_CACHE) > _METADATA_CACHE_MAXSIZE:
                _METADATA_CACHE.popitem(last=False)

        return metadata

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the probe cache. Mainly for tests and in-place file overwrites."""
        with _METADATA_CACHE_LOCK:
            _METADATA_CACHE.clear()

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

        # Input-side -t: trim the SOURCE segment before the filter chain. As
        # an output option it would instead cap the post-filter duration,
        # silently truncating duration-extending filters (slow-motion setpts,
        # freeze-frame loop).
        if self.end_second is not None:
            duration = self.end_second - self.start_second
            cmd.extend(["-t", str(duration)])

        cmd.extend(["-i", str(self.path)])

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
        frames, out_fps, audio = _video_io.decode_video(
            path,
            read_batch_size=read_batch_size,
            start_second=start_second,
            end_second=end_second,
            fps=fps,
            width=width,
            height=height,
        )
        return cls(frames=frames, fps=out_fps, audio=audio)

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

        return _video_io.encode_video(
            self.frames,
            self.fps,
            self.audio,
            filename=filename,
            format=format,
            preset=preset,
            crf=crf,
        )

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

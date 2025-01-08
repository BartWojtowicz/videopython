from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import cv2
import numpy as np
from soundpython import Audio

from videopython.utils.common import generate_random_name

ALLOWED_VIDEO_FORMATS = Literal["mp4", "avi", "mov", "mkv", "webm"]


@dataclass
class VideoMetadata:
    """Class to store video metadata."""

    height: int
    width: int
    fps: float
    frame_count: int
    total_seconds: float

    def __str__(self):
        return f"{self.width}x{self.height} @ {self.fps}fps, {self.total_seconds} seconds"

    def __repr__(self) -> str:
        return self.__str__()

    def get_frame_shape(self):
        """Returns frame shape."""
        return np.array((self.height, self.width, 3))

    def get_video_shape(self):
        """Returns video shape."""
        return np.array((self.frame_count, self.height, self.width, 3))

    @classmethod
    def from_path(cls, video_path: str) -> VideoMetadata:
        """Creates VideoMetadata object from video file."""
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(video.get(cv2.CAP_PROP_FPS), 2)
        height = round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = round(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_seconds = round(frame_count / fps, 2)

        return cls(
            height=height,
            width=width,
            fps=fps,
            frame_count=frame_count,
            total_seconds=total_seconds,
        )

    @classmethod
    def from_video(cls, video: Video) -> VideoMetadata:
        """Creates VideoMetadata object from Video instance."""
        frame_count, height, width, _ = video.frames.shape
        total_seconds = round(frame_count / video.fps, 2)

        return cls(
            height=height,
            width=width,
            fps=video.fps,
            frame_count=frame_count,
            total_seconds=total_seconds,
        )

    def can_be_merged_with(self, other_format: VideoMetadata) -> bool:
        return (
            self.height == other_format.height
            and self.width == other_format.width
            and round(self.fps) == round(other_format.fps)
        )

    def can_be_downsampled_to(self, target_format: VideoMetadata) -> bool:
        """Checks if video can be downsampled to `target_format`.

        Args:
            target_format: Desired video format.

        Returns:
            True if video can be downsampled to `target_format`, False otherwise.
        """
        return (
            self.height >= target_format.height
            and self.width >= target_format.width
            and round(self.fps) >= round(target_format.fps)
            and self.total_seconds >= target_format.total_seconds
        )


class Video:
    def __init__(self):
        self.fps = None
        self.frames = None
        self.audio = None

    @classmethod
    def from_path(cls, path: str) -> Video:
        new_vid = cls()
        new_vid.frames, new_vid.fps = cls._load_video_from_path(path)

        try:
            new_vid.audio = Audio.from_file(path)
        except Exception:
            print(f"No audio found for `{path}`, adding silent track!")
            new_vid.audio = Audio.create_silent(
                duration_seconds=round(new_vid.total_seconds, 2), stereo=True, sample_rate=44100
            )
        return new_vid

    @classmethod
    def from_frames(cls, frames: np.ndarray, fps: float) -> Video:
        new_vid = cls()
        if frames.ndim != 4:
            raise ValueError(f"Unsupported number of dimensions: {frames.shape}!")
        elif frames.shape[-1] == 4:
            frames = frames[:, :, :, :3]
        elif frames.shape[-1] != 3:
            raise ValueError(f"Unsupported number of dimensions: {frames.shape}!")
        new_vid.frames = frames
        new_vid.fps = fps
        new_vid.audio = Audio.create_silent(
            duration_seconds=round(new_vid.total_seconds, 2), stereo=True, sample_rate=44100
        )
        return new_vid

    @classmethod
    def from_image(cls, image: np.ndarray, fps: float = 24.0, length_seconds: float = 1.0) -> Video:
        new_vid = cls()
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        new_vid.frames = np.repeat(image, round(length_seconds * fps), axis=0)
        new_vid.fps = fps
        new_vid.audio = Audio.create_silent(duration_seconds=length_seconds, stereo=True, sample_rate=44100)
        return new_vid

    def copy(self) -> Video:
        copied = Video().from_frames(self.frames.copy(), self.fps)
        copied.audio = self.audio  # Audio objects are immutable, no need to copy
        return copied

    def is_loaded(self) -> bool:
        return self.fps is not None and self.frames is not None and self.audio is not None

    def split(self, frame_idx: int | None = None) -> tuple[Video, Video]:
        if frame_idx:
            assert 0 <= frame_idx <= len(self.frames)
        else:
            frame_idx = len(self.frames) // 2

        split_videos = (
            self.from_frames(self.frames[:frame_idx], self.fps),
            self.from_frames(self.frames[frame_idx:], self.fps),
        )

        # Split audio at the corresponding time point
        split_time = frame_idx / self.fps
        split_videos[0].audio = self.audio.slice(start_seconds=0, end_seconds=split_time)
        split_videos[1].audio = self.audio.slice(start_seconds=split_time)

        return split_videos

    def save(self, filename: str | Path | None = None, format: ALLOWED_VIDEO_FORMATS = "mp4") -> Path:
        """Save video to file with optimized performance.
        
        Args:
            filename: Output filename. If None, generates random name
            format: Output format (mp4, avi, mov, mkv, webm)
            
        Returns:
            Path to saved video file
            
        Raises:
            RuntimeError: If video is not loaded
            ValueError: If format is not supported
        """
        if not self.is_loaded():
            raise RuntimeError("Video is not loaded, cannot save!")

        if format.lower() not in get_args(ALLOWED_VIDEO_FORMATS):
            raise ValueError(
                f"Unsupported format: {format}. Allowed formats are: {', '.join(get_args(ALLOWED_VIDEO_FORMATS))}"
            )

        if filename is None:
            filename = Path(generate_random_name(suffix=f".{format}"))
        else:
            filename = Path(filename).with_suffix(f".{format}")
            filename.parent.mkdir(parents=True, exist_ok=True)

        # Create a temporary raw video file
        with tempfile.NamedTemporaryFile(suffix='.raw') as raw_video:
            # Convert frames to raw video data
            raw_data = self.frames.astype(np.uint8).tobytes()
            raw_video.write(raw_data)
            raw_video.flush()

            # Save audio to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav') as temp_audio:
                self.audio.save(temp_audio.name, format="wav")
                
                # Calculate exact duration
                duration = len(self.frames) / self.fps

                # Construct FFmpeg command for maximum performance
                ffmpeg_command = [
                    "ffmpeg",
                    "-y",
                    # Raw video input settings
                    "-f", "rawvideo",
                    "-pixel_format", "rgb24",
                    "-video_size", f"{self.frame_shape[1]}x{self.frame_shape[0]}",
                    "-framerate", str(self.fps),
                    "-i", raw_video.name,
                    # Audio input
                    "-i", temp_audio.name,
                    # Video encoding settings
                    "-c:v", "libx264",
                    "-preset", "ultrafast",  # Fastest encoding
                    "-tune", "zerolatency",  # Reduce encoding latency
                    "-crf", "23",           # Reasonable quality/size tradeoff
                    # Audio settings  
                    "-c:a", "aac",
                    "-b:a", "192k",
                    # Output settings
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",  # Enable fast start for web playback
                    "-t", str(duration),
                    "-vsync", "cfr",
                    str(filename)
                ]

                try:
                    result = subprocess.run(
                        ffmpeg_command,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    return filename
                except subprocess.CalledProcessError as e:
                    print(f"Error saving video: {e}")
                    print(f"FFmpeg stderr: {e.stderr}")
                    raise

    def add_audio(self, audio: Audio, overlay: bool = True) -> None:
        if self.audio.is_silent:
            self.audio = audio
        elif overlay:
            self.audio = self.audio.overlay(audio, position=0.0)
        else:
            self.audio = audio

    def add_audio_from_file(self, path: str, overlay: bool = True) -> None:
        try:
            new_audio = Audio.from_file(path)
            self.add_audio(new_audio, overlay)
        except Exception:
            print(f"Audio file `{path}` not found or invalid, skipping!")

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

    @staticmethod
    def _load_video_from_path(path: str) -> tuple[np.ndarray, float]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"No frames could be read from the video file: {path}")

        return np.array(frames), fps

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

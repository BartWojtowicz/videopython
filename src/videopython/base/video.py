from __future__ import annotations

import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import cv2
import numpy as np
from pydub import AudioSegment

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
        """Creates VideoMetadata object from video file.

        Args:
            video_path: Path to video file.
        """
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
        """Creates VideoMetadata object from frames.

        Args:
            frames: Frames of the video.
            fps: Frames per second of the video.
        """

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
        audio = cls._load_audio_from_path(path)
        if not audio:
            print(f"No audio found for `{path}`, adding silent track!")
            audio = AudioSegment.silent(duration=round(new_vid.total_seconds * 1000))
        new_vid.audio = audio
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
        new_vid.audio = AudioSegment.silent(duration=round(new_vid.total_seconds * 1000))
        return new_vid

    @classmethod
    def from_image(cls, image: np.ndarray, fps: float = 24.0, length_seconds: float = 1.0) -> Video:
        new_vid = cls()
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        new_vid.frames = np.repeat(image, round(length_seconds * fps), axis=0)
        new_vid.fps = fps
        new_vid.audio = AudioSegment.silent(duration=round(new_vid.total_seconds * 1000))
        return new_vid

    def copy(self) -> Video:
        copied = Video().from_frames(self.frames.copy(), self.fps)
        copied.audio = self.audio
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
        audio_midpoint = (frame_idx / self.fps) * 1000
        split_videos[0].audio = self.audio[:audio_midpoint]
        split_videos[1].audio = self.audio[audio_midpoint:]
        return split_videos

    def save(self, filename: str | Path | None = None, format: ALLOWED_VIDEO_FORMATS = "mp4") -> Path:
        """Saves the video with audio.

        Args:
            filename: Name of the output video file. Generates random name if not provided.
            format: Output format (default is 'mp4').

        Returns:
            Path to the saved video file.
        """
        if not self.is_loaded():
            raise RuntimeError("Video is not loaded, cannot save!")

        # Check if the format is allowed
        if format.lower() not in get_args(ALLOWED_VIDEO_FORMATS):
            raise ValueError(
                f"Unsupported format: {format}. Allowed formats are: {', '.join(get_args(ALLOWED_VIDEO_FORMATS))}"
            )

        if filename is None:
            filename = Path(generate_random_name(suffix=f".{format}"))
        else:
            filename = Path(filename).with_suffix(f".{format}")
            filename.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Save frames as images
            for i, frame in enumerate(self.frames):
                frame_path = temp_dir_path / f"frame_{i:04d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Save audio to a temporary file
            temp_audio = temp_dir_path / "temp_audio.wav"
            self.audio.export(str(temp_audio), format="adts", bitrate="192k")

            # Construct FFmpeg command
            ffmpeg_command = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-r",
                str(self.fps),  # Set the frame rate
                "-i",
                str(temp_dir_path / "frame_%04d.png"),  # Input image sequence
                "-i",
                str(temp_audio),  # Input audio file
                "-c:v",
                "libx264",  # Video codec
                "-preset",
                "medium",  # Encoding preset (tradeoff between encoding speed and compression)
                "-crf",
                "23",  # Constant Rate Factor (lower means better quality, 23 is default)
                "-c:a",
                "copy",  # Audio codec
                "-b:a",
                "192k",  # Audio bitrate
                "-pix_fmt",
                "yuv420p",  # Pixel format
                "-shortest",  # Finish encoding when the shortest input stream ends
                str(filename),
            ]

            try:
                subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
                print(f"Video saved successfully to: {filename}")
                return filename
            except subprocess.CalledProcessError as e:
                print(f"Error saving video: {e}")
                print(f"FFmpeg stderr: {e.stderr}")
                raise

    def add_audio(self, audio: AudioSegment, overlay: bool = True, overlay_gain: int = 0, loop: bool = False) -> None:
        self.audio = self._process_audio(audio=audio, overlay=overlay, overlay_gain=overlay_gain, loop=loop)

    def add_audio_from_file(self, path: str, overlay: bool = True, overlay_gain: int = 0, loop: bool = False) -> None:
        new_audio = self._load_audio_from_path(path)
        if new_audio is None:
            print(f"Audio file `{path}` not found, skipping!")
            return

        self.audio = self._process_audio(audio=new_audio, overlay=overlay, overlay_gain=overlay_gain, loop=loop)

    def _process_audio(
        self, audio: AudioSegment, overlay: bool = True, overlay_gain: int = 0, loop: bool = False
    ) -> AudioSegment:
        if (duration_diff := round(self.total_seconds - audio.duration_seconds)) > 0 and not loop:
            audio = audio + AudioSegment.silent(duration_diff * 1000)
        elif audio.duration_seconds > self.total_seconds:
            audio = audio[: round(self.total_seconds * 1000)]

        if overlay:
            return self.audio.overlay(audio, loop=loop, gain_during_overlay=overlay_gain)
        return audio

    def __add__(self, other: Video) -> Video:
        # TODO: Should it be class method? How to make it work with sum()?
        if self.fps != other.fps:
            raise ValueError("FPS of videos do not match!")
        elif self.frame_shape != other.frame_shape:
            raise ValueError(
                "Resolutions of the images do not match: "
                f"{self.frame_shape} not compatible with {other.frame_shape}."
            )
        new_video = self.from_frames(np.r_["0,2", self.frames, other.frames], fps=self.fps)
        new_video.audio = self.audio + other.audio
        return new_video

    def __str__(self) -> str:
        return str(self.metadata)

    def __getitem__(self, val: slice) -> Video:
        if not isinstance(val, slice):
            raise ValueError("Only slices are supported for video indexing!")

        # Sub-slice video if given a slice
        sliced = self.from_frames(self.frames[val], fps=self.fps)
        # Handle slicing without value for audio
        start = val.start if val.start else 0
        stop = val.stop if val.stop else len(self.frames)
        # Handle negative values for audio slices
        if start < 0:
            start = len(self.frames) + start
        if stop < 0:
            stop = len(self.frames) + stop
        # Append audio to the slice
        audio_start = round(start / self.fps) * 1000
        audio_end = round(stop / self.fps) * 1000
        sliced.audio = self.audio[audio_start:audio_end]
        return sliced

    @staticmethod
    def _load_audio_from_path(path: str) -> AudioSegment | None:
        try:
            audio = AudioSegment.from_file(path)
            return audio
        except IndexError:
            return None

    @staticmethod
    def _load_video_from_path(path: str) -> tuple[np.ndarray, float]:
        """Loads frames and fps information from video file.

        Args:
            path: Path to video file.
        """
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
        """Returns 4D video shape."""
        return self.frames.shape

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Returns 3D frame shape."""
        return self.frames.shape[1:]

    @property
    def total_seconds(self) -> float:
        """Returns total seconds of the video."""
        return round(self.frames.shape[0] / self.fps, 4)

    @property
    def metadata(self) -> VideoMetadata:
        """Returns VideoMetadata object."""
        return VideoMetadata.from_video(self)

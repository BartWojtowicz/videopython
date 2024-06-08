from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pydub import AudioSegment

from videopython.utils.common import check_path, generate_random_name


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

    def save(self, filename: str | None = None) -> str:
        """Saves the video.

        Args:
            filename: Name of the output video file. Generates random UUID name if not provided.
        """
        if not self.is_loaded():
            raise RuntimeError(f"Video is not loaded, cannot save!")

        if filename is None:
            filename = generate_random_name(suffix=".mp4")
        filename = check_path(filename, dir_exists=True, suffix=".mp4")

        ffmpeg_video_command = (
            f"ffmpeg -loglevel error -y -framerate {self.fps} -f rawvideo -pix_fmt rgb24"
            f" -s {self.metadata.width}x{self.metadata.height} "
            f"-i pipe:0 -c:v libx264 -pix_fmt yuv420p {filename}"
        )

        ffmpeg_audio_command = (
            f"ffmpeg -loglevel error -y -i {filename} -f s16le -acodec pcm_s16le "
            f"-ar {self.audio.frame_rate} -ac {self.audio.channels} -i pipe:0 "
            f"-c:v copy -c:a aac -strict experimental {filename}_temp.mp4"
        )

        try:
            print("Saving frames to video...")
            subprocess.run(
                ffmpeg_video_command,
                input=self.frames.tobytes(),
                check=True,
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            print("Error saving frames to video!")
            raise e

        try:
            print("Adding audio track...")
            subprocess.run(ffmpeg_audio_command, input=self.audio.raw_data, check=True, shell=True)
            Path(filename).unlink()
            Path(filename + "_temp.mp4").rename(filename)
        except subprocess.CalledProcessError as e:
            print(f"Error adding audio track!")
            raise e

        print(f"Video saved into `{filename}`!")
        return filename

    def add_audio_from_file(self, path: str, overlay: bool = True, overlay_gain: int = 0, loop: bool = False) -> None:
        new_audio = self._load_audio_from_path(path)
        if new_audio is None:
            print(f"Audio file `{path}` not found, skipping!")
            return

        if (duration_diff := round(self.total_seconds - new_audio.duration_seconds)) > 0 and not loop:
            new_audio = new_audio + AudioSegment.silent(duration_diff * 1000)
        elif new_audio.duration_seconds > self.total_seconds:
            new_audio = new_audio[: round(self.total_seconds * 1000)]

        if overlay:
            self.audio = self.audio.overlay(new_audio, loop=loop, gain_during_overlay=overlay_gain)
        else:
            self.audio = new_audio

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
        metadata = VideoMetadata.from_path(path)
        ffmpeg_command = f"ffmpeg -i {path} -f rawvideo -pix_fmt rgb24 -loglevel quiet pipe:1"

        # Run the ffmpeg command and capture the stdout
        ffmpeg_process = subprocess.Popen(shlex.split(ffmpeg_command), stdout=subprocess.PIPE)
        ffmpeg_out, _ = ffmpeg_process.communicate()

        # Convert the raw video data to a NumPy array
        frames = np.frombuffer(ffmpeg_out, dtype=np.uint8).reshape([-1, metadata.height, metadata.width, 3])
        fps = metadata.fps
        return frames, fps

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

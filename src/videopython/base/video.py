from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from pydub import AudioSegment

from videopython.utils.common import generate_random_name


@dataclass
class VideoMetadata:
    """Class to store video metadata."""

    height: int
    width: int
    fps: int
    frame_count: int
    total_seconds: float
    with_audio: bool = False

    def __str__(self):
        return f"{self.height}x{self.width} @ {self.fps}fps, {self.total_seconds} seconds"

    def __repr__(self) -> str:
        return self.__str__()

    def get_frame_shape(self):
        """Returns frame shape."""
        return np.array((self.height, self.width, 3))

    def get_video_shape(self):
        """Returns video shape."""
        return np.array((self.frame_count, self.height, self.width, 3))

    @classmethod
    def from_path(cls, video_path: str):
        """Creates VideoMetadata object from video file.

        Args:
            video_path: Path to video file.
        """
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(video.get(cv2.CAP_PROP_FPS))
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
    def from_video(cls, video: Video):
        """Creates VideoMetadata object from frames.

        Args:
            frames: Frames of the video.
            fps: Frames per second of the video.
        """

        frame_count, height, width, _ = video.frames.shape
        total_seconds = round(frame_count / video.fps, 2)

        with_audio = bool(video.audio)

        return cls(
            height=height,
            width=width,
            fps=video.fps,
            frame_count=frame_count,
            total_seconds=total_seconds,
            with_audio=with_audio,
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
    def from_path(cls, path):
        new_vid = cls()
        new_vid.frames, new_vid.fps = cls._load_video_from_path(path)
        return new_vid

    @classmethod
    def from_frames(cls, frames, fps):
        new_vid = cls()
        new_vid.frames = frames
        new_vid.fps = fps
        return new_vid

    @classmethod
    def from_image(cls, image: np.ndarray, fps: int = 24, length_seconds: float = 1.0):
        new_vid = cls()
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        new_vid.frames = np.repeat(image, round(length_seconds * fps), axis=0)
        new_vid.fps = fps
        return new_vid

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        num_steps: int = 25,
        height: int = 320,
        width: int = 576,
        num_frames: int = 24,
        gpu_optimized: bool = False,
    ):
        torch_dtype = torch.float16 if gpu_optimized else torch.float32
        # TODO: Make it model independent
        pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch_dtype)
        if gpu_optimized:
            pipe.enable_model_cpu_offload()
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        video_frames = np.asarray(
            pipe(
                prompt,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                num_frames=num_frames,
            ).frames
        )
        return Video.from_frames(video_frames, fps=24)

    def add_audio_from_file(self, audio_path: str):
        self.audio = AudioSegment.from_file(audio_path)

    def __getitem__(self, val):
        if isinstance(val, slice):
            return self.from_frames(self.frames[val], fps=self.fps)
        elif isinstance(val, int):
            return self.frames[val]

    def copy(self):
        return Video().from_frames(self.frames.copy(), self.fps)

    def is_loaded(self) -> bool:
        return self.fps and self.frames

    def split(self, frame_idx: int | None = None):
        if frame_idx:
            assert 0 <= frame_idx <= len(self.frames)
        else:
            frame_idx = len(self.frames) // 2

        return (
            self.from_frames(self.frames[:frame_idx], self.fps),
            self.from_frames(self.frames[frame_idx:], self.fps),
        )

    def _prepare_new_canvas(self, output_path: str):
        """Prepares a new `self._transformed_video` canvas for cut video."""
        canvas = cv2.VideoWriter(
            filename=output_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=self.fps,
            frameSize=(self.video_shape[2], self.video_shape[1]),
        )
        return canvas

    def save(self, filename: str = None) -> str:
        """Transforms the video and saves as `filename`.

        Args:
            filename: Name of the output video file.
        """
        # Check correctness
        if not filename:
            filename = Path(generate_random_name()).resolve()
            directory = filename.parent
        elif not Path(filename).suffix == ".mp4":
            raise ValueError("Only .mp4 save option is supported.")
        else:
            filename = Path(filename)
            directory = filename.parent
            if not directory.exists():
                raise ValueError(f"Selected directory `{directory}` does not exist!")

        filename, directory = str(filename), str(directory)
        # Save video video opencv
        canvas = self._prepare_new_canvas(filename)
        for frame in self.frames[:, :, :, ::-1]:
            canvas.write(frame)
        cv2.destroyAllWindows()
        canvas.release()
        # If Video has audio, overlaay audio using ffmpeg
        if self.audio:
            filename_with_audio = tempfile.NamedTemporaryFile(suffix=".mp4").name

            if len(self.audio) > self.total_seconds * 1000:
                self.audio = self.audio[: self.total_seconds * 1000]
            else:
                self.audio += AudioSegment.silent(duration=self.total_seconds * 1000 - len(self.audio))

            raw_audio = self.audio.raw_data
            channels = self.audio.channels
            frame_rate = self.audio.frame_rate

            ffmpeg_command = (
                f"ffmpeg -loglevel error -y -i {filename} -f s16le -acodec pcm_s16le -ar {frame_rate} -ac "
                f"{channels} -i pipe:0 -c:v copy -c:a aac -strict experimental {filename_with_audio}"
            )

            try:
                subprocess.run(ffmpeg_command, input=raw_audio, check=True, shell=True)
                print("Video with audio saved successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error saving video with audio: {e}")

            Path(filename).unlink()
            Path(filename_with_audio).rename(filename)

        return filename

    def __add__(self, other):
        # TODO: Should it be class method? How to make it work with sum()?
        if self.fps != other.fps:
            raise ValueError("FPS of videos do not match!")
        elif self.frame_shape != other.frame_shape:
            raise ValueError(
                "Resolutions of the images do not match: "
                f"{self.frame_shape} not compatible with {other.frame_shape}."
            )

        return self.from_frames(np.r_["0,2", self.frames, other.frames], fps=self.fps)

    @staticmethod
    def _load_video_from_path(path: str):
        """Loads frames and fps information from video file.

        Args:
            path: Path to video file.
        """
        metadata = VideoMetadata.from_path(path)
        ffmpeg_command = [
            "ffmpeg",
            "-i",
            path,
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-loglevel",
            "quiet",
            "pipe:1",
        ]

        # Run the ffmpeg command and capture the stdout
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE)
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
        return round(self.frames.shape[0] / self.fps, 1)

    @property
    def metadata(self) -> VideoMetadata:
        """Returns VideoMetadata object."""
        return VideoMetadata.from_video(self)

from dataclasses import dataclass
from pathlib import Path

import ffmpeg
import cv2
import numpy as np

from videopython.utils.common import generate_random_video_name


@dataclass
class VideoMetadata:
    """Class to store video metadata."""
    height: int
    width: int
    fps: int
    frame_count: int
    total_seconds: float

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
    def from_video(cls, video_path: str):
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
    def from_frames(cls, frames: np.ndarray, fps: int):
        """Creates VideoMetadata object from frames.

        Args:
            frames: Frames of the video.
            fps: Frames per second of the video.
        """
        frame_count, height, width, _ = frames.shape
        total_seconds = round(frame_count / fps, 2)

        return cls(
            height=height,
            width=width,
            fps=fps,
            frame_count=frame_count,
            total_seconds=total_seconds,
        )

    def can_be_merged_with(self, other_format: "VideoMetadata") -> bool:
        return (self.height == other_format.height and
                self.width == other_format.width and
                round(self.fps) == round(other_format.fps))

    def can_be_downsampled_to(self, target_format: "VideoMetadata") -> bool:
        """Checks if video can be downsampled to `target_format`.

        Args:
            target_format: Desired video format.

        Returns:
            True if video can be downsampled to `target_format`, False otherwise.
        """
        return (self.height >= target_format.height and
                self.width >= target_format.width and
                round(self.fps) >= round(target_format.fps) and
                self.total_seconds >= target_format.total_seconds)


class Video:

    def __init__(self):
        self.fps = None
        self.frames = None

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

    def is_loaded(self) -> bool:
        return self.fps and self.frames

    def split(self, frame_idx: int | None = None):
        if frame_idx:
            assert 0 <= frame_idx <= len(self.frames)
        else:
            frame_idx = len(self.frames) // 2

        return (self.from_frames(self.frames[:frame_idx], self.fps),
                self.from_frames(self.frames[frame_idx:], self.fps))

    def _prepare_new_canvas(self, output_path: str):
        """Prepares a new `self._transformed_video` canvas for cut video."""
        canvas = cv2.VideoWriter(
            filename=output_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=self.fps,
            frameSize=(self.video_shape[2], self.video_shape[1]),
        )
        return canvas

    def save(self, output_dir: Path | str = None):
        """Transforms the video and saves into `output_dir`.

        Args:
            output_path: Output directory for transformed video.
        """
        vid_name = generate_random_video_name()
        output_path = str(Path(output_dir) / vid_name)
        canvas = self._prepare_new_canvas(output_path)
        for frame in self.frames[:, :, :, ::-1]:
            canvas.write(frame)
        cv2.destroyAllWindows()
        canvas.release()
        return output_path

    def __add__(self, other):
        if self.fps != other.fps:
            raise ValueError("FPS of videos do not match!")
        elif self.frame_shape != other.frame_shape:
            raise ValueError(
                "Resolutions of the images do not match: "
                f"{self.frame_shape} not compatible with {other.frame_shape}.")

        self.frames = np.concatenate([self.frames, other.frames],
                                     axis=0).astype(np.uint8)
        return self

    @staticmethod
    def _load_video_from_path(path: str):
        """Loads frames and fps information from video file.
        
        Args:
            path: Path to video file.
        """
        metadata = VideoMetadata.from_video(path)
        ffmpeg_out, _ = (ffmpeg.input(path).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24",
            loglevel="quiet").run(capture_stdout=True))

        frames = np.frombuffer(ffmpeg_out, np.uint8).reshape(
            [-1, metadata.height, metadata.width, 3])
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
        return VideoMetadata.from_frames(self.frames, self.fps)

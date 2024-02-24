from abc import ABC, abstractmethod
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

from videopython.base.video import Video


class Transformation(ABC):
    """Abstract class for transformation on frames of video."""

    @abstractmethod
    def apply(self, video: Video) -> Video:
        pass


class TransformationPipeline:
    def __init__(self, transformations: list[Transformation] | None):
        """Initializes pipeline."""
        self.transformations = transformations if transformations else []

    def add(self, transformation: Transformation):
        """Adds transformation to the pipeline.

        Args:
            transformation: Transformation to add.

        Returns:
            Pipeline with added transformation.
        """
        self.transformations.append(transformation)
        return self

    def run(self, video: Video) -> Video:
        """Applies pipeline to the video.

        Args:
            video: Video to transform.

        Returns:
            Transformed video.
        """
        for transformation in self.transformations:
            video = transformation.apply(video)
        return video

    def __call__(self, video: Video) -> Video:
        return self.run(video)


class CutFrames(Transformation):
    def __init__(self, start_frame: int, end_frame: int):
        self.start_frame = start_frame
        self.end_frame = end_frame

    def apply(self, video: Video) -> Video:
        video = video[self.start_frame : self.end_frame]
        return video


class CutSeconds(Transformation):
    def __init__(self, start_second: float | int, end_second: float | int):
        self.start_second = start_second
        self.end_second = end_second

    def apply(self, video: Video) -> Video:
        video = video[round(self.start_second * video.fps) : round(self.end_second * video.fps)]
        return video


class Resize(Transformation):
    def __init__(self, new_width: int, new_height: int):
        self.new_width = new_width
        self.new_height = new_height

    def _resize_frame(self, frame: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
        return cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def apply(self, video: Video) -> Video:
        with Pool() as pool:
            frames_copy = pool.starmap(
                self._resize_frame,
                [(frame, self.new_width, self.new_height) for frame in video.frames],
            )
        video.frames = np.array(frames_copy)
        return video


class ResampleFPS(Transformation):
    def __init__(self, new_fps: int | float):
        self.new_fps = float(new_fps)

    def _downsample(self, video: Video) -> Video:
        target_frame_count = int(len(video.frames) * (self.new_fps / video.fps))
        new_frame_indices = np.round(np.linspace(0, len(video.frames) - 1, target_frame_count)).astype(int)
        video.frames = video.frames[new_frame_indices]
        video.fps = self.new_fps
        return video

    def _upsample(self, video: Video) -> Video:
        target_frame_count = int(len(video.frames) * (self.new_fps / video.fps))
        new_frame_indices = np.linspace(0, len(video.frames) - 1, target_frame_count)
        new_frames = []
        for i in tqdm(range(len(new_frame_indices) - 1)):
            # Interpolate between the two nearest frames
            ratio = new_frame_indices[i] % 1
            new_frame = (1 - ratio) * video.frames[int(new_frame_indices[i])] + ratio * video.frames[
                int(np.ceil(new_frame_indices[i]))
            ]
            new_frames.append(new_frame.astype(np.uint8))
        video.frames = np.array(new_frames, dtype=np.uint8)
        video.fps = self.new_fps
        return video

    def apply(self, video: Video) -> Video:
        if video.fps == self.new_fps:
            return video
        elif video.fps > self.new_fps:
            print(f"Downsampling video from {video.fps} to {self.new_fps} FPS.")
            video = self._downsample(video)
        else:
            print(f"Upsampling video from {video.fps} to {self.new_fps} FPS.")
            video = self._upsample(video)
        return video

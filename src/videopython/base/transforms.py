from abc import ABC, abstractmethod
from enum import Enum
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
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def apply(self, video: Video) -> Video:
        video = video[self.start : self.end]
        return video


class CutSeconds(Transformation):
    def __init__(self, start: float | int, end: float | int):
        self.start = start
        self.end = end

    def apply(self, video: Video) -> Video:
        video = video[round(self.start * video.fps) : round(self.end * video.fps)]
        return video


class Resize(Transformation):
    def __init__(self, width: int | None = None, height: int | None = None):
        self.width = width
        self.height = height
        if width is None and height is None:
            raise ValueError("You must provide either `width` or `height`!")

    def _resize_frame(self, frame: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
        return cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def apply(self, video: Video) -> Video:
        if self.width and self.height:
            new_height = self.height
            new_width = self.width
        elif self.height is None and self.width:
            video_height = video.video_shape[1]
            video_width = video.video_shape[2]
            new_height = round(video_height * (self.width / video_width))
            new_width = self.width
        elif self.width is None and self.height:
            video_height = video.video_shape[1]
            video_width = video.video_shape[2]
            new_width = round(video_width * (self.height / video_height))
            new_height = self.height

        print(f"Resizing video to: {new_width}x{new_height}!")
        with Pool() as pool:
            frames_copy = pool.starmap(
                self._resize_frame,
                [(frame, new_width, new_height) for frame in video.frames],
            )
        video.frames = np.array(frames_copy)
        return video


class ResampleFPS(Transformation):
    def __init__(self, fps: int | float):
        self.fps = float(fps)

    def _downsample(self, video: Video) -> Video:
        target_frame_count = int(len(video.frames) * (self.fps / video.fps))
        new_frame_indices = np.round(np.linspace(0, len(video.frames) - 1, target_frame_count)).astype(int)
        video.frames = video.frames[new_frame_indices]
        video.fps = self.fps
        return video

    def _upsample(self, video: Video) -> Video:
        target_frame_count = int(len(video.frames) * (self.fps / video.fps))
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
        video.fps = self.fps
        return video

    def apply(self, video: Video) -> Video:
        if video.fps == self.fps:
            return video
        elif video.fps > self.fps:
            print(f"Downsampling video from {video.fps} to {self.fps} FPS.")
            video = self._downsample(video)
        else:
            print(f"Upsampling video from {video.fps} to {self.fps} FPS.")
            video = self._upsample(video)
        return video


class CropMode(Enum):
    CENTER = "center"


class Crop(Transformation):
    def __init__(self, width: int, height: int, mode: CropMode = CropMode.CENTER):
        self.width = width
        self.height = height
        self.mode = mode

    def apply(self, video: Video) -> Video:
        if self.mode == CropMode.CENTER:
            current_shape = video.frame_shape[:2]
            center_height = current_shape[0] // 2
            center_width = current_shape[1] // 2
            width_offset = self.width // 2
            height_offset = self.height // 2
            video.frames = video.frames[
                :,
                center_height - height_offset : center_height + height_offset,
                center_width - width_offset : center_width + width_offset,
                :,
            ]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return video

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np


class Transformation(ABC):
    """Abstract class for transformation on frames of video."""

    @abstractmethod
    def apply(self, frames: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        return self.apply(frames)


class CutFrames(Transformation):
    """Cuts video from `start` to `end` frames."""

    def __init__(self, start: int, end: int):
        """Initializes transformation.
        
        Args:
            start: Start index of frames to cut.
            end: End index of frames to cut.
        """
        self.start = start
        self.end = end

    def apply(self, frames: np.ndarray) -> np.ndarray:
        """Applies transformation to frames.
        
        Args:
            frames: Frames to transform.

        Returns:
            Transformed frames.
        """
        if frames.shape[0] < self.end:
            raise ValueError(
                f"Video has only {frames.shape[0]} frames, but {self.end} are required"
            )
        return frames[self.start:self.end]


class CutSeconds(Transformation):
    """Cuts video from `start` to `end` seconds."""

    def __init__(self, start: int, end: int, fps: int):
        """Initializes transformation.
        
        Args:
            start: Start index of frames to cut.
            end: End index of frames to cut.
            fps: Frames per second of the video.
        """
        self.start = start * fps
        self.end = end * fps

    def apply(self, frames: np.ndarray) -> np.ndarray:
        """Applies transformation to frames.
        
        Args:
            frames: Frames to transform.

        Returns:
            Transformed frames.
        """
        if frames.shape[0] < self.end:
            raise ValueError(
                f"Video has only {frames.shape[0]} frames, but {self.end} are required"
            )
        return frames[self.start:self.end]


class CropStrategy(Enum):
    CENTER = "center"
    LEFT = "left"


class CropVideo(Transformation):
    """Crops video to desired shape."""

    def __init__(self,
                 output_shape: tuple[int, int],
                 strategy: CropStrategy = CropStrategy.CENTER):
        """Initializes transformation.
        
        Args:
            output_shape: Expected output shape.
            strategy: Cropping strategy. Can be "center", "left".
        """
        self.output_shape = output_shape
        self.strategy = strategy

    def apply(self, frames: np.ndarray) -> np.ndarray:
        """Applies transformation to frames.
        
        Args:
            frames: Frames to transform.

        Returns:
            Transformed frames.
        """
        if (frames.shape[1] > self.output_shape[0] or
                frames.shape[2] > self.output_shape[1]):
            raise ValueError(
                f"Input shape {frames.shape} is bigger than output shape {self.output_shape}"
            )

        match self.strategy:
            case CropStrategy.CENTER:
                return self._crop_center(frames)
            case CropStrategy.LEFT:
                return self._crop_left(frames)
            case _:
                raise ValueError(f"Unknown strategy {self.strategy}")

    def _crop_center(self, frames: np.ndarray) -> np.ndarray:
        """Crops frames from center.
        
        Args:
            frames: Frames to transform.

        Returns:
            Transformed frames.
        """
        height, width = frames.shape[1:3]
        target_height, target_width = self.output_shape
        start_height = (height - target_height) // 2
        start_width = (width - target_width) // 2
        return frames[:, start_height:start_height + target_height,
                      start_width:start_width + target_width, :]

    def _crop_left(self, frames: np.ndarray) -> np.ndarray:
        """Crops frames from left.
        
        Args:
            frames: Frames to transform.

        Returns:
            Transformed frames.
        """
        height, width = frames.shape[1:3]
        target_height, target_width = self.output_shape
        start_height = (height - target_height) // 2
        return frames[:, start_height:start_height +
                      target_height, :target_width, :]

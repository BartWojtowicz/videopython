from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

from videopython.base.video import Video

__all__ = [
    "Transformation",
    "CutFrames",
    "CutSeconds",
    "Resize",
    "ResampleFPS",
    "Crop",
    "CropMode",
]


class Transformation(ABC):
    """Abstract class for transformation on frames of video."""

    @abstractmethod
    def apply(self, video: Video) -> Video:
        pass


class CutFrames(Transformation):
    """Cuts video to a specific frame range."""

    def __init__(self, start: int, end: int):
        """Initialize frame cutter.

        Args:
            start: Start frame index (inclusive).
            end: End frame index (exclusive).
        """
        self.start = start
        self.end = end

    def apply(self, video: Video) -> Video:
        """Apply frame cut to video.

        Args:
            video: Input video.

        Returns:
            Video with frames from start to end.
        """
        video = video[self.start : self.end]
        return video


class CutSeconds(Transformation):
    """Cuts video to a specific time range in seconds."""

    def __init__(self, start: float | int, end: float | int):
        """Initialize time-based cutter.

        Args:
            start: Start time in seconds.
            end: End time in seconds.
        """
        self.start = start
        self.end = end

    def apply(self, video: Video) -> Video:
        """Apply time-based cut to video.

        Args:
            video: Input video.

        Returns:
            Video cut from start to end seconds.
        """
        video = video[round(self.start * video.fps) : round(self.end * video.fps)]
        return video


class Resize(Transformation):
    """Resizes video to specified dimensions, maintaining aspect ratio if only one dimension is provided."""

    def __init__(self, width: int | None = None, height: int | None = None):
        """Initialize resizer.

        Args:
            width: Target width in pixels, or None to maintain aspect ratio.
            height: Target height in pixels, or None to maintain aspect ratio.
        """
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
        """Resize video frames to target dimensions.

        Args:
            video: Input video.

        Returns:
            Resized video.
        """
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
    """Resamples video to a different frame rate, upsampling or downsampling as needed."""

    def __init__(self, fps: int | float):
        """Initialize FPS resampler.

        Args:
            fps: Target frames per second.
        """
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
        """Resample video to target FPS.

        Args:
            video: Input video.

        Returns:
            Video with target frame rate.
        """
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
    """Crops video to specified dimensions."""

    def __init__(self, width: int, height: int, mode: CropMode = CropMode.CENTER):
        """Initialize cropper.

        Args:
            width: Target crop width in pixels.
            height: Target crop height in pixels.
            mode: Crop mode, defaults to center crop.
        """
        self.width = width
        self.height = height
        self.mode = mode

    def apply(self, video: Video) -> Video:
        """Crop video to target dimensions.

        Args:
            video: Input video.

        Returns:
            Cropped video.
        """
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

"""Object swapping functionality for video editing.

This module provides tools to swap objects in videos using AI-powered
segmentation, inpainting, and compositing.

Example:
    >>> from videopython.base.video import Video
    >>> from videopython.ai.swapping import ObjectSwapper
    >>>
    >>> video = Video.from_path("street.mp4")
    >>> swapper = ObjectSwapper(backend="local")
    >>>
    >>> # Swap an object with a generated replacement
    >>> result = swapper.swap(video, source_object="red car", target_object="blue motorcycle")
    >>> Video.from_frames(result.swapped_frames, video.fps).save("output.mp4")
    >>>
    >>> # Or use a provided image
    >>> result = swapper.swap_with_image(video, source_object="logo", replacement_image="new_logo.png")
"""

from videopython.ai.swapping.inpainter import VideoInpainter
from videopython.ai.swapping.models import (
    InpaintingConfig,
    ObjectMask,
    ObjectTrack,
    SegmentationConfig,
    SwapConfig,
    SwapResult,
)
from videopython.ai.swapping.segmenter import ObjectSegmenter
from videopython.ai.swapping.swapper import ObjectSwapper

__all__ = [
    # Main classes
    "ObjectSwapper",
    "ObjectSegmenter",
    "VideoInpainter",
    # Data models
    "ObjectMask",
    "ObjectTrack",
    "SwapResult",
    # Configuration
    "SwapConfig",
    "SegmentationConfig",
    "InpaintingConfig",
]

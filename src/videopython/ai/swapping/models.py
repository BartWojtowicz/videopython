"""Data models for object swapping."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ObjectMask:
    """A mask representing an object in a single frame.

    Attributes:
        frame_index: Index of the frame this mask belongs to.
        mask: Binary mask array of shape (H, W) where True indicates object pixels.
        confidence: Confidence score of the segmentation (0.0 to 1.0).
        bounding_box: Optional bounding box as (x1, y1, x2, y2) normalized coordinates.
    """

    frame_index: int
    mask: np.ndarray
    confidence: float
    bounding_box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        """Validate mask shape and values."""
        if self.mask.ndim != 2:
            raise ValueError(f"Mask must be 2D, got shape {self.mask.shape}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    @property
    def height(self) -> int:
        """Height of the mask."""
        return self.mask.shape[0]

    @property
    def width(self) -> int:
        """Width of the mask."""
        return self.mask.shape[1]

    @property
    def area(self) -> int:
        """Number of pixels in the mask."""
        return int(np.sum(self.mask > 0))

    def dilate(self, kernel_size: int = 5) -> ObjectMask:
        """Return a dilated version of this mask.

        Args:
            kernel_size: Size of the dilation kernel.

        Returns:
            New ObjectMask with dilated mask.
        """
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(self.mask.astype(np.uint8), kernel, iterations=1)
        return ObjectMask(
            frame_index=self.frame_index,
            mask=dilated.astype(bool),
            confidence=self.confidence,
            bounding_box=self.bounding_box,
        )


@dataclass
class ObjectTrack:
    """A tracked object across multiple frames.

    Attributes:
        object_id: Unique identifier for this tracked object.
        masks: List of ObjectMask instances for each frame where object appears.
        label: Text label describing the object (e.g., "red car").
        start_frame: First frame index where object appears.
        end_frame: Last frame index where object appears.
    """

    object_id: str
    masks: list[ObjectMask]
    label: str
    start_frame: int
    end_frame: int

    @property
    def num_frames(self) -> int:
        """Number of frames this object appears in."""
        return len(self.masks)

    @property
    def frame_indices(self) -> list[int]:
        """List of frame indices where object appears."""
        return [m.frame_index for m in self.masks]

    @property
    def average_confidence(self) -> float:
        """Average confidence across all masks."""
        if not self.masks:
            return 0.0
        return sum(m.confidence for m in self.masks) / len(self.masks)

    def get_mask_for_frame(self, frame_index: int) -> ObjectMask | None:
        """Get the mask for a specific frame.

        Args:
            frame_index: The frame index to look up.

        Returns:
            The ObjectMask for that frame, or None if not present.
        """
        for mask in self.masks:
            if mask.frame_index == frame_index:
                return mask
        return None

    def get_masks_array(self) -> np.ndarray:
        """Get all masks as a stacked numpy array.

        Returns:
            Array of shape (N, H, W) where N is number of frames.
        """
        if not self.masks:
            raise ValueError("No masks in track")
        return np.stack([m.mask for m in self.masks], axis=0)


@dataclass
class SwapResult:
    """Result of an object swapping operation.

    Attributes:
        swapped_frames: Array of frames with object swapped, shape (N, H, W, C).
        object_track: The tracked object that was swapped.
        inpainted_frames: Frames with object removed (background only), shape (N, H, W, C).
        source_prompt: Text prompt used to identify source object.
        target_prompt: Text prompt for the replacement object (if generated).
        replacement_image: Path to replacement image (if provided).
    """

    swapped_frames: np.ndarray
    object_track: ObjectTrack
    inpainted_frames: np.ndarray | None = None
    source_prompt: str = ""
    target_prompt: str = ""
    replacement_image: str | None = None

    @property
    def num_frames(self) -> int:
        """Number of frames in the result."""
        return self.swapped_frames.shape[0]

    @property
    def frame_size(self) -> tuple[int, int]:
        """Size of frames as (height, width)."""
        return (self.swapped_frames.shape[1], self.swapped_frames.shape[2])

    @property
    def has_inpainted_frames(self) -> bool:
        """Check if inpainted frames are available."""
        return self.inpainted_frames is not None


@dataclass
class SegmentationConfig:
    """Configuration for object segmentation.

    Attributes:
        model_size: SAM2 model size ('tiny', 'small', 'base', 'large').
        points_per_side: Number of points to sample per side for auto-mask generation.
        pred_iou_thresh: IOU threshold for filtering predictions.
        stability_score_thresh: Stability threshold for filtering predictions.
    """

    model_size: str = "large"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_sizes = ["tiny", "small", "base", "large"]
        if self.model_size not in valid_sizes:
            raise ValueError(f"model_size must be one of {valid_sizes}, got {self.model_size}")


@dataclass
class InpaintingConfig:
    """Configuration for video inpainting.

    Attributes:
        model_id: HuggingFace model ID for inpainting.
        num_inference_steps: Number of diffusion steps.
        guidance_scale: Guidance scale for generation.
        mask_dilation: Kernel size for mask dilation (0 to disable).
        batch_size: Number of frames to process in parallel.
    """

    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    mask_dilation: int = 5
    batch_size: int = 1


@dataclass
class SwapConfig:
    """Configuration for object swapping pipeline.

    Attributes:
        segmentation: Configuration for segmentation.
        inpainting: Configuration for inpainting.
        composite_blend: Blend factor for compositing (0.0 = hard edge, 1.0 = soft blend).
        reference_frame: Frame index to use for initial segmentation.
    """

    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    inpainting: InpaintingConfig = field(default_factory=InpaintingConfig)
    composite_blend: float = 0.5
    reference_frame: int = 0

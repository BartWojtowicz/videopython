"""Object segmentation using SAM2 for video object tracking."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from videopython.ai._device import log_device_initialization, select_device
from videopython.ai.swapping.models import ObjectMask, ObjectTrack, SegmentationConfig

if TYPE_CHECKING:
    pass


# Model ID mappings for SAM2
SAM2_MODEL_IDS = {
    "tiny": "facebook/sam2-hiera-tiny",
    "small": "facebook/sam2-hiera-small",
    "base": "facebook/sam2-hiera-base-plus",
    "large": "facebook/sam2-hiera-large",
}

# GroundingDINO model for text-to-bbox
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


class ObjectSegmenter:
    """Segments and tracks objects in video using SAM2.

    Supports multiple input methods:
    - Text prompts (via GroundingDINO for bounding box detection)
    - Point prompts (click on object)
    - Box prompts (draw bounding box)

    Example:
        >>> from videopython.ai.swapping import ObjectSegmenter
        >>> from videopython.base.video import Video
        >>>
        >>> video = Video.from_path("video.mp4")
        >>> segmenter = ObjectSegmenter()
        >>>
        >>> # Segment using text prompt
        >>> track = segmenter.segment_object(video.frames, "red car")
        >>>
        >>> # Segment using point
        >>> track = segmenter.segment_with_point(video.frames, (100, 200))
        >>>
        >>> # Segment using bounding box
        >>> track = segmenter.segment_with_box(video.frames, (50, 50, 200, 200))
    """

    def __init__(
        self,
        config: SegmentationConfig | None = None,
        device: str | None = None,
    ):
        """Initialize the object segmenter.

        Args:
            config: Segmentation configuration. Uses defaults if None.
            device: Device for local models ('cuda', 'mps', or 'cpu').
        """
        self.config = config or SegmentationConfig()
        self.device = device

        # Lazy-loaded models
        self._sam2_model: Any = None
        self._sam2_processor: Any = None
        self._grounding_dino_model: Any = None
        self._grounding_dino_processor: Any = None

    def _get_device(self) -> str:
        """Get the device to use for inference."""
        return select_device(self.device, mps_allowed=True)

    def _init_sam2(self) -> None:
        """Initialize SAM2 model."""
        import torch
        from transformers import Sam2Model, Sam2Processor  # type: ignore[attr-defined]

        model_id = SAM2_MODEL_IDS[self.config.model_size]
        requested_device = self.device
        device = self._get_device()

        self._sam2_processor = Sam2Processor.from_pretrained(model_id)
        self._sam2_model = Sam2Model.from_pretrained(model_id)

        # Use appropriate dtype based on device
        if device == "cuda":
            self._sam2_model = self._sam2_model.to(device).to(torch.float16)
        else:
            self._sam2_model = self._sam2_model.to(device)
        log_device_initialization(
            "ObjectSegmenter",
            requested_device=requested_device,
            resolved_device=device,
        )

    def _init_grounding_dino(self) -> None:
        """Initialize GroundingDINO model for text-to-bbox."""
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor  # type: ignore[attr-defined]

        requested_device = self.device
        device = self._get_device()

        self._grounding_dino_processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
        self._grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID)

        if device == "cuda":
            self._grounding_dino_model = self._grounding_dino_model.to(device).to(torch.float16)
        else:
            self._grounding_dino_model = self._grounding_dino_model.to(device)
        log_device_initialization(
            "ObjectSegmenter",
            requested_device=requested_device,
            resolved_device=device,
        )

    def _detect_object_bbox(
        self,
        frame: np.ndarray,
        text_prompt: str,
        threshold: float = 0.1,
        text_threshold: float = 0.1,
    ) -> tuple[float, float, float, float] | None:
        """Detect object bounding box using GroundingDINO.

        Args:
            frame: RGB frame array of shape (H, W, C).
            text_prompt: Text description of object to find.
            threshold: Confidence threshold for box detection.
            text_threshold: Confidence threshold for text matching.

        Returns:
            Bounding box as (x1, y1, x2, y2) in pixel coordinates, or None if not found.
        """
        if self._grounding_dino_model is None:
            self._init_grounding_dino()

        import torch
        from PIL import Image

        device = self._get_device()

        # Convert to PIL Image
        image = Image.fromarray(frame)

        # Process inputs
        inputs = self._grounding_dino_processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._grounding_dino_model(**inputs)

        # Post-process results
        results = self._grounding_dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],  # (height, width)
        )[0]

        if len(results["boxes"]) == 0:
            return None

        # Get the highest confidence box
        best_idx = results["scores"].argmax().item()
        box = results["boxes"][best_idx].cpu().numpy()

        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))

    def _segment_frame_with_point(
        self,
        frame: np.ndarray,
        point: tuple[int, int],
        frame_index: int,
    ) -> ObjectMask:
        """Segment a single frame using a point prompt.

        Args:
            frame: RGB frame array of shape (H, W, C).
            point: Point coordinates as (x, y).
            frame_index: Index of this frame.

        Returns:
            ObjectMask for this frame.
        """
        if self._sam2_model is None:
            self._init_sam2()

        import torch
        from PIL import Image

        device = self._get_device()

        # Convert to PIL Image
        image = Image.fromarray(frame)

        # Process inputs with point prompt
        inputs = self._sam2_processor(
            images=image,
            input_points=[[[point[0], point[1]]]],
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._sam2_model(**inputs)

        # Get mask from outputs - post_process_masks returns list of tensors
        masks = self._sam2_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
        )[0]  # First batch item: shape (num_prompts, num_masks, H, W)

        # Get best mask (highest IoU score)
        best_mask_idx = outputs.iou_scores[0, 0].argmax().item()
        mask = masks[0, best_mask_idx].cpu().numpy().astype(bool)
        confidence = float(outputs.iou_scores[0, 0, best_mask_idx].cpu().numpy())

        return ObjectMask(
            frame_index=frame_index,
            mask=mask,
            confidence=confidence,
        )

    def _segment_frame_with_box(
        self,
        frame: np.ndarray,
        box: tuple[float, float, float, float],
        frame_index: int,
    ) -> ObjectMask:
        """Segment a single frame using a box prompt.

        Args:
            frame: RGB frame array of shape (H, W, C).
            box: Bounding box as (x1, y1, x2, y2).
            frame_index: Index of this frame.

        Returns:
            ObjectMask for this frame.
        """
        if self._sam2_model is None:
            self._init_sam2()

        import torch
        from PIL import Image

        device = self._get_device()

        # Convert to PIL Image
        image = Image.fromarray(frame)

        # Process inputs with box prompt
        inputs = self._sam2_processor(
            images=image,
            input_boxes=[[[box[0], box[1], box[2], box[3]]]],
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._sam2_model(**inputs)

        # Get mask from outputs - post_process_masks returns list of tensors
        masks = self._sam2_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
        )[0]  # First batch item: shape (num_prompts, num_masks, H, W)

        # Get best mask (highest IoU score)
        best_mask_idx = outputs.iou_scores[0, 0].argmax().item()
        mask = masks[0, best_mask_idx].cpu().numpy().astype(bool)
        confidence = float(outputs.iou_scores[0, 0, best_mask_idx].cpu().numpy())

        # Normalize bounding box
        h, w = frame.shape[:2]
        normalized_box = (box[0] / w, box[1] / h, box[2] / w, box[3] / h)

        return ObjectMask(
            frame_index=frame_index,
            mask=mask,
            confidence=confidence,
            bounding_box=normalized_box,
        )

    def _propagate_mask_to_video(
        self,
        frames: np.ndarray,
        initial_mask: ObjectMask,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> list[ObjectMask]:
        """Propagate a mask through video frames using SAM2 video mode.

        This uses SAM2's video propagation capability for temporal consistency.

        Args:
            frames: Video frames array of shape (N, H, W, C).
            initial_mask: Initial mask from reference frame.
            progress_callback: Optional progress callback.

        Returns:
            List of ObjectMask for each frame.
        """
        if self._sam2_model is None:
            self._init_sam2()

        import torch
        from PIL import Image

        device = self._get_device()
        num_frames = frames.shape[0]
        masks = []

        # Convert initial mask to input format
        initial_frame_idx = initial_mask.frame_index

        # Process frames in sequence, using mask propagation
        # SAM2 video mode: start from reference frame and propagate bidirectionally

        if progress_callback:
            progress_callback("Propagating mask through video", 0.0)

        # For each frame, use the previous frame's mask as guidance
        current_mask = initial_mask.mask

        for i in range(num_frames):
            if progress_callback:
                progress_callback("Propagating mask through video", i / num_frames)

            if i == initial_frame_idx:
                # Use the initial mask directly
                masks.append(initial_mask)
                continue

            frame = frames[i]
            image = Image.fromarray(frame)

            # Find center point of current mask to use as prompt
            if current_mask.sum() > 0:
                y_coords, x_coords = np.where(current_mask)
                center_y = int(y_coords.mean())
                center_x = int(x_coords.mean())
                point = (center_x, center_y)

                # Also compute bounding box from mask
                x1, y1 = x_coords.min(), y_coords.min()
                x2, y2 = x_coords.max(), y_coords.max()
                box = (float(x1), float(y1), float(x2), float(y2))

                # Use box prompt for better tracking
                inputs = self._sam2_processor(
                    images=image,
                    input_boxes=[[[box[0], box[1], box[2], box[3]]]],
                    return_tensors="pt",
                )
            else:
                # Fallback to center point if mask is empty
                h, w = frame.shape[:2]
                point = (w // 2, h // 2)
                inputs = self._sam2_processor(
                    images=image,
                    input_points=[[[point[0], point[1]]]],
                    return_tensors="pt",
                )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._sam2_model(**inputs)

            frame_masks = self._sam2_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
            )[0]  # First batch item: shape (num_prompts, num_masks, H, W)

            best_mask_idx = outputs.iou_scores[0, 0].argmax().item()
            mask = frame_masks[0, best_mask_idx].cpu().numpy().astype(bool)
            confidence = float(outputs.iou_scores[0, 0, best_mask_idx].cpu().numpy())

            object_mask = ObjectMask(
                frame_index=i,
                mask=mask,
                confidence=confidence,
            )
            masks.append(object_mask)

            # Update current mask for next iteration
            current_mask = mask

        if progress_callback:
            progress_callback("Mask propagation complete", 1.0)

        # Sort masks by frame index
        masks.sort(key=lambda m: m.frame_index)
        return masks

    def segment_object(
        self,
        frames: np.ndarray,
        prompt: str,
        reference_frame: int = 0,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ObjectTrack:
        """Segment an object in video using a text prompt.

        Uses GroundingDINO to detect the object bounding box from the text prompt,
        then SAM2 to segment and track the object across all frames.

        Args:
            frames: Video frames array of shape (N, H, W, C) in RGB format.
            prompt: Text description of the object to segment (e.g., "red car").
            reference_frame: Frame index to use for initial detection. Default 0.
            progress_callback: Optional callback for progress updates.
                Called with (stage_name, progress_fraction).

        Returns:
            ObjectTrack containing masks for all frames where object appears.

        Raises:
            ValueError: If object is not found in reference frame.
        """
        if progress_callback:
            progress_callback("Detecting object from text prompt", 0.0)

        # Use GroundingDINO to find bounding box
        ref_frame = frames[reference_frame]
        box = self._detect_object_bbox(ref_frame, prompt)

        if box is None:
            raise ValueError(f"Could not find '{prompt}' in frame {reference_frame}")

        if progress_callback:
            progress_callback("Segmenting object in reference frame", 0.1)

        # Segment reference frame with box
        initial_mask = self._segment_frame_with_box(ref_frame, box, reference_frame)

        if progress_callback:
            progress_callback("Propagating mask through video", 0.2)

        # Propagate mask through video
        masks = self._propagate_mask_to_video(
            frames,
            initial_mask,
            progress_callback=progress_callback,
        )

        # Find start and end frames (where mask has non-zero area)
        valid_masks = [m for m in masks if m.area > 0]
        if not valid_masks:
            start_frame = reference_frame
            end_frame = reference_frame
        else:
            start_frame = min(m.frame_index for m in valid_masks)
            end_frame = max(m.frame_index for m in valid_masks)

        return ObjectTrack(
            object_id=str(uuid.uuid4()),
            masks=masks,
            label=prompt,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def segment_with_point(
        self,
        frames: np.ndarray,
        point: tuple[int, int],
        reference_frame: int = 0,
        label: str = "object",
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ObjectTrack:
        """Segment an object using a point click.

        Args:
            frames: Video frames array of shape (N, H, W, C) in RGB format.
            point: Point coordinates as (x, y) in pixel coordinates.
            reference_frame: Frame index where point is specified. Default 0.
            label: Label for the tracked object. Default "object".
            progress_callback: Optional callback for progress updates.

        Returns:
            ObjectTrack containing masks for all frames.
        """
        if progress_callback:
            progress_callback("Segmenting object from point", 0.0)

        # Segment reference frame
        ref_frame = frames[reference_frame]
        initial_mask = self._segment_frame_with_point(ref_frame, point, reference_frame)

        if progress_callback:
            progress_callback("Propagating mask through video", 0.1)

        # Propagate through video
        masks = self._propagate_mask_to_video(
            frames,
            initial_mask,
            progress_callback=progress_callback,
        )

        valid_masks = [m for m in masks if m.area > 0]
        if not valid_masks:
            start_frame = reference_frame
            end_frame = reference_frame
        else:
            start_frame = min(m.frame_index for m in valid_masks)
            end_frame = max(m.frame_index for m in valid_masks)

        return ObjectTrack(
            object_id=str(uuid.uuid4()),
            masks=masks,
            label=label,
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def segment_with_box(
        self,
        frames: np.ndarray,
        box: tuple[float, float, float, float],
        reference_frame: int = 0,
        label: str = "object",
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> ObjectTrack:
        """Segment an object using a bounding box.

        Args:
            frames: Video frames array of shape (N, H, W, C) in RGB format.
            box: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
            reference_frame: Frame index where box is specified. Default 0.
            label: Label for the tracked object. Default "object".
            progress_callback: Optional callback for progress updates.

        Returns:
            ObjectTrack containing masks for all frames.
        """
        if progress_callback:
            progress_callback("Segmenting object from box", 0.0)

        # Segment reference frame
        ref_frame = frames[reference_frame]
        initial_mask = self._segment_frame_with_box(ref_frame, box, reference_frame)

        if progress_callback:
            progress_callback("Propagating mask through video", 0.1)

        # Propagate through video
        masks = self._propagate_mask_to_video(
            frames,
            initial_mask,
            progress_callback=progress_callback,
        )

        valid_masks = [m for m in masks if m.area > 0]
        if not valid_masks:
            start_frame = reference_frame
            end_frame = reference_frame
        else:
            start_frame = min(m.frame_index for m in valid_masks)
            end_frame = max(m.frame_index for m in valid_masks)

        return ObjectTrack(
            object_id=str(uuid.uuid4()),
            masks=masks,
            label=label,
            start_frame=start_frame,
            end_frame=end_frame,
        )

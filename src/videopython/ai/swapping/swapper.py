"""Object swapping orchestrator combining segmentation, inpainting, and compositing."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from videopython.ai.backends import ObjectSwapperBackend, UnsupportedBackendError
from videopython.ai.config import get_default_backend
from videopython.ai.swapping.inpainter import VideoInpainter
from videopython.ai.swapping.models import SwapConfig, SwapResult
from videopython.ai.swapping.segmenter import ObjectSegmenter

if TYPE_CHECKING:
    from videopython.base.video import Video


class ObjectSwapper:
    """Swaps objects in videos using segmentation, inpainting, and compositing.

    The object swapping pipeline:
    1. Segment source object using SAM2 (track across frames)
    2. Inpaint background where object was removed
    3. Composite replacement (generated or provided image) into cleaned background

    Example:
        >>> from videopython.base.video import Video
        >>> from videopython.ai.swapping import ObjectSwapper
        >>>
        >>> video = Video.from_path("street.mp4")
        >>> swapper = ObjectSwapper(backend="local")
        >>>
        >>> # Option A: Generate replacement from prompt
        >>> result = swapper.swap(video, source_object="red car", target_object="blue motorcycle")
        >>>
        >>> # Option B: Use provided image
        >>> result = swapper.swap_with_image(
        ...     video, source_object="red car", replacement_image="bike.png"
        ... )
        >>>
        >>> # Get result
        >>> swapped_video = Video.from_frames(result.swapped_frames, video.fps)
    """

    SUPPORTED_BACKENDS: list[str] = ["local", "replicate"]

    def __init__(
        self,
        backend: ObjectSwapperBackend | None = None,
        config: SwapConfig | None = None,
        device: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the object swapper.

        Args:
            backend: Backend to use ('local' or 'replicate').
                If None, uses config default or 'local'.
            config: Configuration for the swapping pipeline.
            device: Device for local models ('cuda', 'mps', or 'cpu').
            api_key: API key for cloud backends (Replicate).
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("object_swapper")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: ObjectSwapperBackend = resolved_backend  # type: ignore[assignment]
        self.config = config or SwapConfig()
        self.device = device
        self.api_key = api_key

        # Lazy-loaded components
        self._segmenter: ObjectSegmenter | None = None
        self._inpainter: VideoInpainter | None = None
        self._image_generator: Any = None

    def _get_segmenter(self) -> ObjectSegmenter:
        """Get or create the object segmenter."""
        if self._segmenter is None:
            self._segmenter = ObjectSegmenter(
                backend=self.backend,
                config=self.config.segmentation,
                device=self.device,
                api_key=self.api_key,
            )
        return self._segmenter

    def _get_inpainter(self) -> VideoInpainter:
        """Get or create the video inpainter."""
        if self._inpainter is None:
            self._inpainter = VideoInpainter(
                backend=self.backend,
                config=self.config.inpainting,
                device=self.device,
                api_key=self.api_key,
            )
        return self._inpainter

    def _get_image_generator(self) -> Any:
        """Get or create the image generator for target object generation."""
        if self._image_generator is None:
            from videopython.ai.generation import TextToImage

            self._image_generator = TextToImage(backend="local")
        return self._image_generator

    def _generate_replacement_image(
        self,
        target_prompt: str,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Generate a replacement image from a text prompt.

        Args:
            target_prompt: Text description of replacement object.
            width: Target width.
            height: Target height.

        Returns:
            Generated image as RGB array.
        """
        generator = self._get_image_generator()

        # Generate image at requested size
        image = generator.generate(
            prompt=target_prompt,
            width=width,
            height=height,
        )

        return np.array(image)

    def _load_replacement_image(self, image_path: str | Path) -> np.ndarray:
        """Load a replacement image from file.

        Args:
            image_path: Path to the replacement image.

        Returns:
            Image as RGB array.
        """
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        return np.array(image)

    def _composite_replacement(
        self,
        background: np.ndarray,
        replacement: np.ndarray,
        mask: np.ndarray,
        blend_factor: float = 0.5,
    ) -> np.ndarray:
        """Composite replacement image onto background using mask.

        Args:
            background: Background frame of shape (H, W, C).
            replacement: Replacement image of shape (H, W, C).
            mask: Binary mask of shape (H, W) indicating replacement region.
            blend_factor: Edge blending factor (0=hard, 1=soft).

        Returns:
            Composited frame of shape (H, W, C).
        """
        import cv2

        # Resize replacement to match background
        h, w = background.shape[:2]
        replacement_resized = cv2.resize(replacement, (w, h), interpolation=cv2.INTER_LINEAR)

        # Create soft edge mask if blend_factor > 0
        if blend_factor > 0:
            # Blur mask edges
            blur_size = max(3, int(min(h, w) * blend_factor * 0.05))
            if blur_size % 2 == 0:
                blur_size += 1
            soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (blur_size, blur_size), 0)  # type: ignore[type-var]
        else:
            soft_mask = mask.astype(np.float32)

        # Expand mask to 3 channels
        soft_mask_3d = soft_mask[:, :, np.newaxis]

        # Composite: result = bg * (1 - mask) + replacement * mask
        result = background * (1 - soft_mask_3d) + replacement_resized * soft_mask_3d

        return result.astype(np.uint8)

    def _composite_video(
        self,
        inpainted_frames: np.ndarray,
        replacement: np.ndarray,
        track: Any,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> np.ndarray:
        """Composite replacement onto all video frames.

        Args:
            inpainted_frames: Background frames with object removed.
            replacement: Replacement image.
            track: Object track with masks.
            progress_callback: Progress callback.

        Returns:
            Composited video frames.
        """
        num_frames = inpainted_frames.shape[0]
        composited = []

        for i in range(num_frames):
            if progress_callback:
                progress_callback("Compositing frames", i / num_frames)

            mask_obj = track.get_mask_for_frame(i)
            if mask_obj is None or mask_obj.area == 0:
                # No mask, keep inpainted frame
                composited.append(inpainted_frames[i])
                continue

            # Composite replacement
            frame = self._composite_replacement(
                background=inpainted_frames[i],
                replacement=replacement,
                mask=mask_obj.mask,
                blend_factor=self.config.composite_blend,
            )
            composited.append(frame)

        if progress_callback:
            progress_callback("Compositing complete", 1.0)

        return np.stack(composited, axis=0)

    def swap(
        self,
        video: Video,
        source_object: str,
        target_object: str,
        reference_frame: int | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> SwapResult:
        """Swap an object in video with a generated replacement.

        Segments the source object, removes it via inpainting, and composites
        a generated replacement image based on the target prompt.

        Args:
            video: Input video to process.
            source_object: Text description of object to replace (e.g., "red car").
            target_object: Text description of replacement object (e.g., "blue motorcycle").
            reference_frame: Frame index for initial segmentation. Default: config value.
            progress_callback: Optional callback for progress updates.
                Called with (stage_name, progress_fraction).

        Returns:
            SwapResult containing swapped frames and metadata.

        Example:
            >>> result = swapper.swap(video, "person", "robot")
            >>> Video.from_frames(result.swapped_frames, video.fps).save("output.mp4")
        """
        ref_frame = reference_frame if reference_frame is not None else self.config.reference_frame
        frames = video.frames

        # Stage 1: Segment source object
        if progress_callback:
            progress_callback("Segmenting source object", 0.0)

        segmenter = self._get_segmenter()
        track = segmenter.segment_object(
            frames=frames,
            prompt=source_object,
            reference_frame=ref_frame,
            progress_callback=lambda msg, p: progress_callback(msg, p * 0.3) if progress_callback else None,
        )

        # Stage 2: Inpaint background
        if progress_callback:
            progress_callback("Inpainting background", 0.3)

        inpainter = self._get_inpainter()
        inpainted_frames = inpainter.inpaint(
            frames=frames,
            track=track,
            prompt="background, seamless, natural",
            progress_callback=lambda msg, p: progress_callback(msg, 0.3 + p * 0.3) if progress_callback else None,
        )

        # Stage 3: Generate replacement image
        if progress_callback:
            progress_callback("Generating replacement object", 0.6)

        h, w = frames.shape[1:3]
        replacement = self._generate_replacement_image(
            target_prompt=target_object,
            width=w,
            height=h,
        )

        # Stage 4: Composite replacement
        if progress_callback:
            progress_callback("Compositing frames", 0.7)

        swapped_frames = self._composite_video(
            inpainted_frames=inpainted_frames,
            replacement=replacement,
            track=track,
            progress_callback=lambda msg, p: progress_callback(msg, 0.7 + p * 0.3) if progress_callback else None,
        )

        if progress_callback:
            progress_callback("Complete", 1.0)

        return SwapResult(
            swapped_frames=swapped_frames,
            object_track=track,
            inpainted_frames=inpainted_frames,
            source_prompt=source_object,
            target_prompt=target_object,
        )

    def swap_with_image(
        self,
        video: Video,
        source_object: str,
        replacement_image: str | Path,
        reference_frame: int | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> SwapResult:
        """Swap an object in video with a provided replacement image.

        Segments the source object, removes it via inpainting, and composites
        the provided replacement image in its place.

        Args:
            video: Input video to process.
            source_object: Text description of object to replace (e.g., "red car").
            replacement_image: Path to replacement image file.
            reference_frame: Frame index for initial segmentation. Default: config value.
            progress_callback: Optional callback for progress updates.

        Returns:
            SwapResult containing swapped frames and metadata.

        Example:
            >>> result = swapper.swap_with_image(video, "logo", "new_logo.png")
            >>> Video.from_frames(result.swapped_frames, video.fps).save("output.mp4")
        """
        ref_frame = reference_frame if reference_frame is not None else self.config.reference_frame
        frames = video.frames

        # Stage 1: Segment source object
        if progress_callback:
            progress_callback("Segmenting source object", 0.0)

        segmenter = self._get_segmenter()
        track = segmenter.segment_object(
            frames=frames,
            prompt=source_object,
            reference_frame=ref_frame,
            progress_callback=lambda msg, p: progress_callback(msg, p * 0.3) if progress_callback else None,
        )

        # Stage 2: Inpaint background
        if progress_callback:
            progress_callback("Inpainting background", 0.3)

        inpainter = self._get_inpainter()
        inpainted_frames = inpainter.inpaint(
            frames=frames,
            track=track,
            prompt="background, seamless, natural",
            progress_callback=lambda msg, p: progress_callback(msg, 0.3 + p * 0.4) if progress_callback else None,
        )

        # Stage 3: Load replacement image
        if progress_callback:
            progress_callback("Loading replacement image", 0.7)

        replacement = self._load_replacement_image(replacement_image)

        # Stage 4: Composite replacement
        if progress_callback:
            progress_callback("Compositing frames", 0.75)

        swapped_frames = self._composite_video(
            inpainted_frames=inpainted_frames,
            replacement=replacement,
            track=track,
            progress_callback=lambda msg, p: progress_callback(msg, 0.75 + p * 0.25) if progress_callback else None,
        )

        if progress_callback:
            progress_callback("Complete", 1.0)

        return SwapResult(
            swapped_frames=swapped_frames,
            object_track=track,
            inpainted_frames=inpainted_frames,
            source_prompt=source_object,
            replacement_image=str(replacement_image),
        )

    def remove_object(
        self,
        video: Video,
        source_object: str,
        reference_frame: int | None = None,
        inpaint_prompt: str = "background, seamless, natural",
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> SwapResult:
        """Remove an object from video without replacement.

        Segments the object and inpaints the background to remove it cleanly.

        Args:
            video: Input video to process.
            source_object: Text description of object to remove.
            reference_frame: Frame index for initial segmentation.
            inpaint_prompt: Prompt to guide background generation.
            progress_callback: Optional progress callback.

        Returns:
            SwapResult with inpainted frames (swapped_frames equals inpainted_frames).

        Example:
            >>> result = swapper.remove_object(video, "watermark")
            >>> Video.from_frames(result.swapped_frames, video.fps).save("clean.mp4")
        """
        ref_frame = reference_frame if reference_frame is not None else self.config.reference_frame
        frames = video.frames

        # Stage 1: Segment object
        if progress_callback:
            progress_callback("Segmenting object to remove", 0.0)

        segmenter = self._get_segmenter()
        track = segmenter.segment_object(
            frames=frames,
            prompt=source_object,
            reference_frame=ref_frame,
            progress_callback=lambda msg, p: progress_callback(msg, p * 0.4) if progress_callback else None,
        )

        # Stage 2: Inpaint to remove
        if progress_callback:
            progress_callback("Removing object", 0.4)

        inpainter = self._get_inpainter()
        inpainted_frames = inpainter.inpaint(
            frames=frames,
            track=track,
            prompt=inpaint_prompt,
            progress_callback=lambda msg, p: progress_callback(msg, 0.4 + p * 0.6) if progress_callback else None,
        )

        if progress_callback:
            progress_callback("Complete", 1.0)

        return SwapResult(
            swapped_frames=inpainted_frames,
            object_track=track,
            inpainted_frames=inpainted_frames,
            source_prompt=source_object,
        )

    def segment_only(
        self,
        video: Video,
        source_object: str,
        reference_frame: int | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> SwapResult:
        """Segment an object without swapping or inpainting.

        Useful for previewing segmentation results before full processing.

        Args:
            video: Input video to process.
            source_object: Text description of object to segment.
            reference_frame: Frame index for initial segmentation.
            progress_callback: Optional progress callback.

        Returns:
            SwapResult with original frames and object track (no swapping performed).
        """
        ref_frame = reference_frame if reference_frame is not None else self.config.reference_frame
        frames = video.frames

        segmenter = self._get_segmenter()
        track = segmenter.segment_object(
            frames=frames,
            prompt=source_object,
            reference_frame=ref_frame,
            progress_callback=progress_callback,
        )

        return SwapResult(
            swapped_frames=frames.copy(),
            object_track=track,
            source_prompt=source_object,
        )

    @staticmethod
    def visualize_track(
        frames: np.ndarray,
        track: Any,
        color: tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay object masks on video frames for visualization.

        Args:
            frames: Video frames array of shape (N, H, W, C).
            track: ObjectTrack to visualize.
            color: RGB color for mask overlay.
            alpha: Opacity of mask overlay (0-1).

        Returns:
            Frames with mask overlay.
        """
        visualized = frames.copy()
        overlay_color = np.array(color, dtype=np.float32)

        for i in range(frames.shape[0]):
            mask_obj = track.get_mask_for_frame(i)
            if mask_obj is None or mask_obj.area == 0:
                continue

            mask = mask_obj.mask
            mask_3d = mask[:, :, np.newaxis]

            # Blend original with colored overlay in masked region
            original = visualized[i].astype(np.float32)
            colored = original * (1 - alpha) + overlay_color * alpha
            visualized[i] = np.where(mask_3d, colored, original).astype(np.uint8)

        return visualized

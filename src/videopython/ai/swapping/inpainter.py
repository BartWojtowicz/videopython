"""Video inpainting using diffusion models to remove and fill objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from tqdm import tqdm

from videopython.ai.backends import ObjectSwapperBackend, UnsupportedBackendError, get_api_key
from videopython.ai.config import get_default_backend
from videopython.ai.swapping.models import InpaintingConfig, ObjectTrack

if TYPE_CHECKING:
    pass


class VideoInpainter:
    """Inpaints video frames to remove objects and fill backgrounds.

    Uses SDXL-inpainting model to remove objects defined by masks and generate
    plausible background content in their place.

    Example:
        >>> from videopython.ai.swapping import VideoInpainter, ObjectSegmenter
        >>> from videopython.base.video import Video
        >>>
        >>> video = Video.from_path("video.mp4")
        >>> segmenter = ObjectSegmenter()
        >>> track = segmenter.segment_object(video.frames, "person")
        >>>
        >>> inpainter = VideoInpainter()
        >>> inpainted = inpainter.inpaint(video.frames, track, prompt="empty room")
    """

    SUPPORTED_BACKENDS: list[str] = ["local", "replicate"]

    def __init__(
        self,
        backend: ObjectSwapperBackend | None = None,
        config: InpaintingConfig | None = None,
        device: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the video inpainter.

        Args:
            backend: Backend to use ('local' or 'replicate').
                If None, uses config default or 'local'.
            config: Inpainting configuration. Uses defaults if None.
            device: Device for local models ('cuda', 'mps', or 'cpu').
            api_key: API key for cloud backends.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("object_swapper")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: ObjectSwapperBackend = resolved_backend  # type: ignore[assignment]
        self.config = config or InpaintingConfig()
        self.device = device
        self.api_key = api_key

        # Lazy-loaded model
        self._inpaint_pipeline: Any = None

    def _get_device(self) -> str:
        """Get the device to use for inference."""
        if self.device:
            return self.device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _init_inpaint_pipeline(self) -> None:
        """Initialize the inpainting diffusion pipeline."""
        import torch
        from diffusers import AutoPipelineForInpainting

        device = self._get_device()

        self._inpaint_pipeline = AutoPipelineForInpainting.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        self._inpaint_pipeline.to(device)

        # Enable memory optimizations for CUDA
        if device == "cuda":
            self._inpaint_pipeline.enable_model_cpu_offload()

    def _dilate_mask(self, mask: np.ndarray, kernel_size: int) -> np.ndarray:
        """Dilate a binary mask to ensure clean inpainting edges.

        Args:
            mask: Binary mask array of shape (H, W).
            kernel_size: Size of dilation kernel.

        Returns:
            Dilated mask array.
        """
        if kernel_size <= 0:
            return mask

        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        return dilated.astype(bool)

    def _inpaint_frame_local(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        """Inpaint a single frame using local SDXL-inpainting.

        Args:
            frame: RGB frame array of shape (H, W, C).
            mask: Binary mask of shape (H, W) where True = inpaint region.
            prompt: Text prompt to guide inpainting.

        Returns:
            Inpainted frame array of shape (H, W, C).
        """
        if self._inpaint_pipeline is None:
            self._init_inpaint_pipeline()

        from PIL import Image

        # Convert to PIL Images
        image = Image.fromarray(frame)
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))

        # Resize to multiple of 8 (required by SDXL)
        original_size = image.size
        new_w = (original_size[0] // 8) * 8
        new_h = (original_size[1] // 8) * 8
        image_resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mask_resized = mask_image.resize((new_w, new_h), Image.Resampling.NEAREST)

        # Run inpainting
        result = self._inpaint_pipeline(
            prompt=prompt,
            image=image_resized,
            mask_image=mask_resized,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
        ).images[0]

        # Resize back to original size
        result = result.resize(original_size, Image.Resampling.LANCZOS)

        return np.array(result)

    def _inpaint_replicate(
        self,
        frames: np.ndarray,
        track: ObjectTrack,
        prompt: str,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> np.ndarray:
        """Inpaint video frames using Replicate API.

        Args:
            frames: Video frames array of shape (N, H, W, C).
            track: Object track containing masks to inpaint.
            prompt: Text prompt to guide inpainting.
            progress_callback: Progress callback.

        Returns:
            Inpainted frames array.
        """
        import base64
        import io

        import replicate
        from PIL import Image

        api_key = get_api_key("replicate", self.api_key)
        client = replicate.Client(api_token=api_key)

        inpainted_frames = []
        num_frames = frames.shape[0]

        for i, frame in enumerate(tqdm(frames, desc="Inpainting frames")):
            if progress_callback:
                progress_callback("Inpainting frames", i / num_frames)

            mask_obj = track.get_mask_for_frame(i)
            if mask_obj is None or mask_obj.area == 0:
                # No mask for this frame, keep original
                inpainted_frames.append(frame)
                continue

            mask = mask_obj.mask
            if self.config.mask_dilation > 0:
                mask = self._dilate_mask(mask, self.config.mask_dilation)

            # Convert frame and mask to base64
            frame_image = Image.fromarray(frame)
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))

            frame_buffer = io.BytesIO()
            mask_buffer = io.BytesIO()
            frame_image.save(frame_buffer, format="PNG")
            mask_image.save(mask_buffer, format="PNG")

            frame_b64 = base64.b64encode(frame_buffer.getvalue()).decode("utf-8")
            mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")

            # Run SDXL inpainting on Replicate
            output = client.run(
                "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                input={
                    "image": f"data:image/png;base64,{frame_b64}",
                    "mask": f"data:image/png;base64,{mask_b64}",
                    "prompt": prompt,
                    "num_inference_steps": self.config.num_inference_steps,
                    "guidance_scale": self.config.guidance_scale,
                },
            )

            # Fetch result image
            import requests

            result_url = output[0] if isinstance(output, list) else output
            response = requests.get(str(result_url))
            result_image = Image.open(io.BytesIO(response.content))
            result_frame = np.array(result_image.convert("RGB"))

            inpainted_frames.append(result_frame)

        if progress_callback:
            progress_callback("Inpainting complete", 1.0)

        return np.stack(inpainted_frames, axis=0)

    def inpaint(
        self,
        frames: np.ndarray,
        track: ObjectTrack,
        prompt: str = "background, seamless",
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> np.ndarray:
        """Inpaint video frames to remove tracked object.

        Removes the object defined by the track masks and fills the region
        with generated background content guided by the prompt.

        Args:
            frames: Video frames array of shape (N, H, W, C) in RGB format.
            track: ObjectTrack containing masks defining regions to inpaint.
            prompt: Text prompt to guide inpainting. Default "background, seamless".
            progress_callback: Optional callback for progress updates.
                Called with (stage_name, progress_fraction).

        Returns:
            Inpainted video frames array of shape (N, H, W, C).

        Example:
            >>> # Remove a person and fill with background
            >>> inpainted = inpainter.inpaint(
            ...     frames, person_track, prompt="empty park background"
            ... )
        """
        if self.backend == "replicate":
            return self._inpaint_replicate(
                frames=frames,
                track=track,
                prompt=prompt,
                progress_callback=progress_callback,
            )

        if progress_callback:
            progress_callback("Initializing inpainting model", 0.0)

        num_frames = frames.shape[0]
        inpainted_frames = []

        for i, frame in enumerate(tqdm(frames, desc="Inpainting frames")):
            if progress_callback:
                progress_callback("Inpainting frames", i / num_frames)

            mask_obj = track.get_mask_for_frame(i)
            if mask_obj is None or mask_obj.area == 0:
                # No mask for this frame, keep original
                inpainted_frames.append(frame)
                continue

            mask = mask_obj.mask

            # Dilate mask for cleaner edges
            if self.config.mask_dilation > 0:
                mask = self._dilate_mask(mask, self.config.mask_dilation)

            # Inpaint frame
            inpainted = self._inpaint_frame_local(frame, mask, prompt)
            inpainted_frames.append(inpainted)

        if progress_callback:
            progress_callback("Inpainting complete", 1.0)

        return np.stack(inpainted_frames, axis=0)

    def inpaint_with_temporal_consistency(
        self,
        frames: np.ndarray,
        track: ObjectTrack,
        prompt: str = "background, seamless",
        window_size: int = 3,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> np.ndarray:
        """Inpaint video frames with temporal consistency blending.

        Similar to inpaint() but applies temporal blending to reduce flickering
        between frames. Uses a sliding window average for smoother results.

        Args:
            frames: Video frames array of shape (N, H, W, C) in RGB format.
            track: ObjectTrack containing masks defining regions to inpaint.
            prompt: Text prompt to guide inpainting.
            window_size: Size of temporal window for blending. Default 3.
            progress_callback: Optional callback for progress updates.

        Returns:
            Inpainted video frames with temporal consistency.
        """
        # First do standard inpainting
        inpainted = self.inpaint(
            frames=frames,
            track=track,
            prompt=prompt,
            progress_callback=progress_callback,
        )

        if progress_callback:
            progress_callback("Applying temporal consistency", 0.9)

        # Apply temporal blending in masked regions only
        num_frames = inpainted.shape[0]
        blended = inpainted.copy()
        half_window = window_size // 2

        for i in range(num_frames):
            mask_obj = track.get_mask_for_frame(i)
            if mask_obj is None or mask_obj.area == 0:
                continue

            mask = mask_obj.mask

            # Get window of frames
            start_idx = max(0, i - half_window)
            end_idx = min(num_frames, i + half_window + 1)
            window_frames = inpainted[start_idx:end_idx]

            # Average in masked region
            averaged = np.mean(window_frames, axis=0).astype(np.uint8)

            # Blend only in masked region
            mask_3d = mask[:, :, np.newaxis]
            blended[i] = np.where(mask_3d, averaged, blended[i])

        if progress_callback:
            progress_callback("Temporal consistency applied", 1.0)

        return blended

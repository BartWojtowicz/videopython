"""Image understanding with multi-backend support."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from videopython.ai.backends import ImageToTextBackend, UnsupportedBackendError, get_api_key
from videopython.ai.config import get_default_backend
from videopython.ai.understanding.color import ColorAnalyzer
from videopython.base.description import FrameDescription, SceneDescription
from videopython.base.video import Video


class ImageToText:
    """Generates text descriptions of images."""

    SUPPORTED_BACKENDS: list[str] = ["local", "openai", "gemini"]

    def __init__(
        self,
        backend: ImageToTextBackend | None = None,
        device: str | None = None,
        num_dominant_colors: int = 5,
        api_key: str | None = None,
    ):
        """Initialize image-to-text model.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            device: Device for local backend ('cuda' or 'cpu').
            num_dominant_colors: Number of dominant colors for color analysis.
            api_key: API key for cloud backends. If None, reads from environment.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("image_to_text")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: ImageToTextBackend = resolved_backend  # type: ignore[assignment]
        self.device = device
        self.api_key = api_key
        self.color_analyzer = ColorAnalyzer(num_dominant_colors=num_dominant_colors)

        self._processor: Any = None
        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local BLIP model."""
        import torch
        from transformers.models.blip import BlipForConditionalGeneration, BlipProcessor

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                # MPS is slower than CPU for BLIP (~2x slower in benchmarks),
                # so we default to CPU for best performance
                device = "cpu"

        model_name = "Salesforce/blip-image-captioning-large"
        self._processor = BlipProcessor.from_pretrained(model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(model_name)
        self._model.to(device)

        self.device = device

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _describe_local(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None,
    ) -> str:
        """Generate description using local BLIP model."""
        if self._model is None:
            self._init_local()

        # Convert numpy array to PIL Image if needed
        pil_image = image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)

        inputs = self._processor(pil_image, prompt, return_tensors="pt").to(self.device)
        output = self._model.generate(**inputs, max_new_tokens=50)
        return self._processor.decode(output[0], skip_special_tokens=True)

    def _describe_openai(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None,
    ) -> str:
        """Generate description using OpenAI GPT-4o."""
        from openai import OpenAI

        api_key = get_api_key("openai", self.api_key)
        client = OpenAI(api_key=api_key)

        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_base64 = self._image_to_base64(image)

        system_prompt = "You are an image analysis assistant. Describe images concisely."
        user_prompt = prompt or "Describe this image in 1-2 sentences."

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                        },
                    ],
                },
            ],
            max_tokens=100,
        )

        return response.choices[0].message.content or ""

    def _describe_gemini(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None,
    ) -> str:
        """Generate description using Google Gemini."""
        import google.generativeai as genai

        api_key = get_api_key("gemini", self.api_key)
        genai.configure(api_key=api_key)

        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        model = genai.GenerativeModel("gemini-2.0-flash")
        user_prompt = prompt or "Describe this image in 1-2 sentences."

        response = model.generate_content([user_prompt, image])
        return response.text

    def describe_image(
        self,
        image: np.ndarray | Image.Image,
        prompt: str | None = None,
    ) -> str:
        """Generate a text description of an image.

        Args:
            image: Image as numpy array (H, W, 3) in RGB format or PIL Image.
            prompt: Optional text prompt to guide the description.

        Returns:
            Text description of the image.
        """
        if self.backend == "local":
            return self._describe_local(image, prompt)
        elif self.backend == "openai":
            return self._describe_openai(image, prompt)
        elif self.backend == "gemini":
            return self._describe_gemini(image, prompt)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)

    def describe_frame(
        self,
        video: Video,
        frame_index: int,
        prompt: str | None = None,
        extract_colors: bool = False,
        include_full_histogram: bool = False,
    ) -> FrameDescription:
        """Describe a specific frame from a video.

        Args:
            video: Video object.
            frame_index: Index of the frame to describe.
            prompt: Optional text prompt to guide the description.
            extract_colors: Whether to extract color features from the frame.
            include_full_histogram: Whether to include full HSV histogram.

        Returns:
            FrameDescription object with the frame description.
        """
        if frame_index < 0 or frame_index >= len(video.frames):
            raise ValueError(f"frame_index {frame_index} out of bounds for video with {len(video.frames)} frames")

        frame = video.frames[frame_index]
        description = self.describe_image(frame, prompt)
        timestamp = frame_index / video.fps

        color_histogram = None
        if extract_colors:
            color_histogram = self.color_analyzer.extract_color_features(frame, include_full_histogram)

        return FrameDescription(
            frame_index=frame_index,
            timestamp=timestamp,
            description=description,
            color_histogram=color_histogram,
        )

    def describe_frames(
        self,
        video: Video,
        frame_indices: list[int],
        prompt: str | None = None,
        extract_colors: bool = False,
        include_full_histogram: bool = False,
    ) -> list[FrameDescription]:
        """Describe multiple frames from a video.

        Args:
            video: Video object.
            frame_indices: List of frame indices to describe.
            prompt: Optional text prompt to guide the descriptions.
            extract_colors: Whether to extract color features.
            include_full_histogram: Whether to include full HSV histogram.

        Returns:
            List of FrameDescription objects.
        """
        # Process frames sequentially (thread parallelism causes issues with model initialization)
        return [
            self.describe_frame(video, idx, prompt, extract_colors, include_full_histogram) for idx in frame_indices
        ]

    def describe_scene(
        self,
        video: Video,
        scene: SceneDescription,
        frames_per_second: float = 1.0,
        prompt: str | None = None,
        extract_colors: bool = False,
        include_full_histogram: bool = False,
    ) -> list[FrameDescription]:
        """Describe frames from a scene, sampling at the specified rate.

        Args:
            video: Video object.
            scene: SceneDescription to analyze.
            frames_per_second: Frame sampling rate.
            prompt: Optional text prompt to guide the descriptions.
            extract_colors: Whether to extract color features.
            include_full_histogram: Whether to include full HSV histogram.

        Returns:
            List of FrameDescription objects for the sampled frames.
        """
        if frames_per_second <= 0:
            raise ValueError("frames_per_second must be positive")

        frame_interval = max(1, int(video.fps / frames_per_second))
        frame_indices = list(range(scene.start_frame, scene.end_frame, frame_interval))

        if not frame_indices:
            frame_indices = [scene.start_frame]

        return self.describe_frames(video, frame_indices, prompt, extract_colors, include_full_histogram)

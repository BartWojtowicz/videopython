from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from videopython.base.description import FrameDescription, Scene
from videopython.base.video import Video

IMAGE_TO_TEXT_MODEL = "Salesforce/blip-image-captioning-large"


class ImageToText:
    """Generates text descriptions of images using BLIP image captioning model."""

    def __init__(self, device: str | None = None):
        """Initialize the image-to-text model.

        Args:
            device: Device to run the model on ('cuda', 'cpu', or None for auto-detection)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.processor = BlipProcessor.from_pretrained(IMAGE_TO_TEXT_MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(IMAGE_TO_TEXT_MODEL)
        self.model.to(self.device)

    def describe_image(self, image: np.ndarray | Image.Image, prompt: str | None = None) -> str:
        """Generate a text description of an image.

        Args:
            image: Image as numpy array (H, W, 3) in RGB format or PIL Image
            prompt: Optional text prompt to guide the description

        Returns:
            Text description of the image
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Process the image
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)

        # Generate description
        output = self.model.generate(**inputs, max_new_tokens=50)
        description = self.processor.decode(output[0], skip_special_tokens=True)

        return description

    def describe_frame(self, video: Video, frame_index: int, prompt: str | None = None) -> FrameDescription:
        """Describe a specific frame from a video.

        Args:
            video: Video object
            frame_index: Index of the frame to describe
            prompt: Optional text prompt to guide the description

        Returns:
            FrameDescription object with the frame description
        """
        if frame_index < 0 or frame_index >= len(video.frames):
            raise ValueError(f"frame_index {frame_index} out of bounds for video with {len(video.frames)} frames")

        frame = video.frames[frame_index]
        description = self.describe_image(frame, prompt)
        timestamp = frame_index / video.fps

        return FrameDescription(frame_index=frame_index, timestamp=timestamp, description=description)

    def describe_frames(
        self, video: Video, frame_indices: list[int], prompt: str | None = None
    ) -> list[FrameDescription]:
        """Describe multiple frames from a video.

        Args:
            video: Video object
            frame_indices: List of frame indices to describe
            prompt: Optional text prompt to guide the descriptions

        Returns:
            List of FrameDescription objects
        """
        descriptions = []
        for frame_index in frame_indices:
            description = self.describe_frame(video, frame_index, prompt)
            descriptions.append(description)

        return descriptions

    def describe_scene(
        self, video: Video, scene: Scene, frames_per_second: float = 1.0, prompt: str | None = None
    ) -> list[FrameDescription]:
        """Describe frames from a scene, sampling at the specified rate.

        Args:
            video: Video object
            scene: Scene to analyze
            frames_per_second: Frame sampling rate (default: 1.0 fps)
            prompt: Optional text prompt to guide the descriptions

        Returns:
            List of FrameDescription objects for the sampled frames
        """
        if frames_per_second <= 0:
            raise ValueError("frames_per_second must be positive")

        # Calculate frame interval based on desired fps
        frame_interval = max(1, int(video.fps / frames_per_second))

        # Sample frames from the scene
        frame_indices = list(range(scene.start_frame, scene.end_frame, frame_interval))

        # Ensure we don't have an empty list
        if not frame_indices:
            frame_indices = [scene.start_frame]

        return self.describe_frames(video, frame_indices, prompt)

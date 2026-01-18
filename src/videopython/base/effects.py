from __future__ import annotations

from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import TYPE_CHECKING, Literal, final

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from videopython.base.video import Video

# Minimum frames before using multiprocessing (Pool overhead isn't worth it below this)
MIN_FRAMES_FOR_MULTIPROCESSING = 100

if TYPE_CHECKING:
    from videopython.base.description import BoundingBox

__all__ = ["Effect", "FullImageOverlay", "Blur", "Zoom", "ColorGrading", "Vignette", "KenBurns"]


class Effect(ABC):
    """Abstract class for effect on frames of video.

    The effect must not change the number of frames and the shape of the frames.
    """

    @final
    def apply(self, video: Video, start: float | None = None, stop: float | None = None) -> Video:
        original_shape = video.video_shape
        start = start if start is not None else 0
        stop = stop if stop is not None else video.total_seconds
        # Check for start and stop correctness
        if not 0 <= start <= video.total_seconds:
            raise ValueError(f"Video is only {video.total_seconds} long, but passed start: {start}!")
        elif not start <= stop <= video.total_seconds:
            raise ValueError(f"Video is only {video.total_seconds} long, but passed stop: {stop}!")
        # Apply effect on video slice
        effect_start_frame = round(start * video.fps)
        effect_end_frame = round(stop * video.fps)
        video_with_effect = self._apply(video[effect_start_frame:effect_end_frame])
        old_audio = video.audio
        video = Video.from_frames(
            np.r_[
                "0,2",
                video.frames[:effect_start_frame],
                video_with_effect.frames,
                video.frames[effect_end_frame:],
            ],
            fps=video.fps,
        )
        video.audio = old_audio
        # Check if dimensions didn't change
        if not video.video_shape == original_shape:
            raise RuntimeError("The effect must not change the number of frames and the shape of the frames!")

        return video

    @abstractmethod
    def _apply(self, video: Video) -> Video:
        pass


class FullImageOverlay(Effect):
    """Overlays an image on top of video frames with optional transparency and fade."""

    def __init__(self, overlay_image: np.ndarray, alpha: float | None = None, fade_time: float = 0.0):
        """Initialize image overlay effect.

        Args:
            overlay_image: RGB or RGBA image to overlay, must match video dimensions.
            alpha: Overall opacity from 0 (transparent) to 1 (opaque), defaults to 1.0.
            fade_time: Duration in seconds for fade in/out at start and end.
        """
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("Alpha must be in range [0, 1]!")
        elif not (overlay_image.ndim == 3 and overlay_image.shape[-1] in [3, 4]):
            raise ValueError("Only RGB and RGBA images are supported as an overlay!")
        elif alpha is None:
            alpha = 1.0

        if overlay_image.shape[-1] == 3:
            overlay_image = np.dstack([overlay_image, np.full(overlay_image.shape[:2], 255, dtype=np.uint8)])

        self.alpha = alpha
        self.overlay = overlay_image.astype(np.uint8)
        self.fade_time = fade_time

    def _overlay(self, img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        img_pil = Image.fromarray(img)
        overlay = self.overlay.copy()
        overlay[:, :, 3] = overlay[:, :, 3] * (self.alpha * alpha)
        overlay_pil = Image.fromarray(overlay)
        img_pil.paste(overlay_pil, (0, 0), overlay_pil)
        return np.array(img_pil)

    def _apply(self, video: Video) -> Video:
        if not video.frame_shape == self.overlay[:, :, :3].shape:
            raise ValueError(
                f"Mismatch of overlay shape `{self.overlay.shape}` with video shape: `{video.frame_shape}`!"
            )
        elif not (0 <= 2 * self.fade_time <= video.total_seconds):
            raise ValueError(f"Video is only {video.total_seconds}s long, but fade time is {self.fade_time}s!")

        print("Overlaying video...")
        if self.fade_time == 0:
            video.frames = np.array([self._overlay(frame) for frame in tqdm(video.frames)], dtype=np.uint8)
        else:
            num_video_frames = len(video.frames)
            num_fade_frames = round(self.fade_time * video.fps)
            new_frames = []
            for i, frame in enumerate(tqdm(video.frames)):
                frames_dist_from_end = min(i, num_video_frames - i)
                if frames_dist_from_end >= num_fade_frames:
                    fade_alpha = 1.0
                else:
                    fade_alpha = frames_dist_from_end / num_fade_frames
                new_frames.append(self._overlay(frame, fade_alpha))
            video.frames = np.array(new_frames, dtype=np.uint8)
        return video


class Blur(Effect):
    """Applies Gaussian blur with constant, ascending, or descending intensity."""

    def __init__(
        self,
        mode: Literal["constant", "ascending", "descending"],
        iterations: int,
        kernel_size: tuple[int, int] = (5, 5),
    ):
        """Initialize blur effect.

        Args:
            mode: Blur mode - "constant" (same blur), "ascending" (increasing blur), or "descending" (decreasing blur).
            iterations: Number of blur iterations to apply.
            kernel_size: Gaussian kernel size for blur operation.
        """
        if iterations < 1:
            raise ValueError("Iterations must be at least 1!")
        self.mode = mode
        self.iterations = iterations
        self.kernel_size = kernel_size

    def _blur_frame(self, frame: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian blur to a single frame.

        Args:
            frame: Frame to blur.
            sigma: Gaussian sigma value.

        Returns:
            Blurred frame.
        """
        return cv2.GaussianBlur(frame, self.kernel_size, sigma)

    def _apply(self, video: Video) -> Video:
        n_frames = len(video.frames)

        # Calculate base sigma from kernel size (OpenCV formula)
        base_sigma = 0.3 * ((self.kernel_size[0] - 1) * 0.5 - 1) + 0.8

        # Multiple blur iterations with sigma S approximate single blur with sigma S*sqrt(iterations)
        # This is much faster than iterative application
        max_sigma = base_sigma * np.sqrt(self.iterations)

        # Calculate sigma for each frame based on mode
        if self.mode == "constant":
            sigmas = np.full(n_frames, max_sigma)
        elif self.mode == "ascending":
            # Linearly increase blur intensity from start to end
            iteration_ratios = np.linspace(1 / n_frames, 1.0, n_frames)
            sigmas = base_sigma * np.sqrt(np.maximum(1, np.round(iteration_ratios * self.iterations)))
        elif self.mode == "descending":
            # Linearly decrease blur intensity from start to end
            iteration_ratios = np.linspace(1.0, 1 / n_frames, n_frames)
            sigmas = base_sigma * np.sqrt(np.maximum(1, np.round(iteration_ratios * self.iterations)))
        else:
            raise ValueError(f"Unknown mode: `{self.mode}`.")

        print(f"Applying {self.mode} blur...")

        if n_frames >= MIN_FRAMES_FOR_MULTIPROCESSING:
            with Pool() as pool:
                new_frames = pool.starmap(
                    self._blur_frame,
                    [(frame, sigma) for frame, sigma in zip(video.frames, sigmas)],
                )
        else:
            new_frames = [self._blur_frame(frame, sigma) for frame, sigma in zip(video.frames, sigmas)]

        video.frames = np.array(new_frames, dtype=np.uint8)
        return video


class Zoom(Effect):
    """Applies zoom in or out effect by cropping and scaling frames progressively."""

    def __init__(self, zoom_factor: float, mode: Literal["in", "out"]):
        """Initialize zoom effect.

        Args:
            zoom_factor: Maximum zoom level, must be greater than 1.
            mode: Zoom direction - "in" for zoom in effect, "out" for zoom out effect.
        """
        if zoom_factor <= 1:
            raise ValueError("Zoom factor must be greater than 1!")
        self.zoom_factor = zoom_factor
        self.mode = mode

    def _apply(self, video: Video) -> Video:
        n_frames = len(video.frames)
        new_frames = []

        width = video.metadata.width
        height = video.metadata.height
        crop_sizes_w, crop_sizes_h = (
            np.linspace(width // self.zoom_factor, width, n_frames),
            np.linspace(height // self.zoom_factor, height, n_frames),
        )

        if self.mode == "in":
            for frame, w, h in tqdm(zip(video.frames, reversed(crop_sizes_w), reversed(crop_sizes_h))):
                x = width / 2 - w / 2
                y = height / 2 - h / 2

                cropped_frame = frame[round(y) : round(y + h), round(x) : round(x + w)]
                zoomed_frame = cv2.resize(cropped_frame, (width, height))
                new_frames.append(zoomed_frame)
        elif self.mode == "out":
            for frame, w, h in tqdm(zip(video.frames, crop_sizes_w, crop_sizes_h)):
                x = width / 2 - w / 2
                y = height / 2 - h / 2

                cropped_frame = frame[round(y) : round(y + h), round(x) : round(x + w)]
                zoomed_frame = cv2.resize(cropped_frame, (width, height))
                new_frames.append(zoomed_frame)
        else:
            raise ValueError(f"Unknown mode: `{self.mode}`.")
        video.frames = np.asarray(new_frames)
        return video


class ColorGrading(Effect):
    """Adjusts color properties: brightness, contrast, saturation, and temperature."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        temperature: float = 0.0,
    ):
        """Initialize color grading effect.

        Args:
            brightness: Brightness adjustment (-1.0 to 1.0, default 0).
            contrast: Contrast multiplier (0.5 to 2.0, default 1.0).
            saturation: Saturation multiplier (0.0 to 2.0, default 1.0).
            temperature: Color temperature shift (-1.0=cooler/blue to 1.0=warmer/orange, default 0).
        """
        if not -1.0 <= brightness <= 1.0:
            raise ValueError("Brightness must be between -1.0 and 1.0!")
        if not 0.5 <= contrast <= 2.0:
            raise ValueError("Contrast must be between 0.5 and 2.0!")
        if not 0.0 <= saturation <= 2.0:
            raise ValueError("Saturation must be between 0.0 and 2.0!")
        if not -1.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between -1.0 and 1.0!")

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.temperature = temperature

    def _grade_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply color grading to a single frame."""
        # Convert to float for processing
        img = frame.astype(np.float32) / 255.0

        # Apply brightness
        if self.brightness != 0:
            img = img + self.brightness

        # Apply contrast (around midpoint 0.5)
        if self.contrast != 1.0:
            img = (img - 0.5) * self.contrast + 0.5

        # Apply saturation in HSV space
        if self.saturation != 1.0:
            hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * self.saturation
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float64)

        # Apply temperature (shift red/blue channels)
        if self.temperature != 0:
            # Warm = more red/yellow, less blue
            # Cool = more blue, less red/yellow
            temp_shift = self.temperature * 0.1
            img[:, :, 0] = img[:, :, 0] + temp_shift  # Red
            img[:, :, 2] = img[:, :, 2] - temp_shift  # Blue

        # Clip and convert back to uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return img

    def _apply(self, video: Video) -> Video:
        print("Applying color grading...")
        n_frames = len(video.frames)

        if n_frames >= MIN_FRAMES_FOR_MULTIPROCESSING:
            with Pool() as pool:
                new_frames = pool.map(self._grade_frame, video.frames)
        else:
            new_frames = [self._grade_frame(frame) for frame in video.frames]

        video.frames = np.array(new_frames, dtype=np.uint8)
        return video


class Vignette(Effect):
    """Applies a vignette effect (darkening at edges)."""

    def __init__(self, strength: float = 0.5, radius: float = 1.0):
        """Initialize vignette effect.

        Args:
            strength: How dark the edges become (0.0 to 1.0, default 0.5).
            radius: How far the vignette extends from center (0.5 to 2.0, default 1.0).
        """
        if not 0.0 <= strength <= 1.0:
            raise ValueError("Strength must be between 0.0 and 1.0!")
        if not 0.5 <= radius <= 2.0:
            raise ValueError("Radius must be between 0.5 and 2.0!")

        self.strength = strength
        self.radius = radius
        self._mask: np.ndarray | None = None

    def _create_mask(self, height: int, width: int) -> np.ndarray:
        """Create vignette mask for given dimensions."""
        # Create coordinate grids
        y = np.linspace(-1, 1, height)
        x = np.linspace(-1, 1, width)
        X, Y = np.meshgrid(x, y)

        # Calculate distance from center
        distance = np.sqrt(X**2 + Y**2) / self.radius

        # Create smooth falloff
        mask = 1.0 - np.clip(distance - 0.5, 0, 1) * 2 * self.strength

        return mask.astype(np.float32)

    def _apply(self, video: Video) -> Video:
        print("Applying vignette effect...")
        height, width = video.frame_shape[:2]

        # Create mask once for the video dimensions
        if self._mask is None or self._mask.shape != (height, width):
            self._mask = self._create_mask(height, width)

        # Apply mask to all frames using vectorized operation
        # mask_3d shape: (height, width, 1), frames shape: (n_frames, height, width, 3)
        # Broadcasting handles the multiplication across all frames and channels
        mask_3d = self._mask[:, :, np.newaxis]
        video.frames = (video.frames.astype(np.float32) * mask_3d).astype(np.uint8)
        return video


class KenBurns(Effect):
    """Cinematic pan-and-zoom effect that animates between two regions.

    Named after documentarian Ken Burns who popularized the technique. The effect
    smoothly transitions from a start region to an end region over the duration
    of the video, creating dynamic movement from static content.
    """

    def __init__(
        self,
        start_region: "BoundingBox",
        end_region: "BoundingBox",
        easing: Literal["linear", "ease_in", "ease_out", "ease_in_out"] = "linear",
    ):
        """Initialize Ken Burns effect.

        Args:
            start_region: Starting crop region (normalized 0-1 coordinates).
            end_region: Ending crop region (normalized 0-1 coordinates).
            easing: Animation easing function - "linear", "ease_in", "ease_out", or "ease_in_out".
        """
        from videopython.base.description import BoundingBox

        if not isinstance(start_region, BoundingBox) or not isinstance(end_region, BoundingBox):
            raise TypeError("start_region and end_region must be BoundingBox instances!")

        # Validate regions are within bounds
        for name, region in [("start_region", start_region), ("end_region", end_region)]:
            if not (0 <= region.x <= 1 and 0 <= region.y <= 1):
                raise ValueError(f"{name} position must be in range [0, 1]!")
            if not (0 < region.width <= 1 and 0 < region.height <= 1):
                raise ValueError(f"{name} dimensions must be in range (0, 1]!")
            if region.x + region.width > 1 or region.y + region.height > 1:
                raise ValueError(f"{name} extends beyond image bounds!")

        if easing not in ("linear", "ease_in", "ease_out", "ease_in_out"):
            raise ValueError(f"Unknown easing function: {easing}!")

        self.start_region = start_region
        self.end_region = end_region
        self.easing = easing

    def _ease(self, t: float) -> float:
        """Apply easing function to normalized time value.

        Args:
            t: Normalized time value in range [0, 1].

        Returns:
            Eased value in range [0, 1].
        """
        if self.easing == "linear":
            return t
        elif self.easing == "ease_in":
            return t * t
        elif self.easing == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif self.easing == "ease_in_out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)
        return t

    def _crop_and_scale_frame(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        crop_w: int,
        crop_h: int,
        target_w: int,
        target_h: int,
    ) -> np.ndarray:
        """Crop region from frame and scale to target size."""
        cropped = frame[y : y + crop_h, x : x + crop_w]
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _apply(self, video: Video) -> Video:
        n_frames = len(video.frames)
        height, width = video.frame_shape[:2]
        target_h, target_w = height, width

        # Convert normalized coordinates to pixel values
        start_x = int(self.start_region.x * width)
        start_y = int(self.start_region.y * height)
        start_w = int(self.start_region.width * width)
        start_h = int(self.start_region.height * height)

        end_x = int(self.end_region.x * width)
        end_y = int(self.end_region.y * height)
        end_w = int(self.end_region.width * width)
        end_h = int(self.end_region.height * height)

        print("Applying Ken Burns effect...")
        new_frames = []
        for i, frame in enumerate(tqdm(video.frames)):
            t = i / max(1, n_frames - 1)  # Normalized time [0, 1]
            eased_t = self._ease(t)

            # Interpolate region parameters
            x = int(start_x + (end_x - start_x) * eased_t)
            y = int(start_y + (end_y - start_y) * eased_t)
            crop_w = int(start_w + (end_w - start_w) * eased_t)
            crop_h = int(start_h + (end_h - start_h) * eased_t)

            # Ensure crop region stays within bounds
            x = max(0, min(x, width - crop_w))
            y = max(0, min(y, height - crop_h))

            new_frame = self._crop_and_scale_frame(frame, x, y, crop_w, crop_h, target_w, target_h)
            new_frames.append(new_frame)

        video.frames = np.array(new_frames, dtype=np.uint8)
        return video

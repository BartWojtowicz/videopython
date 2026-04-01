from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from videopython.base.progress import log, progress_iter
from videopython.base.video import Video

if TYPE_CHECKING:
    from videopython.base.description import BoundingBox

__all__ = [
    "Effect",
    "AudioEffect",
    "FullImageOverlay",
    "Blur",
    "Zoom",
    "ColorGrading",
    "Vignette",
    "KenBurns",
    "Fade",
    "VolumeAdjust",
    "TextOverlay",
]


def _resolve_time_range(start: float | None, stop: float | None, total_seconds: float) -> tuple[float, float]:
    """Clamp and validate an effect time range against the video duration.

    Returns resolved (start, stop) in seconds.
    """
    start_s = start if start is not None else 0
    stop_s = stop if stop is not None else total_seconds
    stop_s = min(stop_s, total_seconds)
    start_s = min(start_s, total_seconds)
    if start_s < 0:
        raise ValueError(f"Effect start must be non-negative, got {start_s}!")
    if stop_s < start_s:
        raise ValueError(f"Effect stop ({stop_s}) must be >= start ({start_s})!")
    return start_s, stop_s


class Effect(ABC):
    """Abstract class for effect on frames of video.

    The effect must not change the number of frames and the shape of the frames.
    """

    def apply(self, video: Video, start: float | None = None, stop: float | None = None) -> Video:
        """Apply the effect to a video, optionally within a time range.

        Omit ``start`` to apply from the beginning, omit ``stop`` to apply until
        the end.  Prefer omitting over passing explicit values when the intent is
        full-range application -- this avoids floating-point mismatches with the
        actual video duration.

        Args:
            video: Input video.
            start: Start time in seconds. Omit to apply from the beginning.
                Only set when the effect should begin partway through.
            stop: Stop time in seconds. Omit to apply until the end.
                Only set when the effect should end before the video does.
        """
        original_shape = video.video_shape

        if start is None and stop is None:
            # Full-range: apply directly without slicing or np.r_ reassembly.
            video = self._apply(video)
        else:
            start_s, stop_s = _resolve_time_range(start, stop, video.total_seconds)
            # Apply effect on video slice
            effect_start_frame = round(start_s * video.fps)
            effect_end_frame = round(stop_s * video.fps)
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
        if video.video_shape != original_shape:
            raise RuntimeError("The effect must not change the number of frames and the shape of the frames!")

        return video

    @abstractmethod
    def _apply(self, video: Video) -> Video:
        pass


class FullImageOverlay(Effect):
    """Composites a full-frame image on top of every video frame.

    Useful for watermarks, logos, or static graphic overlays. Supports
    transparency via RGBA images and an overall opacity control.
    """

    def __init__(self, overlay_image: np.ndarray, alpha: float | None = None, fade_time: float = 0.0):
        """Initialize image overlay effect.

        Args:
            overlay_image: RGB or RGBA image array. Must match the video's
                width and height.
            alpha: Overall opacity. 0 = fully transparent, 1 = fully opaque.
                Defaults to 1.0.
            fade_time: Seconds to fade the overlay in at the start and out
                at the end of its time range.
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

        log("Overlaying video...")
        if self.fade_time == 0:
            for i in progress_iter(range(len(video.frames)), desc="Overlaying frames"):
                video.frames[i] = self._overlay(video.frames[i])
        else:
            num_video_frames = len(video.frames)
            num_fade_frames = round(self.fade_time * video.fps)
            for i in progress_iter(range(num_video_frames), desc="Overlaying frames"):
                frames_dist_from_end = min(i, num_video_frames - i)
                fade_alpha = 1.0 if frames_dist_from_end >= num_fade_frames else frames_dist_from_end / num_fade_frames
                video.frames[i] = self._overlay(video.frames[i], fade_alpha)
        return video


class Blur(Effect):
    """Applies Gaussian blur that can stay constant or ramp up/down over the clip."""

    def __init__(
        self,
        mode: Literal["constant", "ascending", "descending"],
        iterations: int,
        kernel_size: tuple[int, int] = (5, 5),
    ):
        """Initialize blur effect.

        Args:
            mode: "constant" applies uniform blur, "ascending" ramps from sharp
                to blurry, "descending" ramps from blurry to sharp.
            iterations: Blur strength. Higher values produce a stronger blur
                (e.g. 5 for subtle, 50+ for heavy).
            kernel_size: Gaussian kernel [width, height] in pixels. Must be odd
                numbers. Larger kernels spread the blur wider.
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

        log(f"Applying {self.mode} blur...")
        for i in progress_iter(range(n_frames), desc="Blurring"):
            video.frames[i] = self._blur_frame(video.frames[i], sigmas[i])
        return video


class Zoom(Effect):
    """Progressively zooms into or out of the frame center over the clip duration."""

    def __init__(self, zoom_factor: float, mode: Literal["in", "out"]):
        """Initialize zoom effect.

        Args:
            zoom_factor: How far to zoom. 1.5 is a subtle push, 2.0 is
                moderate, 3.0+ is dramatic. Must be greater than 1.
            mode: "in" starts wide and pushes into the center,
                "out" starts tight and pulls back.
        """
        if zoom_factor <= 1:
            raise ValueError("Zoom factor must be greater than 1!")
        self.zoom_factor = zoom_factor
        self.mode = mode

    def _apply(self, video: Video) -> Video:
        n_frames = len(video.frames)
        width = video.metadata.width
        height = video.metadata.height
        crop_sizes_w = np.linspace(width // self.zoom_factor, width, n_frames)
        crop_sizes_h = np.linspace(height // self.zoom_factor, height, n_frames)

        if self.mode == "in":
            crop_sizes_w = crop_sizes_w[::-1]
            crop_sizes_h = crop_sizes_h[::-1]
        elif self.mode != "out":
            raise ValueError(f"Unknown mode: `{self.mode}`.")

        for i in progress_iter(range(n_frames), desc="Zooming", total=n_frames):
            w, h = crop_sizes_w[i], crop_sizes_h[i]
            x = width / 2 - w / 2
            y = height / 2 - h / 2
            cropped_frame = video.frames[i][round(y) : round(y + h), round(x) : round(x + w)]
            video.frames[i] = cv2.resize(cropped_frame, (width, height))
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
            brightness: Shift brightness. -1.0 = much darker, 0 = unchanged,
                1.0 = much brighter.
            contrast: Scale contrast. 0.5 = flat/washed out, 1.0 = unchanged,
                2.0 = high contrast.
            saturation: Scale color intensity. 0.0 = grayscale, 1.0 = unchanged,
                2.0 = vivid/oversaturated.
            temperature: Shift color temperature. -1.0 = cool/blue tint,
                0 = neutral, 1.0 = warm/orange tint.
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
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

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
        log("Applying color grading...")
        for i in progress_iter(range(len(video.frames)), desc="Color grading"):
            video.frames[i] = self._grade_frame(video.frames[i])
        return video


class Vignette(Effect):
    """Darkens the edges of the frame, drawing attention to the center."""

    def __init__(self, strength: float = 0.5, radius: float = 1.0):
        """Initialize vignette effect.

        Args:
            strength: Edge darkness amount. 0.0 = no darkening, 0.5 = moderate,
                1.0 = fully black edges.
            radius: Size of the bright center area. Smaller values (0.5) create
                a tight spotlight, larger values (2.0) keep more of the frame lit.
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
        log("Applying vignette effect...")
        height, width = video.frame_shape[:2]

        # Create mask once for the video dimensions
        if self._mask is None or self._mask.shape != (height, width):
            self._mask = self._create_mask(height, width)

        # Apply mask in batches to avoid allocating a full float32 copy of all frames
        mask_3d = self._mask[:, :, np.newaxis]
        batch_size = 64
        for start in range(0, len(video.frames), batch_size):
            end = min(start + batch_size, len(video.frames))
            video.frames[start:end] = (video.frames[start:end].astype(np.float32) * mask_3d).astype(np.uint8)
        return video


class KenBurns(Effect):
    """Cinematic pan-and-zoom that smoothly animates between two crop regions.

    Creates movement by transitioning from a start region to an end region over
    the clip. Use it to add motion to still images or to guide the viewer's eye
    across a scene.
    """

    def __init__(
        self,
        start_region: "BoundingBox",
        end_region: "BoundingBox",
        easing: Literal["linear", "ease_in", "ease_out", "ease_in_out"] = "linear",
    ):
        """Initialize Ken Burns effect.

        Args:
            start_region: Starting crop region as a BoundingBox with normalized
                0-1 coordinates.
            end_region: Ending crop region as a BoundingBox with normalized
                0-1 coordinates.
            easing: Animation curve. "linear" moves at constant speed,
                "ease_in" starts slow, "ease_out" ends slow,
                "ease_in_out" starts and ends slow.
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

        log("Applying Ken Burns effect...")
        for i in progress_iter(range(n_frames), desc="Ken Burns"):
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

            video.frames[i] = self._crop_and_scale_frame(video.frames[i], x, y, crop_w, crop_h, target_w, target_h)
        return video


def _compute_curve(t: np.ndarray, curve: str) -> np.ndarray:
    """Compute alpha values from normalized time array using the given curve type."""
    if curve == "sqrt":
        return np.sqrt(t)
    elif curve == "exponential":
        return t * t
    else:  # linear
        return t


class Fade(Effect):
    """Fades video and audio to or from black."""

    def __init__(
        self,
        mode: Literal["in", "out", "in_out"],
        duration: float = 1.0,
        curve: Literal["sqrt", "linear", "exponential"] = "sqrt",
    ):
        """Initialize fade effect.

        Args:
            mode: "in" fades from black at the start, "out" fades to black
                at the end, "in_out" does both.
            duration: Length of each fade in seconds.
            curve: Brightness ramp shape. "sqrt" feels perceptually even
                (recommended), "linear" is mathematically even, "exponential"
                starts slow and finishes fast.
        """
        if mode not in ("in", "out", "in_out"):
            raise ValueError(f"mode must be 'in', 'out', or 'in_out', got '{mode}'")
        if duration <= 0:
            raise ValueError(f"duration must be > 0, got {duration}")
        self.mode = mode
        self.duration = duration
        self.curve = curve

    def apply(self, video: Video, start: float | None = None, stop: float | None = None) -> Video:
        """Apply fade effect to video and audio.

        Omit ``start`` to apply from the beginning, omit ``stop`` to apply
        until the end. Prefer omitting over passing explicit values when
        the intent is full-range application.

        Args:
            video: Input video.
            start: Start time in seconds. Omit to apply from the beginning.
                Only set when the effect should begin partway through.
            stop: Stop time in seconds. Omit to apply until the end.
                Only set when the effect should end before the video does.
        """
        original_shape = video.video_shape
        start_s, stop_s = _resolve_time_range(start, stop, video.total_seconds)

        effect_start_frame = round(start_s * video.fps)
        effect_end_frame = round(stop_s * video.fps)
        n_effect_frames = effect_end_frame - effect_start_frame
        fade_frames = min(round(self.duration * video.fps), n_effect_frames)

        # Build per-frame alpha array (1.0 = fully visible, 0.0 = black)
        alpha = np.ones(n_effect_frames, dtype=np.float32)
        if self.mode in ("in", "in_out"):
            t = np.linspace(0, 1, fade_frames, dtype=np.float32)
            alpha[:fade_frames] = _compute_curve(t, self.curve)
        if self.mode in ("out", "in_out"):
            t = np.linspace(1, 0, fade_frames, dtype=np.float32)
            alpha[-fade_frames:] = np.minimum(alpha[-fade_frames:], _compute_curve(t, self.curve))

        # Apply to video frames in batches to avoid a full float32 copy
        batch_size = 64
        for batch_start in range(0, n_effect_frames, batch_size):
            batch_end = min(batch_start + batch_size, n_effect_frames)
            batch_alpha = alpha[batch_start:batch_end, np.newaxis, np.newaxis, np.newaxis]
            # Skip batch if all alphas are 1.0 (no change needed)
            if np.all(batch_alpha == 1.0):
                continue
            abs_start = effect_start_frame + batch_start
            abs_end = effect_start_frame + batch_end
            video.frames[abs_start:abs_end] = (video.frames[abs_start:abs_end].astype(np.float32) * batch_alpha).astype(
                np.uint8
            )

        # Verify shape invariant
        if video.video_shape != original_shape:
            raise RuntimeError("The effect must not change the number of frames and the shape of the frames!")

        # Apply to audio
        if video.audio is not None and not video.audio.is_silent:
            sample_rate = video.audio.metadata.sample_rate
            audio_start = round(start_s * sample_rate)
            audio_end = min(round(stop_s * sample_rate), len(video.audio.data))
            n_audio_samples = audio_end - audio_start
            fade_samples = min(round(self.duration * sample_rate), n_audio_samples)

            audio_alpha = np.ones(n_audio_samples, dtype=np.float32)
            if self.mode in ("in", "in_out"):
                t = np.linspace(0, 1, fade_samples, dtype=np.float32)
                audio_alpha[:fade_samples] = _compute_curve(t, self.curve)
            if self.mode in ("out", "in_out"):
                t = np.linspace(1, 0, fade_samples, dtype=np.float32)
                audio_alpha[-fade_samples:] = np.minimum(audio_alpha[-fade_samples:], _compute_curve(t, self.curve))

            audio_data = video.audio.data
            if audio_data.ndim == 1:
                audio_data[audio_start:audio_end] *= audio_alpha
            else:
                audio_data[audio_start:audio_end] *= audio_alpha[:, np.newaxis]
            np.clip(audio_data, -1.0, 1.0, out=audio_data)

        return video

    def _apply(self, video: Video) -> Video:
        raise NotImplementedError("Fade overrides apply() directly")


class AudioEffect(Effect):
    """Abstract base class for audio-only effects.

    Inherits from Effect so isinstance checks in the execution engine pass
    without modification. Overrides apply() to skip frame processing.
    """

    def _apply(self, video: Video) -> Video:
        raise NotImplementedError("AudioEffect does not process frames -- use _apply_audio()")

    def apply(self, video: Video, start: float | None = None, stop: float | None = None) -> Video:
        """Apply the audio effect to a video, optionally within a time range.

        Omit ``start`` to apply from the beginning, omit ``stop`` to apply until
        the end.  Prefer omitting over passing explicit values when the intent is
        full-range application -- this avoids floating-point mismatches with the
        actual video duration.

        Args:
            video: Input video.
            start: Start time in seconds. Omit to apply from the beginning.
                Only set when the effect should begin partway through.
            stop: Stop time in seconds. Omit to apply until the end.
                Only set when the effect should end before the video does.
        """
        start_s, stop_s = _resolve_time_range(start, stop, video.total_seconds)
        video.audio = self._apply_audio(video.audio, start_s, stop_s, video.fps)
        return video

    @abstractmethod
    def _apply_audio(self, audio, start: float, stop: float, fps: float):
        pass


class VolumeAdjust(AudioEffect):
    """Changes audio volume within a time range without affecting video frames."""

    def __init__(self, volume: float = 1.0, ramp_duration: float = 0.0):
        """Initialize volume adjustment effect.

        Args:
            volume: Volume multiplier. 0.0 = silence, 1.0 = original level,
                2.0 = twice as loud (may clip).
            ramp_duration: Seconds to smoothly ramp volume at the start and end
                of the window, preventing audible clicks.
        """
        if volume < 0:
            raise ValueError(f"volume must be >= 0, got {volume}")
        if ramp_duration < 0:
            raise ValueError(f"ramp_duration must be >= 0, got {ramp_duration}")
        self.volume = volume
        self.ramp_duration = ramp_duration

    def _apply_audio(self, audio, start: float, stop: float, fps: float):
        if audio is None or audio.is_silent:
            return audio

        sample_rate = audio.metadata.sample_rate
        start_sample = round(start * sample_rate)
        end_sample = min(round(stop * sample_rate), len(audio.data))
        n_samples = end_sample - start_sample

        # Build volume envelope
        envelope = np.full(n_samples, self.volume, dtype=np.float32)

        if self.ramp_duration > 0:
            ramp_samples = min(round(self.ramp_duration * sample_rate), n_samples // 2)
            if ramp_samples > 0:
                # Ramp from 1.0 to target volume at start
                t = np.linspace(0, 1, ramp_samples, dtype=np.float32)
                ramp_in = 1.0 + (self.volume - 1.0) * np.sqrt(t)
                envelope[:ramp_samples] = ramp_in

                # Ramp from target volume back to 1.0 at end
                t = np.linspace(1, 0, ramp_samples, dtype=np.float32)
                ramp_out = 1.0 + (self.volume - 1.0) * np.sqrt(t)
                envelope[-ramp_samples:] = ramp_out

        if audio.data.ndim == 1:
            audio.data[start_sample:end_sample] *= envelope
        else:
            audio.data[start_sample:end_sample] *= envelope[:, np.newaxis]
        np.clip(audio.data, -1.0, 1.0, out=audio.data)

        return audio


class TextOverlay(Effect):
    """Draws text on video frames, with auto word-wrap and optional background box."""

    def __init__(
        self,
        text: str,
        position: tuple[float, float] = (0.5, 0.9),
        font_size: int = 48,
        text_color: tuple[int, int, int] = (255, 255, 255),
        background_color: tuple[int, int, int, int] | None = (0, 0, 0, 160),
        background_padding: int = 12,
        max_width: float = 0.8,
        anchor: Literal["center", "top_left", "top_center", "bottom_center", "bottom_left", "bottom_right"] = "center",
        font_filename: str | None = None,
    ):
        """Initialize text overlay effect.

        Args:
            text: The string to display. Use \\n for line breaks.
            position: Where to place the text as normalized (x, y) coordinates.
                (0, 0) = top-left corner, (1, 1) = bottom-right corner.
            font_size: Font size in pixels.
            text_color: Text color as [R, G, B], each 0-255.
            background_color: Background box color as [R, G, B, A] (0-255),
                or null to disable the background.
            background_padding: Padding in pixels between text and background edge.
            max_width: Maximum text width as a fraction of frame width (0-1).
                Text longer than this wraps to the next line.
            anchor: Which point of the text box sits at the position coordinate.
            font_filename: Path to a .ttf font file, or None for the default font.
        """
        if not text:
            raise ValueError("text must not be empty")
        if not 0.0 <= position[0] <= 1.0 or not 0.0 <= position[1] <= 1.0:
            raise ValueError("position values must be in range [0, 1]")
        if font_size < 1:
            raise ValueError(f"font_size must be >= 1, got {font_size}")
        if not 0.0 < max_width <= 1.0:
            raise ValueError(f"max_width must be in range (0, 1], got {max_width}")

        self.text = text
        self.position = position
        self.font_size = font_size
        self.text_color = text_color
        self.background_color = background_color
        self.background_padding = background_padding
        self.max_width = max_width
        self.anchor = anchor
        self.font_filename = font_filename
        self._rendered: np.ndarray | None = None

    def _get_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if self.font_filename:
            return ImageFont.truetype(self.font_filename, self.font_size)
        try:
            return ImageFont.truetype("DejaVuSans.ttf", self.font_size)
        except OSError:
            return ImageFont.load_default()

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_px: int) -> str:
        """Word-wrap text to fit within max_px width."""
        lines: list[str] = []
        for paragraph in text.split("\n"):
            words = paragraph.split()
            if not words:
                lines.append("")
                continue
            current = words[0]
            for word in words[1:]:
                test = current + " " + word
                bbox = font.getbbox(test)
                if bbox[2] - bbox[0] <= max_px:
                    current = test
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
        return "\n".join(lines)

    def _render_text_image(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Render text to an RGBA numpy array sized for the given frame dimensions."""
        font = self._get_font()
        max_px = int(self.max_width * frame_width)
        wrapped = self._wrap_text(self.text, font, max_px)

        # Measure text bounds
        temp_img = Image.new("RGBA", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.multiline_textbbox((0, 0), wrapped, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad = self.background_padding
        img_w = text_w + 2 * pad
        img_h = text_h + 2 * pad

        # Create RGBA image
        if self.background_color is not None:
            bg = self.background_color
            img = Image.new("RGBA", (img_w, img_h), bg)
        else:
            img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))

        draw = ImageDraw.Draw(img)
        draw.multiline_text((pad - bbox[0], pad - bbox[1]), wrapped, font=font, fill=(*self.text_color, 255))

        return np.array(img, dtype=np.uint8)

    def _compute_position(self, frame_width: int, frame_height: int, img_w: int, img_h: int) -> tuple[int, int]:
        """Compute top-left pixel position based on normalized position and anchor."""
        px = int(self.position[0] * frame_width)
        py = int(self.position[1] * frame_height)

        if self.anchor == "center":
            return px - img_w // 2, py - img_h // 2
        elif self.anchor == "top_left":
            return px, py
        elif self.anchor == "top_center":
            return px - img_w // 2, py
        elif self.anchor == "bottom_center":
            return px - img_w // 2, py - img_h
        elif self.anchor == "bottom_left":
            return px, py - img_h
        elif self.anchor == "bottom_right":
            return px - img_w, py - img_h
        return px - img_w // 2, py - img_h // 2

    def _apply(self, video: Video) -> Video:
        frame_h, frame_w = video.frame_shape[:2]

        if self._rendered is None:
            self._rendered = self._render_text_image(frame_w, frame_h)

        overlay_rgba = self._rendered
        oh, ow = overlay_rgba.shape[:2]
        x, y = self._compute_position(frame_w, frame_h, ow, oh)

        # Clamp to frame bounds
        src_x = max(0, -x)
        src_y = max(0, -y)
        dst_x = max(0, x)
        dst_y = max(0, y)
        paste_w = min(ow - src_x, frame_w - dst_x)
        paste_h = min(oh - src_y, frame_h - dst_y)

        if paste_w <= 0 or paste_h <= 0:
            return video

        overlay_region = overlay_rgba[src_y : src_y + paste_h, src_x : src_x + paste_w]
        alpha = overlay_region[:, :, 3:4].astype(np.float32) / 255.0
        overlay_rgb = overlay_region[:, :, :3].astype(np.float32)

        log("Applying text overlay...")
        for frame in progress_iter(video.frames, desc="Text overlay"):
            region = frame[dst_y : dst_y + paste_h, dst_x : dst_x + paste_w]
            blended = (overlay_rgb * alpha + region.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            frame[dst_y : dst_y + paste_h, dst_x : dst_x + paste_w] = blended

        return video

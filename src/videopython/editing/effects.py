"""Effect Operations.

An ``Effect`` is an ``Operation`` that preserves video shape and frame count.
Subclasses override :meth:`Effect._apply` for in-memory execution and may
additionally override :meth:`Effect.streaming_init` / :meth:`Effect.process_frame`
for bounded-memory streaming via ``editing/streaming.py``.

Effects that need to modify audio (``Fade``, ``VolumeAdjust``) override
:meth:`Effect.apply` directly so the audio splice can stay coherent with the
window — the base ``Effect.apply`` only splices frames, restoring the original
audio after ``_apply`` returns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import Field, PrivateAttr, model_validator
from tqdm import tqdm

from videopython.base.description import BoundingBox
from videopython.base.fonts import load_font
from videopython.editing.operation import Effect

if TYPE_CHECKING:
    from videopython.audio import Audio
    from videopython.base.video import Video

logger = logging.getLogger(__name__)

__all__ = [
    "Effect",
    "FullImageOverlay",
    "Blur",
    "Zoom",
    "ColorGrading",
    "Vignette",
    "KenBurns",
    "Fade",
    "VolumeAdjust",
    "TextOverlay",
    "Shake",
    "PunchIn",
    "Flash",
    "ChromaticAberration",
    "Glitch",
    "FilmGrain",
    "Sharpen",
    "Pixelate",
    "MirrorFlip",
    "Kaleidoscope",
]


class FullImageOverlay(Effect):
    """Composites a full-frame image on top of every video frame.

    Useful for watermarks, logos, or static graphic overlays. Supports
    transparency via RGBA images and an overall opacity control. The overlay
    is loaded just-in-time from ``source`` so the op stays JSON-serialisable.
    """

    op: Literal["full_image_overlay"] = "full_image_overlay"
    streamable: ClassVar[bool] = True

    source: Path = Field(
        description=(
            "Path to an RGB or RGBA image file. Loaded at apply time; "
            "the image must match the video's width and height."
        ),
    )
    alpha: float = Field(1.0, ge=0, le=1, description="Overall opacity. 0 = fully transparent, 1 = fully opaque.")
    fade_time: float = Field(
        0.0,
        ge=0,
        description="Seconds to fade the overlay in at the start and out at the end of its time range.",
    )

    _overlay_rgba: np.ndarray | None = PrivateAttr(default=None)
    _stream_total: int = PrivateAttr(default=0)
    _stream_fade_frames: int = PrivateAttr(default=0)

    def _load_overlay(self) -> np.ndarray:
        if self._overlay_rgba is not None:
            return self._overlay_rgba
        img = Image.open(self.source).convert("RGBA")
        self._overlay_rgba = np.array(img, dtype=np.uint8)
        return self._overlay_rgba

    def _overlay_frame(self, img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        overlay = self._load_overlay().copy()
        overlay[:, :, 3] = (overlay[:, :, 3].astype(np.float32) * (self.alpha * alpha)).astype(np.uint8)
        img_pil = Image.fromarray(img)
        overlay_pil = Image.fromarray(overlay)
        img_pil.paste(overlay_pil, (0, 0), overlay_pil)
        return np.array(img_pil)

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        self._load_overlay()
        self._stream_total = total_frames
        self._stream_fade_frames = round(self.fade_time * fps) if self.fade_time > 0 else 0

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        if self._stream_fade_frames == 0:
            return self._overlay_frame(frame)
        dist_from_end = min(frame_index, self._stream_total - 1 - frame_index)
        fade_alpha = 1.0 if dist_from_end >= self._stream_fade_frames else dist_from_end / self._stream_fade_frames
        return self._overlay_frame(frame, fade_alpha)

    def _apply(self, video: Video) -> Video:
        overlay = self._load_overlay()
        if video.frame_shape != overlay[:, :, :3].shape:
            raise ValueError(f"Mismatch of overlay shape `{overlay.shape}` with video shape: `{video.frame_shape}`!")
        if not (0 <= 2 * self.fade_time <= video.total_seconds):
            raise ValueError(f"Video is only {video.total_seconds}s long, but fade time is {self.fade_time}s!")

        logger.info("Overlaying video...")
        n = len(video.frames)
        num_fade_frames = round(self.fade_time * video.fps) if self.fade_time > 0 else 0
        for i in tqdm(range(n), desc="Overlaying frames"):
            if num_fade_frames == 0:
                video.frames[i] = self._overlay_frame(video.frames[i])
            else:
                dist_from_end = min(i, n - i)
                fade_alpha = 1.0 if dist_from_end >= num_fade_frames else dist_from_end / num_fade_frames
                video.frames[i] = self._overlay_frame(video.frames[i], fade_alpha)
        return video


class Blur(Effect):
    """Applies Gaussian blur that can stay constant or ramp up/down over the clip."""

    op: Literal["blur_effect"] = "blur_effect"
    streamable: ClassVar[bool] = True

    mode: Literal["constant", "ascending", "descending"] = Field(
        description=(
            '"constant" applies uniform blur, "ascending" ramps from sharp to blurry, '
            '"descending" ramps from blurry to sharp.'
        ),
    )
    iterations: int = Field(
        ge=1,
        description="Blur strength. Higher values produce a stronger blur (e.g. 5 for subtle, 50+ for heavy).",
    )
    kernel_size: tuple[int, int] = Field(
        (5, 5),
        description=(
            "Gaussian kernel [width, height] in pixels. Must be odd numbers. Larger kernels spread the blur wider."
        ),
    )

    _stream_sigmas: np.ndarray | None = PrivateAttr(default=None)

    def _blur_frame(self, frame: np.ndarray, sigma: float) -> np.ndarray:
        return cv2.GaussianBlur(frame, self.kernel_size, sigma)

    def _compute_sigmas(self, n_frames: int) -> np.ndarray:
        base_sigma = 0.3 * ((self.kernel_size[0] - 1) * 0.5 - 1) + 0.8
        max_sigma = base_sigma * np.sqrt(self.iterations)

        if self.mode == "constant":
            return np.full(n_frames, max_sigma)
        if self.mode == "ascending":
            ratios = np.linspace(1 / n_frames, 1.0, n_frames)
        else:  # descending
            ratios = np.linspace(1.0, 1 / n_frames, n_frames)
        return base_sigma * np.sqrt(np.maximum(1, np.round(ratios * self.iterations)))

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        self._stream_sigmas = self._compute_sigmas(total_frames)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._stream_sigmas is not None
        idx = min(frame_index, len(self._stream_sigmas) - 1)
        return self._blur_frame(frame, self._stream_sigmas[idx])

    def _apply(self, video: Video) -> Video:
        n_frames = len(video.frames)
        sigmas = self._compute_sigmas(n_frames)
        logger.info(f"Applying {self.mode} blur...")
        for i in tqdm(range(n_frames), desc="Blurring"):
            video.frames[i] = self._blur_frame(video.frames[i], sigmas[i])
        return video


class Zoom(Effect):
    """Progressively zooms into or out of the frame center over the clip duration."""

    op: Literal["zoom_effect"] = "zoom_effect"
    streamable: ClassVar[bool] = True

    zoom_factor: float = Field(
        gt=1,
        description="How far to zoom. 1.5 is a subtle push, 2.0 is moderate, 3.0+ is dramatic. Must be greater than 1.",
    )
    mode: Literal["in", "out"] = Field(
        description='"in" starts wide and pushes into the center, "out" starts tight and pulls back.',
    )

    _stream_crops: np.ndarray | None = PrivateAttr(default=None)
    _stream_width: int = PrivateAttr(default=0)
    _stream_height: int = PrivateAttr(default=0)

    def _crop_sizes(self, n_frames: int, width: int, height: int) -> np.ndarray:
        crop_w = np.linspace(width // self.zoom_factor, width, n_frames)
        crop_h = np.linspace(height // self.zoom_factor, height, n_frames)
        if self.mode == "in":
            crop_w, crop_h = crop_w[::-1], crop_h[::-1]
        return np.stack([crop_w, crop_h], axis=1)

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        self._stream_crops = self._crop_sizes(total_frames, width, height)
        self._stream_width = width
        self._stream_height = height

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._stream_crops is not None
        idx = min(frame_index, len(self._stream_crops) - 1)
        w, h = self._stream_crops[idx]
        width, height = self._stream_width, self._stream_height
        x = width / 2 - w / 2
        y = height / 2 - h / 2
        cropped = frame[round(y) : round(y + h), round(x) : round(x + w)]
        return cv2.resize(cropped, (width, height))

    def _apply(self, video: Video) -> Video:
        n_frames = len(video.frames)
        width = video.metadata.width
        height = video.metadata.height
        crops = self._crop_sizes(n_frames, width, height)
        for i in tqdm(range(n_frames), desc="Zooming"):
            w, h = crops[i]
            x = width / 2 - w / 2
            y = height / 2 - h / 2
            cropped = video.frames[i][round(y) : round(y + h), round(x) : round(x + w)]
            video.frames[i] = cv2.resize(cropped, (width, height))
        return video


class ColorGrading(Effect):
    """Adjusts color properties: brightness, contrast, saturation, and temperature."""

    op: Literal["color_adjust"] = "color_adjust"
    streamable: ClassVar[bool] = True

    brightness: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Shift brightness. -1.0 = much darker, 0 = unchanged, 1.0 = much brighter.",
    )
    contrast: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="Scale contrast. 0.5 = flat/washed out, 1.0 = unchanged, 2.0 = high contrast.",
    )
    saturation: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="Scale color intensity. 0.0 = grayscale, 1.0 = unchanged, 2.0 = vivid/oversaturated.",
    )
    temperature: float = Field(
        0.0,
        ge=-1.0,
        le=1.0,
        description="Shift color temperature. -1.0 = cool/blue tint, 0 = neutral, 1.0 = warm/orange tint.",
    )

    def _grade_frame(self, frame: np.ndarray) -> np.ndarray:
        img = frame.astype(np.float32) / 255.0

        if self.brightness != 0:
            img = img + self.brightness
        if self.contrast != 1.0:
            img = (img - 0.5) * self.contrast + 0.5
        if self.saturation != 1.0:
            hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.saturation, 0, 1)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        if self.temperature != 0:
            temp_shift = self.temperature * 0.1
            img[:, :, 0] = img[:, :, 0] + temp_shift
            img[:, :, 2] = img[:, :, 2] - temp_shift

        return np.clip(img * 255, 0, 255).astype(np.uint8)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return self._grade_frame(frame)

    def _apply(self, video: Video) -> Video:
        logger.info("Applying color grading...")
        for i in tqdm(range(len(video.frames)), desc="Color grading"):
            video.frames[i] = self._grade_frame(video.frames[i])
        return video


class Vignette(Effect):
    """Darkens the edges of the frame, drawing attention to the center."""

    op: Literal["vignette"] = "vignette"
    streamable: ClassVar[bool] = True

    strength: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Edge darkness amount. 0.0 = no darkening, 0.5 = moderate, 1.0 = fully black edges.",
    )
    radius: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description=(
            "Size of the bright center area. Smaller values (0.5) create a tight spotlight, "
            "larger values (2.0) keep more of the frame lit."
        ),
    )

    _mask: np.ndarray | None = PrivateAttr(default=None)
    _stream_mask_3d: np.ndarray | None = PrivateAttr(default=None)

    def _create_mask(self, height: int, width: int) -> np.ndarray:
        y = np.linspace(-1, 1, height)
        x = np.linspace(-1, 1, width)
        X, Y = np.meshgrid(x, y)
        distance = np.sqrt(X**2 + Y**2) / self.radius
        mask = 1.0 - np.clip(distance - 0.5, 0, 1) * 2 * self.strength
        return mask.astype(np.float32)

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        if self._mask is None or self._mask.shape != (height, width):
            self._mask = self._create_mask(height, width)
        self._stream_mask_3d = self._mask[:, :, np.newaxis]

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._stream_mask_3d is not None
        return (frame.astype(np.float32) * self._stream_mask_3d).astype(np.uint8)

    def _apply(self, video: Video) -> Video:
        logger.info("Applying vignette effect...")
        height, width = video.frame_shape[:2]
        if self._mask is None or self._mask.shape != (height, width):
            self._mask = self._create_mask(height, width)
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

    op: Literal["ken_burns"] = "ken_burns"
    streamable: ClassVar[bool] = True

    start_region: BoundingBox = Field(
        description="Starting crop region as a BoundingBox with normalized 0-1 coordinates."
    )
    end_region: BoundingBox = Field(description="Ending crop region as a BoundingBox with normalized 0-1 coordinates.")
    easing: Literal["linear", "ease_in", "ease_out", "ease_in_out"] = Field(
        "linear",
        description=(
            'Animation curve. "linear" moves at constant speed, "ease_in" starts slow, '
            '"ease_out" ends slow, "ease_in_out" starts and ends slow.'
        ),
    )

    _stream_regions: np.ndarray | None = PrivateAttr(default=None)
    _stream_target_w: int = PrivateAttr(default=0)
    _stream_target_h: int = PrivateAttr(default=0)

    @model_validator(mode="after")
    def _validate_regions(self) -> KenBurns:
        for name, region in [("start_region", self.start_region), ("end_region", self.end_region)]:
            if not (0 <= region.x <= 1 and 0 <= region.y <= 1):
                raise ValueError(f"{name} position must be in range [0, 1]!")
            if not (0 < region.width <= 1 and 0 < region.height <= 1):
                raise ValueError(f"{name} dimensions must be in range (0, 1]!")
            if region.x + region.width > 1 or region.y + region.height > 1:
                raise ValueError(f"{name} extends beyond image bounds!")
        return self

    def _ease(self, t: float) -> float:
        if self.easing == "linear":
            return t
        if self.easing == "ease_in":
            return t * t
        if self.easing == "ease_out":
            return 1 - (1 - t) * (1 - t)
        # ease_in_out
        if t < 0.5:
            return 2 * t * t
        return 1 - 2 * (1 - t) * (1 - t)

    def _crop_and_scale_frame(
        self, frame: np.ndarray, x: int, y: int, crop_w: int, crop_h: int, target_w: int, target_h: int
    ) -> np.ndarray:
        cropped = frame[y : y + crop_h, x : x + crop_w]
        return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def _precompute_regions(self, n_frames: int, width: int, height: int) -> np.ndarray:
        sx = int(self.start_region.x * width)
        sy = int(self.start_region.y * height)
        sw = int(self.start_region.width * width)
        sh = int(self.start_region.height * height)
        ex = int(self.end_region.x * width)
        ey = int(self.end_region.y * height)
        ew = int(self.end_region.width * width)
        eh = int(self.end_region.height * height)

        regions = np.empty((n_frames, 4), dtype=np.int32)
        for i in range(n_frames):
            t = i / max(1, n_frames - 1)
            et = self._ease(t)
            crop_w = int(sw + (ew - sw) * et)
            crop_h = int(sh + (eh - sh) * et)
            x = max(0, min(int(sx + (ex - sx) * et), width - crop_w))
            y = max(0, min(int(sy + (ey - sy) * et), height - crop_h))
            regions[i] = (x, y, crop_w, crop_h)
        return regions

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        self._stream_regions = self._precompute_regions(total_frames, width, height)
        self._stream_target_w = width
        self._stream_target_h = height

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._stream_regions is not None
        idx = min(frame_index, len(self._stream_regions) - 1)
        x, y, cw, ch = self._stream_regions[idx]
        return self._crop_and_scale_frame(frame, x, y, cw, ch, self._stream_target_w, self._stream_target_h)

    def _apply(self, video: Video) -> Video:
        n_frames = len(video.frames)
        height, width = video.frame_shape[:2]
        regions = self._precompute_regions(n_frames, width, height)
        logger.info("Applying Ken Burns effect...")
        for i in tqdm(range(n_frames), desc="Ken Burns"):
            x, y, cw, ch = regions[i]
            video.frames[i] = self._crop_and_scale_frame(video.frames[i], x, y, cw, ch, width, height)
        return video


def _compute_curve(t: np.ndarray, curve: str) -> np.ndarray:
    if curve == "sqrt":
        return np.sqrt(t)
    if curve == "exponential":
        return t * t
    return t  # linear


class Fade(Effect):
    """Fades video and audio to or from black."""

    op: Literal["fade"] = "fade"
    streamable: ClassVar[bool] = True

    mode: Literal["in", "out", "in_out"] = Field(
        description=('"in" fades from black at the start, "out" fades to black at the end, "in_out" does both.'),
    )
    duration: float = Field(1.0, gt=0, description="Length of each fade in seconds.")
    curve: Literal["sqrt", "linear", "exponential"] = Field(
        "sqrt",
        description=(
            'Brightness ramp shape. "sqrt" feels perceptually even (recommended), '
            '"linear" is mathematically even, "exponential" starts slow and finishes fast.'
        ),
    )

    _stream_alpha: np.ndarray | None = PrivateAttr(default=None)

    def _fade_envelope(self, length: int, rate: float) -> np.ndarray:
        """Build the per-sample (or per-frame) alpha envelope for ``length`` ticks.

        ``rate`` is fps for video frames or sample_rate for audio samples;
        ``self.duration`` is converted to a tick count via ``rate`` and clipped
        to ``length``. The ramp shape follows ``self.curve``.
        """
        ramp = min(round(self.duration * rate), length)
        alpha = np.ones(length, dtype=np.float32)
        if self.mode in ("in", "in_out"):
            t = np.linspace(0, 1, ramp, dtype=np.float32)
            alpha[:ramp] = _compute_curve(t, self.curve)
        if self.mode in ("out", "in_out"):
            t = np.linspace(1, 0, ramp, dtype=np.float32)
            alpha[-ramp:] = np.minimum(alpha[-ramp:], _compute_curve(t, self.curve))
        return alpha

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        self._stream_alpha = self._fade_envelope(total_frames, fps)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._stream_alpha is not None
        idx = min(frame_index, len(self._stream_alpha) - 1)
        a = self._stream_alpha[idx]
        if a == 1.0:
            return frame
        return (frame.astype(np.float32) * a).astype(np.uint8)

    def apply(self, video: Video, **context: Any) -> Video:
        start_s, stop_s = self._resolved_window(video.total_seconds)
        start_f = round(start_s * video.fps)
        end_f = round(stop_s * video.fps)
        n_effect = end_f - start_f
        alpha = self._fade_envelope(n_effect, video.fps)

        batch_size = 64
        for batch_start in range(0, n_effect, batch_size):
            batch_end = min(batch_start + batch_size, n_effect)
            batch_alpha = alpha[batch_start:batch_end, np.newaxis, np.newaxis, np.newaxis]
            if np.all(batch_alpha == 1.0):
                continue
            abs_start = start_f + batch_start
            abs_end = start_f + batch_end
            video.frames[abs_start:abs_end] = (video.frames[abs_start:abs_end].astype(np.float32) * batch_alpha).astype(
                np.uint8
            )

        if video.audio is not None and not video.audio.is_silent:
            self._apply_audio(video.audio, start_s, stop_s)
        return video

    def _apply_audio(self, audio: Audio, start_s: float, stop_s: float) -> None:
        sample_rate = audio.metadata.sample_rate
        audio_start = round(start_s * sample_rate)
        audio_end = min(round(stop_s * sample_rate), len(audio.data))
        alpha = self._fade_envelope(audio_end - audio_start, sample_rate)

        if audio.data.ndim == 1:
            audio.data[audio_start:audio_end] *= alpha
        else:
            audio.data[audio_start:audio_end] *= alpha[:, np.newaxis]
        np.clip(audio.data, -1.0, 1.0, out=audio.data)


class VolumeAdjust(Effect):
    """Changes audio volume within a time range without affecting video frames."""

    op: Literal["volume_adjust"] = "volume_adjust"
    streamable: ClassVar[bool] = True

    volume: float = Field(
        1.0,
        ge=0,
        description="Volume multiplier. 0.0 = silence, 1.0 = original level, 2.0 = twice as loud (may clip).",
    )
    ramp_duration: float = Field(
        0.0,
        ge=0,
        description="Seconds to smoothly ramp volume at the start and end of the window, preventing audible clicks.",
    )

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return frame

    def apply(self, video: Video, **context: Any) -> Video:
        start_s, stop_s = self._resolved_window(video.total_seconds)
        if video.audio is not None and not video.audio.is_silent:
            self._apply_audio(video.audio, start_s, stop_s)
        return video

    def _apply_audio(self, audio: Audio, start_s: float, stop_s: float) -> None:
        sample_rate = audio.metadata.sample_rate
        start_sample = round(start_s * sample_rate)
        end_sample = min(round(stop_s * sample_rate), len(audio.data))
        n_samples = end_sample - start_sample
        envelope = np.full(n_samples, self.volume, dtype=np.float32)

        if self.ramp_duration > 0:
            ramp_samples = min(round(self.ramp_duration * sample_rate), n_samples // 2)
            if ramp_samples > 0:
                t = np.linspace(0, 1, ramp_samples, dtype=np.float32)
                envelope[:ramp_samples] = 1.0 + (self.volume - 1.0) * np.sqrt(t)
                t = np.linspace(1, 0, ramp_samples, dtype=np.float32)
                envelope[-ramp_samples:] = 1.0 + (self.volume - 1.0) * np.sqrt(t)

        if audio.data.ndim == 1:
            audio.data[start_sample:end_sample] *= envelope
        else:
            audio.data[start_sample:end_sample] *= envelope[:, np.newaxis]
        np.clip(audio.data, -1.0, 1.0, out=audio.data)


class TextOverlay(Effect):
    """Draws text on video frames, with auto word-wrap and optional background box."""

    op: Literal["text_overlay"] = "text_overlay"
    streamable: ClassVar[bool] = True

    text: str = Field(min_length=1, description=r"The string to display. Use \n for line breaks.")
    position: tuple[float, float] = Field(
        (0.5, 0.9),
        description=(
            "Where to place the text as normalized (x, y) coordinates. "
            "(0, 0) = top-left corner, (1, 1) = bottom-right corner."
        ),
    )
    font_size: int = Field(48, ge=1, description="Font size in pixels.")
    text_color: tuple[int, int, int] = Field((255, 255, 255), description="Text color as [R, G, B], each 0-255.")
    background_color: tuple[int, int, int, int] | None = Field(
        (0, 0, 0, 160),
        description="Background box color as [R, G, B, A] (0-255), or null to disable the background.",
    )
    background_padding: int = Field(12, ge=0, description="Padding in pixels between text and background edge.")
    max_width: float = Field(
        0.8,
        gt=0.0,
        le=1.0,
        description=(
            "Maximum text width as a fraction of frame width (0-1). Text longer than this wraps to the next line."
        ),
    )
    anchor: Literal["center", "top_left", "top_center", "bottom_center", "bottom_left", "bottom_right"] = Field(
        "center",
        description="Which point of the text box sits at the position coordinate.",
    )
    font_filename: str | None = Field(None, description="Path to a .ttf font file, or None for the default font.")

    _rendered: np.ndarray | None = PrivateAttr(default=None)
    _stream_noop: bool = PrivateAttr(default=False)
    _stream_alpha: np.ndarray | None = PrivateAttr(default=None)
    _stream_rgb: np.ndarray | None = PrivateAttr(default=None)
    _stream_dst: tuple[int, int, int, int] = PrivateAttr(default=(0, 0, 0, 0))

    @model_validator(mode="after")
    def _validate_position(self) -> TextOverlay:
        if not (0.0 <= self.position[0] <= 1.0 and 0.0 <= self.position[1] <= 1.0):
            raise ValueError("position values must be in range [0, 1]")
        return self

    def _get_font(self) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        return load_font(self.font_filename, self.font_size)

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_px: int) -> str:
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
        font = self._get_font()
        max_px = int(self.max_width * frame_width)
        wrapped = self._wrap_text(self.text, font, max_px)

        temp_img = Image.new("RGBA", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.multiline_textbbox((0, 0), wrapped, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad = self.background_padding
        img_w = text_w + 2 * pad
        img_h = text_h + 2 * pad

        if self.background_color is not None:
            img = Image.new("RGBA", (img_w, img_h), self.background_color)
        else:
            img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))

        draw = ImageDraw.Draw(img)
        draw.multiline_text((pad - bbox[0], pad - bbox[1]), wrapped, font=font, fill=(*self.text_color, 255))

        return np.array(img, dtype=np.uint8)

    def _compute_position(self, frame_width: int, frame_height: int, img_w: int, img_h: int) -> tuple[int, int]:
        px = int(self.position[0] * frame_width)
        py = int(self.position[1] * frame_height)

        if self.anchor == "center":
            return px - img_w // 2, py - img_h // 2
        if self.anchor == "top_left":
            return px, py
        if self.anchor == "top_center":
            return px - img_w // 2, py
        if self.anchor == "bottom_center":
            return px - img_w // 2, py - img_h
        if self.anchor == "bottom_left":
            return px, py - img_h
        # bottom_right
        return px - img_w, py - img_h

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        if self._rendered is None:
            self._rendered = self._render_text_image(width, height)
        oh, ow = self._rendered.shape[:2]
        x, y = self._compute_position(width, height, ow, oh)
        src_x = max(0, -x)
        src_y = max(0, -y)
        dst_x = max(0, x)
        dst_y = max(0, y)
        paste_w = min(ow - src_x, width - dst_x)
        paste_h = min(oh - src_y, height - dst_y)
        if paste_w <= 0 or paste_h <= 0:
            self._stream_noop = True
            return
        self._stream_noop = False
        overlay_region = self._rendered[src_y : src_y + paste_h, src_x : src_x + paste_w]
        self._stream_alpha = overlay_region[:, :, 3:4].astype(np.float32) / 255.0
        self._stream_rgb = overlay_region[:, :, :3].astype(np.float32)
        self._stream_dst = (dst_y, dst_x, paste_h, paste_w)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        if self._stream_noop:
            return frame
        assert self._stream_alpha is not None and self._stream_rgb is not None
        dy, dx, ph, pw = self._stream_dst
        region = frame[dy : dy + ph, dx : dx + pw]
        blended = (
            self._stream_rgb * self._stream_alpha + region.astype(np.float32) * (1.0 - self._stream_alpha)
        ).astype(np.uint8)
        frame[dy : dy + ph, dx : dx + pw] = blended
        return frame

    def _apply(self, video: Video) -> Video:
        frame_h, frame_w = video.frame_shape[:2]
        if self._rendered is None:
            self._rendered = self._render_text_image(frame_w, frame_h)

        overlay_rgba = self._rendered
        oh, ow = overlay_rgba.shape[:2]
        x, y = self._compute_position(frame_w, frame_h, ow, oh)

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

        logger.info("Applying text overlay...")
        for frame in tqdm(video.frames, desc="Text overlay"):
            region = frame[dst_y : dst_y + paste_h, dst_x : dst_x + paste_w]
            blended = (overlay_rgb * alpha + region.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
            frame[dst_y : dst_y + paste_h, dst_x : dst_x + paste_w] = blended

        return video


class Shake(Effect):
    """Per-frame camera shake: jitters every frame by a random or rhythmic offset.

    The frame is translated by ``(dx, dy)`` and cropped back to the original
    canvas, so the visible area shrinks slightly at the edges. Useful for
    reaction emphasis, impact moments, or music-synced vibration.
    """

    op: Literal["shake"] = "shake"
    streamable: ClassVar[bool] = True

    intensity_px: float = Field(
        gt=0,
        description="Maximum displacement in pixels at peak intensity (e.g. 5 = subtle, 20 = heavy).",
    )
    mode: Literal["random", "rhythmic", "decay"] = Field(
        "random",
        description=(
            '"random" jitters independently each frame, "rhythmic" oscillates as a sine wave, '
            '"decay" starts at full intensity and fades to zero.'
        ),
    )
    frequency_hz: float = Field(
        8.0,
        gt=0,
        description='Oscillation frequency for "rhythmic" mode. Ignored for other modes.',
    )
    seed: int = Field(
        0,
        description='Seed for the random number generator (used in "random" mode). Same seed = reproducible.',
    )

    _stream_offsets: np.ndarray | None = PrivateAttr(default=None)

    def _compute_offsets(self, n_frames: int, fps: float) -> np.ndarray:
        offsets = np.zeros((n_frames, 2), dtype=np.float32)
        if self.mode == "random":
            rng = np.random.default_rng(self.seed)
            offsets[:] = rng.uniform(-self.intensity_px, self.intensity_px, size=(n_frames, 2))
        elif self.mode == "rhythmic":
            t = np.arange(n_frames, dtype=np.float32) / max(fps, 1e-6)
            phase = 2 * np.pi * self.frequency_hz * t
            offsets[:, 0] = self.intensity_px * np.sin(phase)
            offsets[:, 1] = self.intensity_px * np.cos(phase)
        else:  # decay
            rng = np.random.default_rng(self.seed)
            envelope = np.linspace(1.0, 0.0, n_frames, dtype=np.float32)
            jitter = rng.uniform(-1.0, 1.0, size=(n_frames, 2)).astype(np.float32)
            offsets[:] = jitter * envelope[:, np.newaxis] * self.intensity_px
        return offsets

    def _shake_frame(self, frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
        h, w = frame.shape[:2]
        M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        self._stream_offsets = self._compute_offsets(total_frames, fps)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._stream_offsets is not None
        idx = min(frame_index, len(self._stream_offsets) - 1)
        dx, dy = self._stream_offsets[idx]
        return self._shake_frame(frame, float(dx), float(dy))

    def _apply(self, video: Video) -> Video:
        offsets = self._compute_offsets(len(video.frames), video.fps)
        for i in tqdm(range(len(video.frames)), desc="Shaking"):
            dx, dy = offsets[i]
            video.frames[i] = self._shake_frame(video.frames[i], float(dx), float(dy))
        return video


class PunchIn(Effect):
    """Snap-zoom emphasis: rapidly zooms into the center, holds, optionally releases.

    Different from ``Zoom`` (which ramps continuously over the whole clip).
    ``PunchIn`` reaches the target zoom in ``attack_frames`` and stays there;
    if ``release_frames > 0`` it eases back out at the end.
    """

    op: Literal["punch_in"] = "punch_in"
    streamable: ClassVar[bool] = True

    zoom_factor: float = Field(
        gt=1.0,
        description="Target zoom level. 1.2 is subtle emphasis, 1.5 is moderate, 2.0+ is dramatic.",
    )
    attack_frames: int = Field(
        3,
        ge=0,
        description="Frames to reach full zoom from 1.0. 0 = instant snap, 3 = ~one beat at 30fps.",
    )
    release_frames: int = Field(
        0,
        ge=0,
        description="Frames at the end to ease zoom back to 1.0. 0 = stays zoomed.",
    )

    _stream_zooms: np.ndarray | None = PrivateAttr(default=None)
    _stream_width: int = PrivateAttr(default=0)
    _stream_height: int = PrivateAttr(default=0)

    def _zoom_envelope(self, n_frames: int) -> np.ndarray:
        zooms = np.full(n_frames, self.zoom_factor, dtype=np.float32)
        attack = min(self.attack_frames, n_frames)
        if attack > 0:
            t = np.linspace(0.0, 1.0, attack, dtype=np.float32)
            ease = 1.0 - (1.0 - t) * (1.0 - t)  # ease_out
            zooms[:attack] = 1.0 + (self.zoom_factor - 1.0) * ease
        release = min(self.release_frames, n_frames - attack)
        if release > 0:
            t = np.linspace(1.0, 0.0, release, dtype=np.float32)
            ease = 1.0 - (1.0 - t) * (1.0 - t)
            zooms[-release:] = 1.0 + (self.zoom_factor - 1.0) * ease
        return zooms

    def _zoom_frame(self, frame: np.ndarray, zoom: float, width: int, height: int) -> np.ndarray:
        if zoom <= 1.0 + 1e-6:
            return frame
        crop_w = max(1, int(width / zoom))
        crop_h = max(1, int(height / zoom))
        x = (width - crop_w) // 2
        y = (height - crop_h) // 2
        cropped = frame[y : y + crop_h, x : x + crop_w]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        self._stream_zooms = self._zoom_envelope(total_frames)
        self._stream_width = width
        self._stream_height = height

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._stream_zooms is not None
        idx = min(frame_index, len(self._stream_zooms) - 1)
        return self._zoom_frame(frame, float(self._stream_zooms[idx]), self._stream_width, self._stream_height)

    def _apply(self, video: Video) -> Video:
        n = len(video.frames)
        height, width = video.frame_shape[:2]
        zooms = self._zoom_envelope(n)
        for i in tqdm(range(n), desc="Punching in"):
            video.frames[i] = self._zoom_frame(video.frames[i], float(zooms[i]), width, height)
        return video


class Flash(Effect):
    """Solid-color frame flash that fades in over ``attack_frames`` and out over ``decay_frames``.

    Commonly used between hard cuts, on impact moments, or as a strobe. The
    flash color is blended over the source using an alpha curve that peaks
    at ``peak_alpha`` in the middle of the window.
    """

    op: Literal["flash"] = "flash"
    streamable: ClassVar[bool] = True

    color: tuple[int, int, int] = Field(
        (255, 255, 255),
        description="Flash color as [R, G, B] each 0-255. (255,255,255) is white, (0,0,0) is a blackout.",
    )
    peak_alpha: float = Field(
        1.0,
        gt=0.0,
        le=1.0,
        description="Maximum opacity of the flash. 1.0 fully replaces the frame at peak, 0.5 is a half-blend.",
    )
    attack_frames: int = Field(
        2,
        ge=0,
        description="Frames to ramp from 0 alpha to peak. 0 = instant cut to color.",
    )
    decay_frames: int = Field(
        4,
        ge=0,
        description="Frames to ramp from peak back to 0. 0 = abrupt end.",
    )

    _stream_alpha: np.ndarray | None = PrivateAttr(default=None)
    _stream_color: np.ndarray | None = PrivateAttr(default=None)

    def _alpha_envelope(self, n_frames: int) -> np.ndarray:
        alpha = np.zeros(n_frames, dtype=np.float32)
        attack = min(self.attack_frames, n_frames)
        if attack > 0:
            alpha[:attack] = np.linspace(0.0, self.peak_alpha, attack, dtype=np.float32)
        else:
            alpha[: max(1, attack)] = self.peak_alpha
        decay = min(self.decay_frames, n_frames - attack)
        if decay > 0:
            alpha[attack : attack + decay] = np.linspace(self.peak_alpha, 0.0, decay, dtype=np.float32)
        if attack + decay < n_frames:
            alpha[attack + decay :] = 0.0
        # If attack is 0 and decay is 0, hold peak for the whole window
        if attack == 0 and decay == 0:
            alpha[:] = self.peak_alpha
        return alpha

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        self._stream_alpha = self._alpha_envelope(total_frames)
        self._stream_color = np.array(self.color, dtype=np.float32)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._stream_alpha is not None and self._stream_color is not None
        idx = min(frame_index, len(self._stream_alpha) - 1)
        a = float(self._stream_alpha[idx])
        if a <= 0:
            return frame
        return (frame.astype(np.float32) * (1.0 - a) + self._stream_color * a).astype(np.uint8)

    def _apply(self, video: Video) -> Video:
        n = len(video.frames)
        alpha = self._alpha_envelope(n)
        color = np.array(self.color, dtype=np.float32)
        for i in tqdm(range(n), desc="Flashing"):
            a = float(alpha[i])
            if a <= 0:
                continue
            video.frames[i] = (video.frames[i].astype(np.float32) * (1.0 - a) + color * a).astype(np.uint8)
        return video


class ChromaticAberration(Effect):
    """Splits R and B channels by ``shift_px`` to mimic lens chromatic aberration.

    A defining look of glitch / vaporwave / experimental edits. Use a small
    shift (1-3 px) for a stylistic edge, larger (8+ px) for impact frames.
    """

    op: Literal["chromatic_aberration"] = "chromatic_aberration"
    streamable: ClassVar[bool] = True

    shift_px: int = Field(
        gt=0,
        description="Channel displacement in pixels. 2 is subtle, 6 is noticeable, 12+ is dramatic.",
    )
    mode: Literal["horizontal", "vertical", "radial"] = Field(
        "horizontal",
        description=(
            '"horizontal" shifts R/B sideways, "vertical" shifts them up/down, '
            '"radial" scales R outward and B inward from the center (lens-like).'
        ),
    )

    _stream_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = PrivateAttr(default=None)

    def _build_radial_maps(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cx, cy = width / 2.0, height / 2.0
        max_d = float(max(width, height))
        scale_r = 1.0 - self.shift_px / max_d  # red sampled from slightly outward
        scale_b = 1.0 + self.shift_px / max_d  # blue sampled from slightly inward
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        r_map_x = (x - cx) * scale_r + cx
        r_map_y = (y - cy) * scale_r + cy
        b_map_x = (x - cx) * scale_b + cx
        b_map_y = (y - cy) * scale_b + cy
        return r_map_x, r_map_y, b_map_x, b_map_y

    def _aberrate(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame.copy()
        r = frame[:, :, 0]
        b = frame[:, :, 2]
        if self.mode == "horizontal":
            M_r = np.array([[1.0, 0.0, float(self.shift_px)], [0.0, 1.0, 0.0]], dtype=np.float32)
            M_b = np.array([[1.0, 0.0, float(-self.shift_px)], [0.0, 1.0, 0.0]], dtype=np.float32)
            out[:, :, 0] = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
            out[:, :, 2] = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)
        elif self.mode == "vertical":
            M_r = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, float(self.shift_px)]], dtype=np.float32)
            M_b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, float(-self.shift_px)]], dtype=np.float32)
            out[:, :, 0] = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
            out[:, :, 2] = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)
        else:  # radial
            if self._stream_maps is None:
                self._stream_maps = self._build_radial_maps(w, h)
            r_map_x, r_map_y, b_map_x, b_map_y = self._stream_maps
            out[:, :, 0] = cv2.remap(r, r_map_x, r_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[:, :, 2] = cv2.remap(b, b_map_x, b_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        if self.mode == "radial":
            self._stream_maps = self._build_radial_maps(width, height)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return self._aberrate(frame)

    def _apply(self, video: Video) -> Video:
        h, w = video.frame_shape[:2]
        if self.mode == "radial":
            self._stream_maps = self._build_radial_maps(w, h)
        for i in tqdm(range(len(video.frames)), desc="Chromatic aberration"):
            video.frames[i] = self._aberrate(video.frames[i])
        return video


class Glitch(Effect):
    """Random horizontal slice displacement + channel offsets for a digital-corruption look.

    Each frame gets a fresh set of slices shuffled left/right plus a small R/B
    channel shift. Deterministic given ``seed`` -- the same plan produces the
    same glitch every run.
    """

    op: Literal["glitch"] = "glitch"
    streamable: ClassVar[bool] = True

    intensity: float = Field(
        0.5,
        gt=0.0,
        le=1.0,
        description="Overall glitch strength. 0.2 = subtle, 0.5 = moderate, 1.0 = chaotic.",
    )
    slice_count: int = Field(
        12,
        gt=0,
        description="Number of horizontal slices displaced per frame. More slices = more granular corruption.",
    )
    channel_shift_px: int = Field(
        4,
        ge=0,
        description="Pixels to shift R/B channels for the chromatic-aberration component. 0 disables it.",
    )
    seed: int = Field(0, description="Seed for the per-frame RNG. Same seed = reproducible glitch.")

    def _glitch_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        h, w = frame.shape[:2]
        rng = np.random.default_rng(self.seed + frame_index)
        out = frame.copy()

        max_shift = int(self.intensity * w * 0.15)
        if max_shift > 0 and self.slice_count > 0:
            edges = np.sort(rng.integers(0, h, size=self.slice_count + 1))
            edges[0] = 0
            edges[-1] = h
            for i in range(self.slice_count):
                y0, y1 = int(edges[i]), int(edges[i + 1])
                if y1 <= y0:
                    continue
                shift = int(rng.integers(-max_shift, max_shift + 1))
                if shift == 0:
                    continue
                out[y0:y1] = np.roll(out[y0:y1], shift, axis=1)

        if self.channel_shift_px > 0:
            s = int(self.channel_shift_px * self.intensity) or 1
            out[:, :, 0] = np.roll(out[:, :, 0], s, axis=1)
            out[:, :, 2] = np.roll(out[:, :, 2], -s, axis=1)

        return out

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        return None

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return self._glitch_frame(frame, frame_index)

    def _apply(self, video: Video) -> Video:
        for i in tqdm(range(len(video.frames)), desc="Glitching"):
            video.frames[i] = self._glitch_frame(video.frames[i], i)
        return video


class FilmGrain(Effect):
    """Additive Gaussian noise simulating film grain.

    Seeded per-frame so renders are reproducible. ``monochrome=True`` keeps
    the noise luma-only (cinematic), False adds independent RGB noise
    (digital / 8-bit look).
    """

    op: Literal["film_grain"] = "film_grain"
    streamable: ClassVar[bool] = True

    intensity: float = Field(
        0.1,
        gt=0.0,
        le=1.0,
        description="Noise standard deviation as a fraction of full-scale. 0.05 subtle, 0.2 heavy.",
    )
    monochrome: bool = Field(
        True,
        description="True applies the same noise to all RGB channels (luma-only). False uses per-channel noise.",
    )
    seed: int = Field(0, description="Seed for the noise RNG. Same seed = same grain pattern.")

    def _grain_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + frame_index)
        h, w = frame.shape[:2]
        amp = self.intensity * 255.0
        if self.monochrome:
            noise = rng.standard_normal((h, w, 1), dtype=np.float32) * amp
        else:
            noise = rng.standard_normal((h, w, 3), dtype=np.float32) * amp
        return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        return None

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return self._grain_frame(frame, frame_index)

    def _apply(self, video: Video) -> Video:
        for i in tqdm(range(len(video.frames)), desc="Adding grain"):
            video.frames[i] = self._grain_frame(video.frames[i], i)
        return video


class Sharpen(Effect):
    """Unsharp-mask sharpening: blur the frame and subtract from itself with weight.

    ``amount=0`` returns the original frame; higher values produce a crisper
    look at the cost of edge halos.
    """

    op: Literal["sharpen"] = "sharpen"
    streamable: ClassVar[bool] = True

    amount: float = Field(
        1.0,
        ge=0.0,
        le=3.0,
        description="Sharpening strength. 0.5 is subtle, 1.0 moderate, 2.0+ aggressive (may cause halos).",
    )
    kernel_size: int = Field(
        5,
        ge=3,
        description="Gaussian blur kernel size used to build the unsharp mask. Must be odd; larger = wider halos.",
    )

    @model_validator(mode="after")
    def _validate_kernel(self) -> Sharpen:
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")
        return self

    def _sharpen_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.amount == 0:
            return frame
        blurred = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), 0)
        sharpened = cv2.addWeighted(frame, 1.0 + self.amount, blurred, -self.amount, 0)
        return sharpened

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        return None

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return self._sharpen_frame(frame)

    def _apply(self, video: Video) -> Video:
        for i in tqdm(range(len(video.frames)), desc="Sharpening"):
            video.frames[i] = self._sharpen_frame(video.frames[i])
        return video


class Pixelate(Effect):
    """Mosaic blocks: downscale + nearest-neighbour upscale, optionally limited to a region.

    Useful for face censoring (combine with ``BoundingBox`` from face
    detection) or a stylistic 8-bit look.
    """

    op: Literal["pixelate"] = "pixelate"
    streamable: ClassVar[bool] = True

    block_size: int = Field(
        gt=1,
        description="Mosaic block size in pixels. 8 is coarse, 32 is censor-grade, 64 is heavy.",
    )
    region: BoundingBox | None = Field(
        None,
        description="Optional normalized region (0-1) to pixelate. None = full frame.",
    )

    _stream_region_px: tuple[int, int, int, int] | None = PrivateAttr(default=None)

    def _resolve_region(self, width: int, height: int) -> tuple[int, int, int, int]:
        if self.region is None:
            return 0, 0, width, height
        x = max(0, int(self.region.x * width))
        y = max(0, int(self.region.y * height))
        w = max(1, min(int(self.region.width * width), width - x))
        h = max(1, min(int(self.region.height * height), height - y))
        return x, y, w, h

    def _pixelate_frame(self, frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = region
        crop = frame[y : y + h, x : x + w]
        small_w = max(1, w // self.block_size)
        small_h = max(1, h // self.block_size)
        small = cv2.resize(crop, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y : y + h, x : x + w] = big
        return frame

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        self._stream_region_px = self._resolve_region(width, height)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert self._stream_region_px is not None
        return self._pixelate_frame(frame, self._stream_region_px)

    def _apply(self, video: Video) -> Video:
        h, w = video.frame_shape[:2]
        region = self._resolve_region(w, h)
        for i in tqdm(range(len(video.frames)), desc="Pixelating"):
            video.frames[i] = self._pixelate_frame(video.frames[i], region)
        return video


class MirrorFlip(Effect):
    """Flip frames or reflect one half onto the other.

    ``horizontal`` / ``vertical`` are plain mirror flips. The ``mirror_*``
    modes reflect one half of the frame onto the opposite half, producing
    a symmetric image with the chosen half preserved.
    """

    op: Literal["mirror_flip"] = "mirror_flip"
    streamable: ClassVar[bool] = True

    mode: Literal[
        "horizontal",
        "vertical",
        "mirror_left",
        "mirror_right",
        "mirror_top",
        "mirror_bottom",
    ] = Field(
        description=(
            '"horizontal" / "vertical" flip the whole frame. '
            '"mirror_left" reflects the left half onto the right (and analogously for the other mirror_ modes).'
        ),
    )

    def _flip_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.mode == "horizontal":
            return cv2.flip(frame, 1)
        if self.mode == "vertical":
            return cv2.flip(frame, 0)

        out = frame.copy()
        if self.mode == "mirror_left":
            half = w // 2
            out[:, w - half :] = out[:, :half][:, ::-1]
        elif self.mode == "mirror_right":
            half = w // 2
            out[:, :half] = out[:, w - half :][:, ::-1]
        elif self.mode == "mirror_top":
            half = h // 2
            out[h - half :, :] = out[:half, :][::-1, :]
        else:  # mirror_bottom
            half = h // 2
            out[:half, :] = out[h - half :, :][::-1, :]
        return out

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        return None

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return self._flip_frame(frame)

    def _apply(self, video: Video) -> Video:
        for i in tqdm(range(len(video.frames)), desc="Mirroring"):
            video.frames[i] = self._flip_frame(video.frames[i])
        return video


class Kaleidoscope(Effect):
    """N-way radial mirror around the frame center.

    Samples one wedge of the frame and reflects it ``segments`` times around
    the center. The mapping is precomputed once per stream, so per-frame cost
    is a single ``cv2.remap``.
    """

    op: Literal["kaleidoscope"] = "kaleidoscope"
    streamable: ClassVar[bool] = True

    segments: int = Field(
        6,
        ge=2,
        le=24,
        description="Number of mirror segments. 6 is a classic snowflake, 12 is dense, 2 is minimal.",
    )
    angle_offset: float = Field(
        0.0,
        description="Rotation of the kaleidoscope pattern in radians (e.g. pi/2 rotates by 90 degrees).",
    )

    _stream_map_x: np.ndarray | None = PrivateAttr(default=None)
    _stream_map_y: np.ndarray | None = PrivateAttr(default=None)

    def _build_maps(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx) - self.angle_offset

        wedge = 2.0 * np.pi / self.segments
        half_wedge = wedge * 0.5
        theta_mod = np.mod(theta, wedge)
        theta_folded = np.where(theta_mod > half_wedge, wedge - theta_mod, theta_mod)
        theta_final = theta_folded + self.angle_offset

        src_x = (cx + r * np.cos(theta_final)).astype(np.float32)
        src_y = (cy + r * np.sin(theta_final)).astype(np.float32)
        return src_x, src_y

    def _kaleidoscope_frame(self, frame: np.ndarray) -> np.ndarray:
        assert self._stream_map_x is not None and self._stream_map_y is not None
        return cv2.remap(
            frame,
            self._stream_map_x,
            self._stream_map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        self._stream_map_x, self._stream_map_y = self._build_maps(width, height)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return self._kaleidoscope_frame(frame)

    def _apply(self, video: Video) -> Video:
        h, w = video.frame_shape[:2]
        self._stream_map_x, self._stream_map_y = self._build_maps(w, h)
        for i in tqdm(range(len(video.frames)), desc="Kaleidoscope"):
            video.frames[i] = self._kaleidoscope_frame(video.frames[i])
        return video

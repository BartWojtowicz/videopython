"""Effect Operations.

An ``Effect`` is an ``Operation`` that preserves video shape and frame count.
Subclasses override :meth:`Effect._apply` for in-memory execution and may
additionally override :meth:`Effect.streaming_init` / :meth:`Effect.process_frame`
for bounded-memory streaming via ``base/streaming.py``.

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
from videopython.base.operation import Effect

if TYPE_CHECKING:
    from videopython.base.audio import Audio
    from videopython.base.video import Video

logger = logging.getLogger(__name__)

# Transitional alias: editing/video_edit.py still imports ``AudioEffect`` until
# it is rewritten in the same migration. ``VolumeAdjust`` is the only consumer
# and now inherits directly from ``Effect``.
AudioEffect = Effect

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


class FullImageOverlay(Effect):
    """Composites a full-frame image on top of every video frame.

    Useful for watermarks, logos, or static graphic overlays. Supports
    transparency via RGBA images and an overall opacity control. The overlay
    is loaded just-in-time from ``source`` so the op stays JSON-serialisable.

    Args:
        source: Path to an RGB or RGBA image file. Loaded at apply time; the
            image must match the video's width and height.
        alpha: Overall opacity. 0 = fully transparent, 1 = fully opaque.
        fade_time: Seconds to fade the overlay in at the start and out at the
            end of its time range.
    """

    op: Literal["full_image_overlay"] = "full_image_overlay"
    streamable: ClassVar[bool] = True

    source: Path
    alpha: float = Field(1.0, ge=0, le=1)
    fade_time: float = Field(0.0, ge=0)

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
    """Applies Gaussian blur that can stay constant or ramp up/down over the clip.

    Args:
        mode: "constant" applies uniform blur, "ascending" ramps from sharp to
            blurry, "descending" ramps from blurry to sharp.
        iterations: Blur strength. Higher values produce a stronger blur
            (e.g. 5 for subtle, 50+ for heavy).
        kernel_size: Gaussian kernel [width, height] in pixels. Must be odd
            numbers. Larger kernels spread the blur wider.
    """

    op: Literal["blur_effect"] = "blur_effect"
    streamable: ClassVar[bool] = True

    mode: Literal["constant", "ascending", "descending"]
    iterations: int = Field(ge=1)
    kernel_size: tuple[int, int] = (5, 5)

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
    """Progressively zooms into or out of the frame center over the clip duration.

    Args:
        zoom_factor: How far to zoom. 1.5 is a subtle push, 2.0 is moderate,
            3.0+ is dramatic. Must be greater than 1.
        mode: "in" starts wide and pushes into the center, "out" starts tight
            and pulls back.
    """

    op: Literal["zoom_effect"] = "zoom_effect"
    streamable: ClassVar[bool] = True

    zoom_factor: float = Field(gt=1)
    mode: Literal["in", "out"]

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
    """Adjusts color properties: brightness, contrast, saturation, and temperature.

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

    op: Literal["color_adjust"] = "color_adjust"
    streamable: ClassVar[bool] = True

    brightness: float = Field(0.0, ge=-1.0, le=1.0)
    contrast: float = Field(1.0, ge=0.5, le=2.0)
    saturation: float = Field(1.0, ge=0.0, le=2.0)
    temperature: float = Field(0.0, ge=-1.0, le=1.0)

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
    """Darkens the edges of the frame, drawing attention to the center.

    Args:
        strength: Edge darkness amount. 0.0 = no darkening, 0.5 = moderate,
            1.0 = fully black edges.
        radius: Size of the bright center area. Smaller values (0.5) create a
            tight spotlight, larger values (2.0) keep more of the frame lit.
    """

    op: Literal["vignette"] = "vignette"
    streamable: ClassVar[bool] = True

    strength: float = Field(0.5, ge=0.0, le=1.0)
    radius: float = Field(1.0, ge=0.5, le=2.0)

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

    Args:
        start_region: Starting crop region as a BoundingBox with normalized
            0-1 coordinates.
        end_region: Ending crop region as a BoundingBox with normalized
            0-1 coordinates.
        easing: Animation curve. "linear" moves at constant speed, "ease_in"
            starts slow, "ease_out" ends slow, "ease_in_out" starts and ends
            slow.
    """

    op: Literal["ken_burns"] = "ken_burns"
    streamable: ClassVar[bool] = True

    start_region: BoundingBox
    end_region: BoundingBox
    easing: Literal["linear", "ease_in", "ease_out", "ease_in_out"] = "linear"

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
    """Fades video and audio to or from black.

    Args:
        mode: "in" fades from black at the start, "out" fades to black at the
            end, "in_out" does both.
        duration: Length of each fade in seconds.
        curve: Brightness ramp shape. "sqrt" feels perceptually even
            (recommended), "linear" is mathematically even, "exponential"
            starts slow and finishes fast.
    """

    op: Literal["fade"] = "fade"
    streamable: ClassVar[bool] = True

    mode: Literal["in", "out", "in_out"]
    duration: float = Field(1.0, gt=0)
    curve: Literal["sqrt", "linear", "exponential"] = "sqrt"

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

    def apply(self, video: Video, **context: Any) -> Video:  # type: ignore[override]
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
    """Changes audio volume within a time range without affecting video frames.

    Args:
        volume: Volume multiplier. 0.0 = silence, 1.0 = original level,
            2.0 = twice as loud (may clip).
        ramp_duration: Seconds to smoothly ramp volume at the start and end of
            the window, preventing audible clicks.
    """

    op: Literal["volume_adjust"] = "volume_adjust"
    streamable: ClassVar[bool] = True

    volume: float = Field(1.0, ge=0)
    ramp_duration: float = Field(0.0, ge=0)

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        return frame

    def apply(self, video: Video, **context: Any) -> Video:  # type: ignore[override]
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
    r"""Draws text on video frames, with auto word-wrap and optional background box.

    Args:
        text: The string to display. Use \n for line breaks.
        position: Where to place the text as normalized (x, y) coordinates.
            (0, 0) = top-left corner, (1, 1) = bottom-right corner.
        font_size: Font size in pixels.
        text_color: Text color as [R, G, B], each 0-255.
        background_color: Background box color as [R, G, B, A] (0-255), or
            null to disable the background.
        background_padding: Padding in pixels between text and background edge.
        max_width: Maximum text width as a fraction of frame width (0-1). Text
            longer than this wraps to the next line.
        anchor: Which point of the text box sits at the position coordinate.
        font_filename: Path to a .ttf font file, or None for the default font.
    """

    op: Literal["text_overlay"] = "text_overlay"
    streamable: ClassVar[bool] = True

    text: str = Field(min_length=1)
    position: tuple[float, float] = (0.5, 0.9)
    font_size: int = Field(48, ge=1)
    text_color: tuple[int, int, int] = (255, 255, 255)
    background_color: tuple[int, int, int, int] | None = (0, 0, 0, 160)
    background_padding: int = Field(12, ge=0)
    max_width: float = Field(0.8, gt=0.0, le=1.0)
    anchor: Literal["center", "top_left", "top_center", "bottom_center", "bottom_left", "bottom_right"] = "center"
    font_filename: str | None = None

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
        if self.font_filename:
            return ImageFont.truetype(self.font_filename, self.font_size)
        try:
            return ImageFont.truetype("DejaVuSans.ttf", self.font_size)
        except OSError:
            return ImageFont.load_default()

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

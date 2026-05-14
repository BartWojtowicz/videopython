"""Transform Operations.

A transform is any Operation that produces a new ``Video`` from a single
input video, free to change dimensions, fps, duration, or frame count.
See ``base/operation.py`` for the ``Operation`` base.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import cv2
import numpy as np
from pydantic import Field, model_validator
from tqdm import tqdm

from videopython.base.operation import FilterCtx, OpCategory, Operation
from videopython.base.video import Video, _round_dimension_to_even

if TYPE_CHECKING:
    from videopython.base.audio import Audio
    from videopython.base.text.transcription import Transcription
    from videopython.base.video import VideoMetadata

logger = logging.getLogger(__name__)

__all__ = [
    "Transformation",
    "CutFrames",
    "CutSeconds",
    "Resize",
    "ResampleFPS",
    "Crop",
    "CropMode",
    "SpeedChange",
    "PictureInPicture",
    "Reverse",
    "FreezeFrame",
    "SilenceRemoval",
]

# Transitional alias kept while editing/video_edit.py and ai/transforms.py are
# rewritten in follow-up commits; deleted along with the registry once those
# consumers are gone.
Transformation = Operation


class CutFrames(Operation):
    """Cuts video to a specific frame range.

    Args:
        start: Start frame index (inclusive).
        end: End frame index (exclusive).
    """

    op: Literal["cut_frames"] = "cut_frames"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_range(self) -> CutFrames:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self

    def apply(self, video: Video) -> Video:
        return video[self.start : self.end]

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        if self.end > meta.frame_count:
            raise ValueError(f"end frame ({self.end}) exceeds frame count ({meta.frame_count})")
        duration = round((self.end - self.start) / meta.fps, 4)
        return meta.with_duration(duration)


class CutSeconds(Operation):
    """Cuts video to a specific time range in seconds.

    Args:
        start: Start time in seconds.
        end: End time in seconds.
    """

    op: Literal["cut"] = "cut"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def _validate_range(self) -> CutSeconds:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self

    def apply(self, video: Video) -> Video:
        return video[round(self.start * video.fps) : round(self.end * video.fps)]

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        if self.end > meta.total_seconds:
            raise ValueError(f"end time ({self.end}) exceeds video duration ({meta.total_seconds})")
        # Mirror apply(): round both endpoints to frames before computing the duration.
        start_f = round(self.start * meta.fps)
        end_f = round(self.end * meta.fps)
        duration = round((end_f - start_f) / meta.fps, 4)
        return meta.with_duration(duration)


class Resize(Operation):
    """Resizes video to specified dimensions, preserving aspect ratio if only one dimension is given.

    Args:
        width: Target width in pixels, or None to maintain aspect ratio.
        height: Target height in pixels, or None to maintain aspect ratio.
        round_to_even: If True (default), snap output width/height to even numbers.
    """

    op: Literal["resize"] = "resize"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    streamable: ClassVar[bool] = True

    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)
    round_to_even: bool = True

    @model_validator(mode="after")
    def _require_one_dimension(self) -> Resize:
        if self.width is None and self.height is None:
            raise ValueError("Resize requires `width`, `height`, or both.")
        return self

    def _resolve_dims(self, src_w: int, src_h: int) -> tuple[int, int]:
        if self.width is not None and self.height is not None:
            new_w, new_h = self.width, self.height
        elif self.width is not None:
            new_w = self.width
            new_h = round(src_h * (self.width / src_w))
        else:
            assert self.height is not None
            new_h = self.height
            new_w = round(src_w * (self.height / src_h))
        if self.round_to_even:
            new_w = _round_dimension_to_even(new_w)
            new_h = _round_dimension_to_even(new_h)
        return new_w, new_h

    def apply(self, video: Video) -> Video:
        new_w, new_h = self._resolve_dims(video.video_shape[2], video.video_shape[1])
        logger.info(f"Resizing video to: {new_w}x{new_h}!")
        video.frames = np.asarray(
            [cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA) for frame in video.frames],
            dtype=np.uint8,
        )
        return video

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        new_w, new_h = self._resolve_dims(meta.width, meta.height)
        return meta.with_dimensions(new_w, new_h)

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        new_w, new_h = self._resolve_dims(ctx.width, ctx.height)
        return f"scale={new_w}:{new_h}"


class ResampleFPS(Operation):
    """Resamples video to a different frame rate, upsampling or downsampling as needed.

    Args:
        fps: Target frames per second.
    """

    op: Literal["resample_fps"] = "resample_fps"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    streamable: ClassVar[bool] = True

    fps: float = Field(gt=0)

    def _downsample(self, video: Video) -> Video:
        target = int(len(video.frames) * (self.fps / video.fps))
        idx = np.round(np.linspace(0, len(video.frames) - 1, target)).astype(int)
        video.frames = video.frames[idx]
        video.fps = self.fps
        return video

    def _upsample(self, video: Video) -> Video:
        target = int(len(video.frames) * (self.fps / video.fps))
        positions = np.linspace(0, len(video.frames) - 1, target)
        new_frames = []
        for pos in tqdm(positions, desc="Interpolating frames"):
            ratio = pos % 1
            low = int(pos)
            high = int(np.ceil(pos))
            frame = (1 - ratio) * video.frames[low] + ratio * video.frames[high]
            new_frames.append(frame.astype(np.uint8))
        video.frames = np.array(new_frames, dtype=np.uint8)
        video.fps = self.fps
        return video

    def apply(self, video: Video) -> Video:
        if video.fps == self.fps:
            return video
        if video.fps > self.fps:
            logger.info(f"Downsampling video from {video.fps} to {self.fps} FPS.")
            video = self._downsample(video)
        else:
            logger.info(f"Upsampling video from {video.fps} to {self.fps} FPS.")
            video = self._upsample(video)
        if video.audio is not None:
            video.audio = video.audio.fit_to_duration(len(video.frames) / video.fps)
        return video

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        return meta.with_fps(self.fps)

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        return f"fps={self.fps}"


class CropMode(str, Enum):
    CENTER = "center"
    CUSTOM = "custom"


class Crop(Operation):
    """Crops the frame to a smaller region.

    Accepts pixel values (int) or normalized 0-1 fractions (float). For
    example, width=0.5 crops to 50% of the original width.

    Args:
        width: Crop width in pixels (int) or fraction in (0, 1] of source width.
        height: Crop height in pixels (int) or fraction in (0, 1] of source height.
        x: Left edge X (only with mode='custom'). Pixels (int) or fraction in [0, 1].
        y: Top edge Y (only with mode='custom'). Pixels (int) or fraction in [0, 1].
        mode: 'center' crops from the middle, 'custom' uses x/y coordinates.
    """

    op: Literal["crop"] = "crop"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    streamable: ClassVar[bool] = True

    width: int | float
    height: int | float
    x: int | float = 0
    y: int | float = 0
    mode: CropMode = CropMode.CENTER

    @staticmethod
    def _to_pixels(value: int | float, dimension: int) -> int:
        if isinstance(value, float) and 0 < value <= 1:
            return int(value * dimension)
        return int(value)

    def _resolve_box(self, src_w: int, src_h: int) -> tuple[int, int, int, int]:
        """Returns (x, y, width, height) in pixels for the resolved crop box."""
        cw = self._to_pixels(self.width, src_w)
        ch = self._to_pixels(self.height, src_h)
        if self.mode == CropMode.CENTER:
            cx = (src_w - cw) // 2
            cy = (src_h - ch) // 2
        else:
            cx = self._to_pixels(self.x, src_w)
            cy = self._to_pixels(self.y, src_h)
        return cx, cy, cw, ch

    def apply(self, video: Video) -> Video:
        src_h, src_w = video.frame_shape[:2]
        cx, cy, cw, ch = self._resolve_box(src_w, src_h)
        if self.mode == CropMode.CENTER:
            mid_w, mid_h = src_w // 2, src_h // 2
            off_w, off_h = cw // 2, ch // 2
            video.frames = video.frames[:, mid_h - off_h : mid_h + off_h, mid_w - off_w : mid_w + off_w, :]
        else:
            video.frames = video.frames[:, cy : cy + ch, cx : cx + cw, :]
        return video

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        _, _, cw, ch = self._resolve_box(meta.width, meta.height)
        if cw > meta.width or ch > meta.height:
            raise ValueError(f"Crop {cw}x{ch} exceeds source {meta.width}x{meta.height}")
        return meta.with_dimensions(cw, ch)

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        cx, cy, cw, ch = self._resolve_box(ctx.width, ctx.height)
        return f"crop={cw}:{ch}:{cx}:{cy}"


class SpeedChange(Operation):
    """Speeds up or slows down video playback, optionally ramping between two speeds.

    Args:
        speed: Playback speed multiplier. 2.0 = twice as fast, 0.5 = half speed.
        end_speed: If set, smoothly ramp from speed to end_speed over the clip duration.
        interpolate: Blend between frames when slowing down for smoother motion.
        adjust_audio: Time-stretch audio to match the new speed.
    """

    op: Literal["speed_change"] = "speed_change"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    speed: float = Field(gt=0)
    end_speed: float | None = Field(None, gt=0)
    interpolate: bool = True
    adjust_audio: bool = True

    def _new_frame_count(self, n_frames: int) -> int:
        if self.end_speed is None:
            return int(n_frames / self.speed)
        avg = (self.speed + self.end_speed) / 2
        return int(n_frames / avg)

    def apply(self, video: Video) -> Video:
        n_frames = len(video.frames)

        if self.end_speed is None:
            logger.info(f"Applying {self.speed}x speed change...")
            new_count = int(n_frames / self.speed)
            if new_count == 0:
                raise ValueError(f"Speed {self.speed}x would result in 0 frames!")
            source_indices = np.linspace(0, n_frames - 1, new_count)
        else:
            logger.info(f"Applying speed ramp from {self.speed}x to {self.end_speed}x...")
            num_samples = 1000
            speeds = np.linspace(self.speed, self.end_speed, num_samples)
            time_per_sample = 1.0 / speeds
            cumulative_time = np.cumsum(time_per_sample)
            cumulative_time = cumulative_time / cumulative_time[-1]
            new_count = self._new_frame_count(n_frames)
            if new_count == 0:
                raise ValueError("Speed ramp would result in 0 frames!")
            output_positions = np.linspace(0, 1, new_count)
            source_positions = np.interp(output_positions, cumulative_time, np.linspace(0, 1, num_samples))
            source_indices = source_positions * (n_frames - 1)

        slow = self.speed < 1.0 or (self.end_speed is not None and self.end_speed < 1.0)
        if self.interpolate and slow:
            new_frames = []
            for idx in tqdm(source_indices, desc="Interpolating frames"):
                idx_low = int(idx)
                idx_high = min(idx_low + 1, n_frames - 1)
                ratio = idx - idx_low
                if ratio == 0 or idx_low == idx_high:
                    new_frames.append(video.frames[idx_low])
                else:
                    frame = (1 - ratio) * video.frames[idx_low] + ratio * video.frames[idx_high]
                    new_frames.append(frame.astype(np.uint8))
            video.frames = np.array(new_frames, dtype=np.uint8)
        else:
            indices = np.clip(np.round(source_indices).astype(int), 0, n_frames - 1)
            video.frames = video.frames[indices]

        target_duration = len(video.frames) / video.fps
        if video.audio is not None and not video.audio.is_silent:
            if self.adjust_audio:
                eff_speed = self.speed if self.end_speed is None else (self.speed + self.end_speed) / 2
                video.audio = video.audio.time_stretch(eff_speed)
            video.audio = video.audio.fit_to_duration(target_duration)
        elif video.audio is not None:
            video.audio = video.audio.fit_to_duration(target_duration)
        return video

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        new_count = self._new_frame_count(meta.frame_count)
        if new_count == 0:
            raise ValueError(f"Speed {self.speed}x would result in 0 frames!")
        from videopython.base.video import VideoMetadata as _Meta

        return _Meta(
            height=meta.height,
            width=meta.width,
            fps=meta.fps,
            frame_count=new_count,
            total_seconds=round(new_count / meta.fps, 4),
        )


class PictureInPicture(Operation):
    """Places a smaller video on top of the main video (picture-in-picture).

    The overlay is loaded just-in-time from ``source`` when ``apply`` runs, so
    a ``PictureInPicture`` instance is fully JSON-serialisable.

    Args:
        source: Path to the overlay video file.
        position: Center of the inset as normalized (x, y) in [0, 1].
        scale: Inset width as a fraction of the main video width, in (0, 1].
        border_width: Border thickness in pixels. 0 = no border.
        border_color: Border color as [R, G, B], each in [0, 255].
        corner_radius: Rounded corner radius in pixels. 0 = square corners.
        opacity: Inset transparency in [0, 1]. 0 = invisible, 1 = fully opaque.
        audio_mode: 'main' keeps only main audio, 'overlay' uses inset audio,
            'mix' blends both.
        audio_mix: Volume levels as [main, overlay] when audio_mode is 'mix'.
    """

    op: Literal["picture_in_picture"] = "picture_in_picture"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    source: Path
    position: tuple[float, float] = (0.7, 0.7)
    scale: float = Field(0.25, gt=0, le=1)
    border_width: int = Field(0, ge=0)
    border_color: tuple[int, int, int] = (255, 255, 255)
    corner_radius: int = Field(0, ge=0)
    opacity: float = Field(1.0, ge=0, le=1)
    audio_mode: Literal["main", "overlay", "mix"] = "main"
    audio_mix: tuple[float, float] = (1.0, 1.0)

    @model_validator(mode="after")
    def _validate(self) -> PictureInPicture:
        if not (0 <= self.position[0] <= 1 and 0 <= self.position[1] <= 1):
            raise ValueError("position must be normalized values in [0, 1]")
        if self.audio_mix[0] < 0 or self.audio_mix[1] < 0:
            raise ValueError("audio_mix factors must be non-negative")
        return self

    def _create_rounded_mask(self, width: int, height: int) -> np.ndarray:
        if self.corner_radius <= 0:
            return np.ones((height, width), dtype=np.float32)
        radius = min(self.corner_radius, width // 2, height // 2)
        mask = np.ones((height, width), dtype=np.float32)
        corners = [
            (0, radius, 0, radius, radius, radius),
            (0, radius, width - radius, width, width - radius, radius),
            (height - radius, height, 0, radius, radius, height - radius),
            (height - radius, height, width - radius, width, width - radius, height - radius),
        ]
        radius_sq = float(radius * radius)
        for y_start, y_end, x_start, x_end, center_x, center_y in corners:
            yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
            outside = ((xx - center_x) ** 2 + (yy - center_y) ** 2) > radius_sq
            mask[y_start:y_end, x_start:x_end][outside] = 0.0
        return mask

    def _add_border(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.border_width <= 0:
            return frame
        bordered = frame.copy()
        border_color = np.array(self.border_color, dtype=np.uint8)
        bordered[: self.border_width, :] = border_color
        bordered[-self.border_width :, :] = border_color
        bordered[:, : self.border_width] = border_color
        bordered[:, -self.border_width :] = border_color
        if self.corner_radius > 0:
            bordered = bordered * mask[:, :, None].astype(np.uint8)
        return bordered

    def apply(self, video: Video) -> Video:
        overlay = Video.from_path(str(self.source))
        main_h, main_w = video.frame_shape[:2]
        n_main = len(video.frames)
        n_overlay = len(overlay.frames)

        overlay_w = int(main_w * self.scale)
        overlay_aspect = overlay.metadata.width / overlay.metadata.height
        overlay_h = int(overlay_w / overlay_aspect)

        pos_x = int(self.position[0] * main_w - overlay_w / 2)
        pos_y = int(self.position[1] * main_h - overlay_h / 2)
        pos_x = max(0, min(pos_x, main_w - overlay_w))
        pos_y = max(0, min(pos_y, main_h - overlay_h))

        logger.info(f"Resizing overlay to {overlay_w}x{overlay_h}...")
        resized_overlay = np.asarray(
            [
                cv2.resize(frame, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
                for frame in tqdm(overlay.frames, desc="Resizing overlay")
            ],
            dtype=np.uint8,
        )

        mask = self._create_rounded_mask(overlay_w, overlay_h)
        alpha = mask[:, :, None] * self.opacity

        if self.border_width > 0:
            resized_overlay = np.asarray(
                [self._add_border(frame, mask) for frame in resized_overlay],
                dtype=np.uint8,
            )

        logger.info("Applying picture-in-picture...")
        new_frames = []
        for i in tqdm(range(n_main), desc="Picture-in-picture"):
            main_frame = video.frames[i].copy()
            overlay_frame = resized_overlay[i % n_overlay]
            region = main_frame[pos_y : pos_y + overlay_h, pos_x : pos_x + overlay_w]
            if self.corner_radius > 0 or self.opacity < 1.0:
                blended = (overlay_frame * alpha + region * (1 - alpha)).astype(np.uint8)
            else:
                blended = overlay_frame
            main_frame[pos_y : pos_y + overlay_h, pos_x : pos_x + overlay_w] = blended
            new_frames.append(main_frame)

        video.frames = np.array(new_frames, dtype=np.uint8)
        video.audio = self._handle_audio(video.audio, overlay, len(video.frames) / video.fps)
        return video

    def _handle_audio(self, main_audio: Audio, overlay: Video, target_duration: float) -> Audio:
        if self.audio_mode == "main":
            return main_audio.fit_to_duration(target_duration)
        if self.audio_mode == "overlay":
            return self._prepare_overlay_audio(overlay, target_duration)
        scaled_main = main_audio.scale_volume(self.audio_mix[0])
        overlay_audio = self._prepare_overlay_audio(overlay, target_duration)
        scaled_overlay = overlay_audio.scale_volume(self.audio_mix[1])
        if scaled_main.metadata.sample_rate != scaled_overlay.metadata.sample_rate:
            scaled_overlay = scaled_overlay.resample(scaled_main.metadata.sample_rate)
        return scaled_main.overlay(scaled_overlay, position=0.0)

    def _prepare_overlay_audio(self, overlay: Video, target_duration: float) -> Audio:
        from videopython.base.audio import Audio as _Audio

        overlay_audio = overlay.audio
        if overlay_audio is None or overlay_audio.is_silent:
            return _Audio.create_silent(target_duration, stereo=True, sample_rate=44100)
        if overlay_audio.metadata.duration_seconds < target_duration:
            loops = int(np.ceil(target_duration / overlay_audio.metadata.duration_seconds))
            looped = overlay_audio
            for _ in range(loops - 1):
                looped = looped.concat(overlay_audio)
            return looped.slice(0, target_duration)
        return overlay_audio.slice(0, target_duration)


class Reverse(Operation):
    """Plays the video backwards, with optional audio reversal.

    Args:
        reverse_audio: If true, reverse the audio track along with the video.
    """

    op: Literal["reverse"] = "reverse"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    reverse_audio: bool = True

    def apply(self, video: Video) -> Video:
        video.frames = video.frames[::-1].copy()
        if self.reverse_audio and video.audio is not None:
            video.audio.data = np.flip(video.audio.data, axis=0).copy()
        return video


class FreezeFrame(Operation):
    """Pauses video at a specific moment by holding a single frame.

    Args:
        timestamp: Time in seconds at which to capture the frame.
        duration: How long to hold the frozen frame, in seconds.
        position: 'after' / 'before' inserts frames; 'replace' swaps existing frames out.
    """

    op: Literal["freeze_frame"] = "freeze_frame"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    timestamp: float = Field(ge=0)
    duration: float = Field(2.0, gt=0)
    position: Literal["before", "after", "replace"] = "after"

    def apply(self, video: Video) -> Video:
        if self.timestamp >= video.total_seconds:
            raise ValueError(f"timestamp ({self.timestamp}) must be less than video duration ({video.total_seconds})")

        frame_idx = round(self.timestamp * video.fps)
        freeze_count = round(self.duration * video.fps)
        frozen = np.tile(video.frames[frame_idx : frame_idx + 1], (freeze_count, 1, 1, 1))

        if self.position == "after":
            insert_idx = frame_idx + 1
            video.frames = np.concatenate([video.frames[:insert_idx], frozen, video.frames[insert_idx:]], axis=0)
        elif self.position == "before":
            video.frames = np.concatenate([video.frames[:frame_idx], frozen, video.frames[frame_idx:]], axis=0)
        else:  # replace
            replace_end = min(frame_idx + freeze_count, len(video.frames))
            video.frames = np.concatenate([video.frames[:frame_idx], frozen, video.frames[replace_end:]], axis=0)

        if video.audio is not None:
            target_duration = len(video.frames) / video.fps
            sample_rate = video.audio.metadata.sample_rate
            channels = video.audio.metadata.channels
            silence_samples = round(self.duration * sample_rate)
            silence_shape = (silence_samples, channels) if channels > 1 else (silence_samples,)
            silence = np.zeros(silence_shape, dtype=np.float32)
            timestamp_sample = round(self.timestamp * sample_rate)

            if self.position == "after":
                insert_sample = min(timestamp_sample + round(sample_rate / video.fps), len(video.audio.data))
                video.audio.data = np.concatenate(
                    [video.audio.data[:insert_sample], silence, video.audio.data[insert_sample:]], axis=0
                )
            elif self.position == "before":
                video.audio.data = np.concatenate(
                    [video.audio.data[:timestamp_sample], silence, video.audio.data[timestamp_sample:]], axis=0
                )
            else:
                replace_end_sample = min(timestamp_sample + silence_samples, len(video.audio.data))
                video.audio.data = np.concatenate(
                    [video.audio.data[:timestamp_sample], silence, video.audio.data[replace_end_sample:]], axis=0
                )
            video.audio = video.audio.fit_to_duration(target_duration)
        return video

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        if self.timestamp >= meta.total_seconds:
            raise ValueError(f"timestamp ({self.timestamp}) must be less than video duration ({meta.total_seconds})")
        freeze_count = round(self.duration * meta.fps)
        if self.position in ("after", "before"):
            new_count = meta.frame_count + freeze_count
        else:  # replace
            frame_idx = round(self.timestamp * meta.fps)
            replace_end = min(frame_idx + freeze_count, meta.frame_count)
            new_count = meta.frame_count - (replace_end - frame_idx) + freeze_count
        from videopython.base.video import VideoMetadata as _Meta

        return _Meta(
            height=meta.height,
            width=meta.width,
            fps=meta.fps,
            frame_count=new_count,
            total_seconds=round(new_count / meta.fps, 4),
        )


class SilenceRemoval(Operation):
    """Cuts or fast-forwards through silent gaps between speech.

    Uses word-level transcription timestamps to identify silent sections and
    either removes them entirely or speeds them up.

    Args:
        min_silence_duration: Ignore silences shorter than this many seconds.
        padding: Seconds of breathing room around each speech boundary.
        mode: 'cut' removes silent sections entirely; 'speed_up' speeds them up.
        speed_factor: Speed multiplier for silent sections when mode='speed_up'.
    """

    op: Literal["silence_removal"] = "silence_removal"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    requires: ClassVar[tuple[str, ...]] = ("transcription",)

    min_silence_duration: float = Field(1.0, gt=0)
    padding: float = Field(0.15, ge=0)
    mode: Literal["cut", "speed_up"] = "cut"
    speed_factor: float = Field(3.0, gt=1.0)

    def _silence_ranges(self, words: list[Any], total_seconds: float) -> list[tuple[float, float]]:
        speech: list[tuple[float, float]] = []
        for word in words:
            start = max(0.0, word.start - self.padding)
            end = min(total_seconds, word.end + self.padding)
            if speech and start <= speech[-1][1]:
                speech[-1] = (speech[-1][0], max(speech[-1][1], end))
            else:
                speech.append((start, end))
        silences: list[tuple[float, float]] = []
        prev_end = 0.0
        for s_start, s_end in speech:
            if s_start - prev_end >= self.min_silence_duration:
                silences.append((prev_end, s_start))
            prev_end = s_end
        if total_seconds - prev_end >= self.min_silence_duration:
            silences.append((prev_end, total_seconds))
        return silences

    def apply(self, video: Video, transcription: Transcription | None = None) -> Video:  # type: ignore[override]
        if transcription is None:
            raise ValueError(
                "SilenceRemoval requires transcription data. "
                "Pass it via VideoEdit.run(context={'transcription': ...}) or directly to apply()."
            )
        words = transcription.words
        if not words:
            return video
        silences = self._silence_ranges(words, video.total_seconds)
        if not silences:
            return video
        if self.mode == "cut":
            return self._apply_cut(video, silences)
        return self._apply_speed_up(video, silences)

    def _apply_cut(self, video: Video, silence_ranges: list[tuple[float, float]]) -> Video:
        keep: list[tuple[int, int]] = []
        prev_frame = 0
        for s_start, s_end in silence_ranges:
            cut_start = round(s_start * video.fps)
            cut_end = round(s_end * video.fps)
            if cut_start > prev_frame:
                keep.append((prev_frame, cut_start))
            prev_frame = cut_end
        if prev_frame < len(video.frames):
            keep.append((prev_frame, len(video.frames)))
        if not keep:
            return video
        video.frames = np.concatenate([video.frames[s:e] for s, e in keep], axis=0)
        if video.audio is not None:
            sample_rate = video.audio.metadata.sample_rate
            chunks = []
            for start_f, end_f in keep:
                a_start = round((start_f / video.fps) * sample_rate)
                a_end = round((end_f / video.fps) * sample_rate)
                chunks.append(video.audio.data[a_start:a_end])
            video.audio.data = np.concatenate(chunks, axis=0)
            video.audio = video.audio.fit_to_duration(len(video.frames) / video.fps)
        return video

    def _apply_speed_up(self, video: Video, silence_ranges: list[tuple[float, float]]) -> Video:
        segments: list[tuple[int, int, float]] = []
        prev_frame = 0
        for s_start, s_end in silence_ranges:
            silence_start = round(s_start * video.fps)
            silence_end = round(s_end * video.fps)
            if silence_start > prev_frame:
                segments.append((prev_frame, silence_start, 1.0))
            segments.append((silence_start, silence_end, self.speed_factor))
            prev_frame = silence_end
        if prev_frame < len(video.frames):
            segments.append((prev_frame, len(video.frames), 1.0))

        frame_parts: list[np.ndarray] = []
        for start_f, end_f, speed in segments:
            n = end_f - start_f
            if n <= 0:
                continue
            if speed == 1.0:
                frame_parts.append(video.frames[start_f:end_f])
            else:
                target = max(1, round(n / speed))
                idx = np.linspace(start_f, end_f - 1, target).astype(int)
                frame_parts.append(video.frames[idx])
        if not frame_parts:
            return video
        video.frames = np.concatenate(frame_parts, axis=0)

        if video.audio is not None and not video.audio.is_silent:
            sample_rate = video.audio.metadata.sample_rate
            audio_parts: list[np.ndarray] = []
            for start_f, end_f, speed in segments:
                a_start = round((start_f / video.fps) * sample_rate)
                a_end = round((end_f / video.fps) * sample_rate)
                chunk = video.audio.data[a_start:a_end]
                if speed == 1.0 or len(chunk) == 0:
                    audio_parts.append(chunk)
                else:
                    target = max(1, round(len(chunk) / speed))
                    idx = np.linspace(0, len(chunk) - 1, target).astype(int)
                    audio_parts.append(chunk[idx])
            video.audio.data = np.concatenate(audio_parts, axis=0)
            video.audio = video.audio.fit_to_duration(len(video.frames) / video.fps)
        return video

    def predict_metadata(  # type: ignore[override]
        self,
        meta: VideoMetadata,
        transcription: Transcription | None = None,
    ) -> VideoMetadata:
        from videopython.base.video import VideoMetadata as _Meta

        identity = _Meta(
            height=meta.height,
            width=meta.width,
            fps=meta.fps,
            frame_count=meta.frame_count,
            total_seconds=meta.total_seconds,
        )
        if transcription is None or not getattr(transcription, "words", None):
            return identity
        silences = self._silence_ranges(transcription.words, meta.total_seconds)
        if not silences:
            return identity

        if self.mode == "cut":
            keep = 0
            prev_frame = 0
            for s_start, s_end in silences:
                cut_start = round(s_start * meta.fps)
                cut_end = round(s_end * meta.fps)
                if cut_start > prev_frame:
                    keep += cut_start - prev_frame
                prev_frame = cut_end
            if prev_frame < meta.frame_count:
                keep += meta.frame_count - prev_frame
            new_count = max(1, keep)
        else:
            saved = 0
            for s_start, s_end in silences:
                gap = round((s_end - s_start) * meta.fps)
                sped = max(1, round(gap / self.speed_factor))
                saved += gap - sped
            new_count = max(1, meta.frame_count - saved)

        return _Meta(
            height=meta.height,
            width=meta.width,
            fps=meta.fps,
            frame_count=new_count,
            total_seconds=round(new_count / meta.fps, 4),
        )

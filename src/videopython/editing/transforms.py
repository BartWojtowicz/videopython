"""Transform Operations.

A transform is any Operation that produces a new ``Video`` from a single
input video, free to change dimensions, fps, duration, or frame count.
See ``editing/operation.py`` for the ``Operation`` base.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import cv2
import numpy as np
from pydantic import Field, model_validator
from tqdm import tqdm

from videopython.base._dimensions import floor_to_even, round_to_even
from videopython.base.video import Video
from videopython.editing.operation import FilterCtx, OpCategory, Operation

if TYPE_CHECKING:
    from videopython.base.transcription import Transcription
    from videopython.base.video import VideoMetadata

logger = logging.getLogger(__name__)

__all__ = [
    "CutFrames",
    "CutSeconds",
    "Resize",
    "ResampleFPS",
    "Crop",
    "CropMode",
    "SpeedChange",
    "Reverse",
    "FreezeFrame",
    "SilenceRemoval",
]


class CutFrames(Operation):
    """Cuts video to a specific frame range."""

    op: Literal["cut_frames"] = "cut_frames"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    start: int = Field(ge=0, description="Start frame index (inclusive).")
    end: int = Field(ge=0, description="End frame index (exclusive).")

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
    """Cuts video to a specific time range in seconds."""

    op: Literal["cut"] = "cut"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    start: float = Field(ge=0, description="Start time in seconds.")
    end: float = Field(ge=0, description="End time in seconds.")

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
    """Resizes video to specified dimensions, preserving aspect ratio if only one dimension is given."""

    op: Literal["resize"] = "resize"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    streamable: ClassVar[bool] = True

    width: int | None = Field(None, gt=0, description="Target width in pixels, or None to maintain aspect ratio.")
    height: int | None = Field(None, gt=0, description="Target height in pixels, or None to maintain aspect ratio.")
    round_to_even: bool = Field(True, description="If True (default), snap output width/height to even numbers.")

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
            new_w = round_to_even(new_w)
            new_h = round_to_even(new_h)
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
    """Resamples video to a different frame rate, upsampling or downsampling as needed."""

    op: Literal["resample_fps"] = "resample_fps"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    streamable: ClassVar[bool] = True

    fps: float = Field(gt=0, description="Target frames per second.")

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
    example, ``width=0.5`` crops to 50% of the original width.
    """

    op: Literal["crop"] = "crop"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    streamable: ClassVar[bool] = True

    width: int | float = Field(description="Crop width in pixels (int) or fraction in (0, 1] of source width.")
    height: int | float = Field(description="Crop height in pixels (int) or fraction in (0, 1] of source height.")
    x: int | float = Field(0, description="Left edge X (only with mode='custom'). Pixels or fraction in [0, 1].")
    y: int | float = Field(0, description="Top edge Y (only with mode='custom'). Pixels or fraction in [0, 1].")
    mode: CropMode = Field(
        CropMode.CENTER, description="'center' crops from the middle, 'custom' uses x/y coordinates."
    )

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
        if self.mode == CropMode.CENTER:
            # Mirror apply()'s `mid - cw//2 : mid + cw//2` slice, which
            # produces 2 * (cw // 2) pixels — odd targets round down.
            cw = floor_to_even(cw)
            ch = floor_to_even(ch)
        return meta.with_dimensions(cw, ch)

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        cx, cy, cw, ch = self._resolve_box(ctx.width, ctx.height)
        return f"crop={cw}:{ch}:{cx}:{cy}"


class SpeedChange(Operation):
    """Speeds up or slows down video playback, optionally ramping between two speeds."""

    op: Literal["speed_change"] = "speed_change"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    speed: float = Field(gt=0, description="Playback speed multiplier. 2.0 = twice as fast, 0.5 = half speed.")
    end_speed: float | None = Field(
        None,
        gt=0,
        description="If set, smoothly ramp from speed to end_speed over the clip duration.",
    )
    interpolate: bool = Field(True, description="Blend between frames when slowing down for smoother motion.")
    adjust_audio: bool = Field(True, description="Time-stretch audio to match the new speed.")

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


class Reverse(Operation):
    """Plays the video backwards, with optional audio reversal."""

    op: Literal["reverse"] = "reverse"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    reverse_audio: bool = Field(True, description="If true, reverse the audio track along with the video.")

    def apply(self, video: Video) -> Video:
        video.frames = video.frames[::-1].copy()
        if self.reverse_audio and video.audio is not None:
            video.audio.data = np.flip(video.audio.data, axis=0).copy()
        return video


class FreezeFrame(Operation):
    """Pauses video at a specific moment by holding a single frame."""

    op: Literal["freeze_frame"] = "freeze_frame"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    timestamp: float = Field(ge=0, description="Time in seconds at which to capture the frame.")
    duration: float = Field(2.0, gt=0, description="How long to hold the frozen frame, in seconds.")
    position: Literal["before", "after", "replace"] = Field(
        "after",
        description="'after' / 'before' inserts frames; 'replace' swaps existing frames out.",
    )

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
    """

    op: Literal["silence_removal"] = "silence_removal"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    requires: ClassVar[tuple[str, ...]] = ("transcription",)

    min_silence_duration: float = Field(1.0, gt=0, description="Ignore silences shorter than this many seconds.")
    padding: float = Field(0.15, ge=0, description="Seconds of breathing room around each speech boundary.")
    mode: Literal["cut", "speed_up"] = Field(
        "cut",
        description="'cut' removes silent sections entirely; 'speed_up' speeds them up.",
    )
    speed_factor: float = Field(3.0, gt=1.0, description="Speed multiplier for silent sections when mode='speed_up'.")

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

    def apply(self, video: Video, transcription: Transcription | None = None) -> Video:
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

    def predict_metadata(
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

"""Transform Operations.

A transform is any Operation that produces a new ``Video`` from a single
input video, free to change dimensions, fps, duration, or frame count.
See ``editing/operation.py`` for the ``Operation`` base.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import cv2
import numpy as np
from pydantic import Field, model_validator
from tqdm import tqdm

from videopython.audio import Audio
from videopython.base._dimensions import floor_to_even, round_to_even
from videopython.base.exceptions import PlanError, PlanErrorCode, PlanValidationError
from videopython.base.video import Video
from videopython.editing.operation import BoundedTimeField, FilterCtx, OpCategory, Operation

if TYPE_CHECKING:
    from videopython.base.transcription import Transcription
    from videopython.base.video import VideoMetadata

logger = logging.getLogger(__name__)

# Shared tolerance (seconds) for duration bounds checks across the editing layer,
# so the located segment guard and the cut transforms accept the same boundary.
DURATION_EPS = 1e-3

__all__ = [
    "DURATION_EPS",
    "CutFrames",
    "CutSeconds",
    "Resize",
    "ResampleFPS",
    "Crop",
    "CropMode",
    "SpeedChange",
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
        # ints; eps inert -- DURATION_EPS is seconds-scale, never flips an int compare.
        if self.end > meta.frame_count + DURATION_EPS:
            message = f"end frame ({self.end}) exceeds frame count ({meta.frame_count})"
            raise PlanValidationError(
                message,
                [
                    PlanError(
                        code=PlanErrorCode.CUT_EXCEEDS_DURATION,
                        op=self.op,
                        field="end",
                        value=self.end,
                        limit=meta.frame_count,
                    )
                ],
            )
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
        if self.end > meta.total_seconds + DURATION_EPS:
            message = f"end time ({self.end}) exceeds video duration ({meta.total_seconds})"
            raise PlanValidationError(
                message,
                [
                    PlanError(
                        code=PlanErrorCode.CUT_EXCEEDS_DURATION,
                        op=self.op,
                        field="end",
                        value=self.end,
                        limit=meta.total_seconds,
                        predicted_duration=meta.total_seconds,
                    )
                ],
            )
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
            message = f"Crop {cw}x{ch} exceeds source {meta.width}x{meta.height}"
            raise PlanValidationError(
                message,
                [
                    PlanError(
                        code=PlanErrorCode.CROP_EXCEEDS_SOURCE,
                        op=self.op,
                        field="width" if cw > meta.width else "height",
                        value=float(cw if cw > meta.width else ch),
                        limit=float(meta.width if cw > meta.width else meta.height),
                    )
                ],
            )
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
    streamable: ClassVar[bool] = True
    changes_duration: ClassVar[bool] = True

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

    @property
    def _is_slow(self) -> bool:
        return self.speed < 1.0 or (self.end_speed is not None and self.end_speed < 1.0)

    def _eff_speed(self) -> float:
        return self.speed if self.end_speed is None else (self.speed + self.end_speed) / 2

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        """Compile to a ``setpts`` retime plus a CFR resampler.

        Constant speed: ``setpts=(PTS-STARTPTS)/k`` with a ``(k-1)/(2k)``-frame
        forward-bias correction so the ``fps`` filter's tick rounding selects
        the same nearest source frame a per-frame sampler would.

        Ramp: the eager sampler varies speed linearly across the *source*
        timeline and renormalizes so the output spans ``frame_count / avg``
        frames; the closed form of that curve is
        ``out(T) = D_out * ln(1 + (b-a)*T/(a*D_in)) / ln(b/a)``, evaluated
        per frame by ``setpts`` in double precision.

        With ``interpolate`` on a slowdown, the CFR resampler is the
        ``framerate`` filter (blends adjacent frames) instead of ``fps``
        (nearest), matching the eager path's frame blending in spirit -- the
        blend weighting is libavfilter's, not pixel-identical.
        """
        k = self.speed
        if self.end_speed is None or self.end_speed == self.speed:
            # Forward-bias so the fps filter's tick rounding picks the same
            # nearest source frame a per-frame sampler would. Speedups only:
            # for slowdowns the slot rounding already centers, and a negative
            # bias would retime head frames to negative PTS (dropped by the
            # resampler, shorting the predicted count by ~1/(2k) frames).
            bias = max(0.0, (k - 1) / (2 * k))
            retime = f"setpts=(PTS-STARTPTS)/{k:.10g}+{bias:.10g}/(FR*TB)"
        else:
            if ctx.frame_count <= 0:
                return None  # ramp needs the input duration; unknown -> not streamable
            a, b = self.speed, self.end_speed
            d_in = ctx.frame_count / ctx.fps
            d_out = self._new_frame_count(ctx.frame_count) / ctx.fps
            c_warp = (b - a) / (a * d_in)
            c_norm = d_out / math.log(b / a)
            retime = f"setpts='{c_norm:.10g}*log(1+{c_warp:.10g}*T)/TB'"
        if self.interpolate and self._is_slow:
            resample = f"framerate=fps={ctx.fps:.10g}"
        else:
            resample = f"fps={ctx.fps:.10g}"
        return f"{retime},{resample}"

    def transform_audio(self, audio: Audio, output_duration: float, fps: float, **_context: Any) -> Audio:
        """Time-stretch by the (average) speed, then fit the predicted duration.

        Ramps are approximated with a single constant stretch -- the same
        approximation the eager path has always used, so the two paths agree.
        """
        if not audio.is_silent and self.adjust_audio:
            audio = audio.time_stretch(self._eff_speed())
        return audio.fit_to_duration(output_duration)

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

        if self.interpolate and self._is_slow:
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
        if video.audio is not None:
            video.audio = self.transform_audio(video.audio, target_duration, video.fps)
        return video

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        new_count = self._new_frame_count(meta.frame_count)
        if new_count == 0:
            message = f"Speed {self.speed}x would result in 0 frames!"
            raise PlanValidationError(
                message,
                [
                    PlanError(
                        code=PlanErrorCode.DEGENERATE_DURATION,
                        op=self.op,
                        field="speed",
                        value=self.speed,
                    )
                ],
            )
        from videopython.base.video import VideoMetadata as _Meta

        return _Meta(
            height=meta.height,
            width=meta.width,
            fps=meta.fps,
            frame_count=new_count,
            total_seconds=round(new_count / meta.fps, 4),
        )


class FreezeFrame(Operation):
    """Pauses video at a specific moment by holding a single frame."""

    op: Literal["freeze_frame"] = "freeze_frame"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    streamable: ClassVar[bool] = True
    changes_duration: ClassVar[bool] = True
    # `timestamp` indexes a frame, so it must be strictly < the clip duration;
    # repair clamps an out-of-range value to the last frame.
    time_fields: ClassVar[tuple[BoundedTimeField, ...]] = (BoundedTimeField("timestamp", exclusive_end=True),)

    timestamp: float = Field(ge=0, description="Time in seconds at which to capture the frame.")
    duration: float = Field(2.0, gt=0, description="How long to hold the frozen frame, in seconds.")
    position: Literal["before", "after", "replace"] = Field(
        "after",
        description="'after' / 'before' inserts frames; 'replace' swaps existing frames out.",
    )

    def apply(self, video: Video) -> Video:
        if self.timestamp >= video.total_seconds:
            raise ValueError(f"timestamp ({self.timestamp}) must be less than video duration ({video.total_seconds})")

        frame_idx = min(round(self.timestamp * video.fps), len(video.frames) - 1)
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
            video.audio = self.transform_audio(video.audio, target_duration, video.fps)
        return video

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        """Compile to a linear ``loop``-based freeze chain.

        Insert modes (``after``/``before``): ``loop`` duplicates the held
        frame in place with continuous PTS -- the inserted copies are
        identical to the boundary frame, so both modes compile to the same
        chain. Replace mode stays linear too: ``loop`` adds the copies, a
        ``select`` drops the originals they replace (shifted behind the loop
        region), and ``setpts`` regenerates CFR timing. Needs the input
        frame count (``ctx.frame_count``); unknown -> not streamable.

        Raises the same out-of-range error as the eager path when
        ``timestamp`` lies past the clip end -- at compile, before decode.
        """
        if ctx.frame_count <= 0:
            return None
        input_duration = ctx.frame_count / ctx.fps
        if self.timestamp >= input_duration:
            raise ValueError(f"timestamp ({self.timestamp}) must be less than video duration ({input_duration})")
        frame_idx = min(round(self.timestamp * ctx.fps), ctx.frame_count - 1)
        freeze_count = round(self.duration * ctx.fps)
        if freeze_count == 0:
            return "null"
        # Every chain ends in its own CFR resampler: FrameIterator suppresses
        # its trailing fps= whenever any element starts with "fps=" (e.g. a
        # resample_fps op earlier in the plan), and without a resampler the
        # select/loop output re-duplicates frames at the rawvideo pipe.
        resample = f"fps={ctx.fps:.10g}"
        if self.position in ("after", "before"):
            return f"loop=loop={freeze_count}:size=1:start={frame_idx},setpts=N/FRAME_RATE/TB,{resample}"
        # replace: hold N frames of frame_idx while dropping the originals
        # they cover. `loop` adds N-1 copies (original + copies = N held);
        # the replaced originals sit right behind the loop region.
        replaced = min(freeze_count, ctx.frame_count - frame_idx)
        chain = f"loop=loop={freeze_count - 1}:size=1:start={frame_idx}"
        if replaced >= 2:
            drop_from = frame_idx + freeze_count
            drop_to = drop_from + replaced - 2
            chain += f",select='not(between(n,{drop_from},{drop_to}))'"
        return chain + f",setpts=N/FRAME_RATE/TB,{resample}"

    def transform_audio(self, audio: Audio, output_duration: float, fps: float, **_context: Any) -> Audio:
        """Insert the freeze's silence at the held position, then fit the duration."""
        sample_rate = audio.metadata.sample_rate
        channels = audio.metadata.channels
        silence_samples = round(self.duration * sample_rate)
        silence_shape = (silence_samples, channels) if channels > 1 else (silence_samples,)
        silence = np.zeros(silence_shape, dtype=np.float32)
        timestamp_sample = round(self.timestamp * sample_rate)

        if self.position == "after":
            insert_sample = min(timestamp_sample + round(sample_rate / fps), len(audio.data))
            data = np.concatenate([audio.data[:insert_sample], silence, audio.data[insert_sample:]], axis=0)
        elif self.position == "before":
            data = np.concatenate([audio.data[:timestamp_sample], silence, audio.data[timestamp_sample:]], axis=0)
        else:
            replace_end_sample = min(timestamp_sample + silence_samples, len(audio.data))
            data = np.concatenate([audio.data[:timestamp_sample], silence, audio.data[replace_end_sample:]], axis=0)
        # Rebuild metadata from the new sample count: fit_to_duration trusts
        # metadata.duration_seconds, so a raw .data mutation would leave it
        # stale and make the fit pad the insertion a second time.
        metadata = dataclasses.replace(
            audio.metadata,
            duration_seconds=data.shape[0] / sample_rate,
            frame_count=data.shape[0],
        )
        return Audio(data, metadata).fit_to_duration(output_duration)

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        if self.timestamp >= meta.total_seconds:
            message = f"timestamp ({self.timestamp}) must be less than video duration ({meta.total_seconds})"
            raise PlanValidationError(
                message,
                [
                    PlanError(
                        code=PlanErrorCode.OP_TIMESTAMP_OUT_OF_RANGE,
                        op=self.op,
                        field="timestamp",
                        value=self.timestamp,
                        limit=meta.total_seconds,
                        predicted_duration=meta.total_seconds,
                    )
                ],
            )
        freeze_count = round(self.duration * meta.fps)
        if self.position in ("after", "before"):
            new_count = meta.frame_count + freeze_count
        else:  # replace
            frame_idx = min(round(self.timestamp * meta.fps), meta.frame_count - 1)
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
    """Cuts silent gaps between speech, using word-level transcription timestamps.

    Compiles to a ``select``/``aselect``-style keep-window cut on the
    streaming path: the transcription is consumed at plan-compile time and
    the silent frame ranges are dropped by the decoder's filter chain.
    """

    op: Literal["silence_removal"] = "silence_removal"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    streamable: ClassVar[bool] = True
    changes_duration: ClassVar[bool] = True
    requires: ClassVar[tuple[str, ...]] = ("transcription",)

    min_silence_duration: float = Field(1.0, gt=0, description="Ignore silences shorter than this many seconds.")
    padding: float = Field(0.15, ge=0, description="Seconds of breathing room around each speech boundary.")

    _MISSING_CONTEXT = (
        "SilenceRemoval requires transcription data. "
        "Pass it via VideoEdit.run(context={'transcription': ...}) or directly to apply()."
    )

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

    def _keep_frame_ranges(
        self, transcription: Transcription, total_seconds: float, fps: float, n_frames: int
    ) -> list[tuple[int, int]] | None:
        """Frame ranges to keep, or ``None`` for "nothing to cut" (identity).

        The single source of the cut math, shared by the eager apply, the
        filter compile, the audio twin, and ``predict_metadata`` -- all four
        must agree on the output timeline.
        """
        words = transcription.words
        if not words:
            return None
        silences = self._silence_ranges(words, total_seconds)
        if not silences:
            return None
        keep: list[tuple[int, int]] = []
        prev_frame = 0
        for s_start, s_end in silences:
            cut_start = round(s_start * fps)
            cut_end = round(s_end * fps)
            if cut_start > prev_frame:
                keep.append((prev_frame, cut_start))
            prev_frame = cut_end
        if prev_frame < n_frames:
            keep.append((prev_frame, n_frames))
        return keep or None

    def apply(self, video: Video, transcription: Transcription | None = None) -> Video:
        if transcription is None:
            raise ValueError(self._MISSING_CONTEXT)
        keep = self._keep_frame_ranges(transcription, video.total_seconds, video.fps, len(video.frames))
        if keep is None:
            return video
        video.frames = np.concatenate([video.frames[s:e] for s, e in keep], axis=0)
        if video.audio is not None:
            video.audio = self.transform_audio(
                video.audio, len(video.frames) / video.fps, video.fps, transcription=transcription
            )
        return video

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        """Compile the cut to a ``select`` keep-window filter.

        Consumes the segment-local transcription from ``ctx.context``;
        missing context raises the op's clear error at plan compile, before
        any decode. No silences -> ``null`` (identity).
        """
        from videopython.base.transcription import Transcription as _Transcription

        transcription = ctx.context.get("transcription")
        if not isinstance(transcription, _Transcription):
            raise ValueError(self._MISSING_CONTEXT)
        if ctx.frame_count <= 0:
            return None
        keep = self._keep_frame_ranges(transcription, ctx.frame_count / ctx.fps, ctx.fps, ctx.frame_count)
        if keep is None:
            return "null"
        terms = "+".join(f"between(n,{s},{e - 1})" for s, e in keep)
        # Trailing resampler: see FreezeFrame.to_ffmpeg_filter.
        return f"select='{terms}',setpts=N/FRAME_RATE/TB,fps={ctx.fps:.10g}"

    def transform_audio(self, audio: Audio, output_duration: float, fps: float, **context: Any) -> Audio:
        """Cut the same keep windows out of the audio, then fit the duration."""
        transcription = context.get("transcription")
        if transcription is None:
            raise ValueError(self._MISSING_CONTEXT)
        total_seconds = audio.metadata.duration_seconds
        keep = self._keep_frame_ranges(transcription, total_seconds, fps, round(total_seconds * fps))
        if keep is None:
            return audio
        sample_rate = audio.metadata.sample_rate
        chunks = [audio.data[round(s / fps * sample_rate) : round(e / fps * sample_rate)] for s, e in keep]
        data = np.concatenate(chunks, axis=0)
        metadata = dataclasses.replace(
            audio.metadata,
            duration_seconds=data.shape[0] / sample_rate,
            frame_count=data.shape[0],
        )
        return Audio(data, metadata).fit_to_duration(output_duration)

    def predict_metadata(self, meta: VideoMetadata, transcription: Transcription | None = None) -> VideoMetadata:
        """Predict the cut duration; identity when no transcription is in the
        validate context (the same conditional guarantee as time re-basing)."""
        if transcription is None:
            return meta
        keep = self._keep_frame_ranges(transcription, meta.total_seconds, meta.fps, meta.frame_count)
        if keep is None:
            return meta
        new_count = sum(e - s for s, e in keep)
        from videopython.base.video import VideoMetadata as _Meta

        return _Meta(
            height=meta.height,
            width=meta.width,
            fps=meta.fps,
            frame_count=new_count,
            total_seconds=round(new_count / meta.fps, 4),
        )

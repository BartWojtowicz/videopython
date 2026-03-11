from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from videopython.base.progress import log, progress_iter
from videopython.base.video import Video, _round_dimension_to_even

if TYPE_CHECKING:
    from videopython.base.audio import Audio
    from videopython.base.text.transcription import Transcription

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


class Transformation(ABC):
    """Abstract class for transformation on frames of video."""

    @abstractmethod
    def apply(self, video: Video) -> Video:
        pass


class CutFrames(Transformation):
    """Cuts video to a specific frame range."""

    def __init__(self, start: int, end: int):
        """Initialize frame cutter.

        Args:
            start: Start frame index (inclusive).
            end: End frame index (exclusive).
        """
        self.start = start
        self.end = end

    def apply(self, video: Video) -> Video:
        """Apply frame cut to video.

        Args:
            video: Input video.

        Returns:
            Video with frames from start to end.
        """
        video = video[self.start : self.end]
        return video


class CutSeconds(Transformation):
    """Cuts video to a specific time range in seconds."""

    def __init__(self, start: float | int, end: float | int):
        """Initialize time-based cutter.

        Args:
            start: Start time in seconds.
            end: End time in seconds.
        """
        self.start = start
        self.end = end

    def apply(self, video: Video) -> Video:
        """Apply time-based cut to video.

        Args:
            video: Input video.

        Returns:
            Video cut from start to end seconds.
        """
        video = video[round(self.start * video.fps) : round(self.end * video.fps)]
        return video


class Resize(Transformation):
    """Resizes video to specified dimensions, maintaining aspect ratio if only one dimension is provided."""

    def __init__(self, width: int | None = None, height: int | None = None, round_to_even: bool = True):
        """Initialize resizer.

        Args:
            width: Target width in pixels, or None to maintain aspect ratio.
            height: Target height in pixels, or None to maintain aspect ratio.
            round_to_even: If True (default), snap output width/height to even numbers.
        """
        self.width = width
        self.height = height
        self.round_to_even = round_to_even
        if width is None and height is None:
            raise ValueError("You must provide either `width` or `height`!")

    def _resize_frame(self, frame: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
        return cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def apply(self, video: Video) -> Video:
        """Resize video frames to target dimensions.

        Args:
            video: Input video.

        Returns:
            Resized video.
        """
        if self.width and self.height:
            new_height = self.height
            new_width = self.width
        elif self.height is None and self.width:
            video_height = video.video_shape[1]
            video_width = video.video_shape[2]
            new_height = round(video_height * (self.width / video_width))
            new_width = self.width
        elif self.width is None and self.height:
            video_height = video.video_shape[1]
            video_width = video.video_shape[2]
            new_width = round(video_width * (self.height / video_height))
            new_height = self.height

        if self.round_to_even:
            new_width = _round_dimension_to_even(new_width)
            new_height = _round_dimension_to_even(new_height)

        log(f"Resizing video to: {new_width}x{new_height}!")
        video.frames = np.asarray(
            [self._resize_frame(frame, new_width, new_height) for frame in video.frames],
            dtype=np.uint8,
        )
        return video


class ResampleFPS(Transformation):
    """Resamples video to a different frame rate, upsampling or downsampling as needed."""

    def __init__(self, fps: int | float):
        """Initialize FPS resampler.

        Args:
            fps: Target frames per second.
        """
        self.fps = float(fps)

    def _downsample(self, video: Video) -> Video:
        target_frame_count = int(len(video.frames) * (self.fps / video.fps))
        new_frame_indices = np.round(np.linspace(0, len(video.frames) - 1, target_frame_count)).astype(int)
        video.frames = video.frames[new_frame_indices]
        video.fps = self.fps
        return video

    def _upsample(self, video: Video) -> Video:
        target_frame_count = int(len(video.frames) * (self.fps / video.fps))
        new_frame_indices = np.linspace(0, len(video.frames) - 1, target_frame_count)
        new_frames = []
        for i in progress_iter(range(len(new_frame_indices)), desc="Interpolating frames"):
            # Interpolate between the two nearest frames
            ratio = new_frame_indices[i] % 1
            new_frame = (1 - ratio) * video.frames[int(new_frame_indices[i])] + ratio * video.frames[
                int(np.ceil(new_frame_indices[i]))
            ]
            new_frames.append(new_frame.astype(np.uint8))
        video.frames = np.array(new_frames, dtype=np.uint8)
        video.fps = self.fps
        return video

    def apply(self, video: Video) -> Video:
        """Resample video to target FPS.

        Args:
            video: Input video.

        Returns:
            Video with target frame rate.
        """
        if video.fps == self.fps:
            return video
        elif video.fps > self.fps:
            log(f"Downsampling video from {video.fps} to {self.fps} FPS.")
            video = self._downsample(video)
        else:
            log(f"Upsampling video from {video.fps} to {self.fps} FPS.")
            video = self._upsample(video)
        if video.audio is not None:
            target_duration = len(video.frames) / video.fps
            video.audio = video.audio.fit_to_duration(target_duration)
        return video


class CropMode(Enum):
    CENTER = "center"
    CUSTOM = "custom"  # Use x, y coordinates for positioning


class Crop(Transformation):
    """Crops the frame to a smaller region.

    Accepts pixel values (int) or normalized 0-1 fractions (float). For
    example, width=0.5 crops to 50% of the original width.
    """

    def __init__(
        self,
        width: int | float,
        height: int | float,
        x: int | float = 0,
        y: int | float = 0,
        mode: CropMode = CropMode.CENTER,
    ):
        """Initialize cropper.

        Args:
            width: Crop width. Pass an int for pixels, or a float in (0, 1]
                as a fraction of the video width.
            height: Crop height. Pass an int for pixels, or a float in (0, 1]
                as a fraction of the video height.
            x: Left edge X position. Only used when mode is "custom". Pass an
                int for pixels or a float in [0, 1] for a fraction.
            y: Top edge Y position. Only used when mode is "custom". Pass an
                int for pixels or a float in [0, 1] for a fraction.
            mode: "center" crops from the middle of the frame, "custom" uses
                the x/y coordinates you provide.
        """
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.mode = mode

    def _to_pixels(self, value: int | float, dimension: int) -> int:
        """Convert value to pixels. Floats in (0, 1] are treated as normalized."""
        if isinstance(value, float) and 0 < value <= 1:
            return int(value * dimension)
        return int(value)

    def apply(self, video: Video) -> Video:
        """Crop video to target dimensions.

        Args:
            video: Input video.

        Returns:
            Cropped video.
        """
        current_height, current_width = video.frame_shape[:2]

        # Convert to pixels (handles both int and normalized float)
        crop_width = self._to_pixels(self.width, current_width)
        crop_height = self._to_pixels(self.height, current_height)
        crop_x = self._to_pixels(self.x, current_width)
        crop_y = self._to_pixels(self.y, current_height)

        if self.mode == CropMode.CENTER:
            center_height = current_height // 2
            center_width = current_width // 2
            width_offset = crop_width // 2
            height_offset = crop_height // 2
            video.frames = video.frames[
                :,
                center_height - height_offset : center_height + height_offset,
                center_width - width_offset : center_width + width_offset,
                :,
            ]
        else:
            # Custom position crop using x, y coordinates
            video.frames = video.frames[
                :,
                crop_y : crop_y + crop_height,
                crop_x : crop_x + crop_width,
                :,
            ]
        return video


class SpeedChange(Transformation):
    """Speeds up or slows down video playback.

    Values above 1.0 speed up (2.0 = twice as fast), values below 1.0 slow
    down (0.5 = half speed). Can also smoothly ramp between two speeds.
    """

    def __init__(
        self,
        speed: float,
        end_speed: float | None = None,
        interpolate: bool = True,
        adjust_audio: bool = True,
    ):
        """Initialize speed changer.

        Args:
            speed: Playback speed multiplier. 2.0 = twice as fast, 0.5 = half
                speed, 1.0 = original speed.
            end_speed: If set, smoothly ramp from speed to end_speed over the
                clip duration. Omit for constant speed.
            interpolate: Blend between frames when slowing down for smoother
                motion. Disable for a choppy/stylistic look.
            adjust_audio: Time-stretch audio to match the new speed. If false,
                audio is sliced or padded instead (no pitch correction).
        """
        if speed <= 0:
            raise ValueError("Speed must be positive!")
        if end_speed is not None and end_speed <= 0:
            raise ValueError("End speed must be positive!")

        self.speed = speed
        self.end_speed = end_speed
        self.interpolate = interpolate
        self.adjust_audio = adjust_audio

    def apply(self, video: Video) -> Video:
        """Apply speed change to video.

        Args:
            video: Input video.

        Returns:
            Video with adjusted speed.
        """
        n_frames = len(video.frames)

        if self.end_speed is None:
            # Constant speed change
            log(f"Applying {self.speed}x speed change...")
            new_frame_count = int(n_frames / self.speed)

            if new_frame_count == 0:
                raise ValueError(f"Speed {self.speed}x would result in 0 frames!")

            # Map new frame indices to original frame positions
            source_indices = np.linspace(0, n_frames - 1, new_frame_count)
        else:
            # Speed ramp: smoothly transition from speed to end_speed
            log(f"Applying speed ramp from {self.speed}x to {self.end_speed}x...")

            # Calculate frame positions with varying speed
            # Use cumulative sum of inverse speeds to get source positions
            num_samples = 1000  # High resolution for smooth ramp
            speeds = np.linspace(self.speed, self.end_speed, num_samples)
            # Time spent at each sample point (inverse of speed)
            time_per_sample = 1.0 / speeds
            cumulative_time = np.cumsum(time_per_sample)
            cumulative_time = cumulative_time / cumulative_time[-1]  # Normalize to [0, 1]

            # Total output duration based on average speed
            avg_speed = (self.speed + self.end_speed) / 2
            new_frame_count = int(n_frames / avg_speed)

            if new_frame_count == 0:
                raise ValueError("Speed ramp would result in 0 frames!")

            # Map output frames to source positions using the cumulative time curve
            output_positions = np.linspace(0, 1, new_frame_count)
            source_positions = np.interp(output_positions, cumulative_time, np.linspace(0, 1, num_samples))
            source_indices = source_positions * (n_frames - 1)

        # Generate new frames
        if self.interpolate and (self.speed < 1.0 or (self.end_speed and self.end_speed < 1.0)):
            # Interpolate for smoother slow motion
            new_frames = []
            for idx in progress_iter(source_indices, desc="Interpolating frames"):
                idx_low = int(idx)
                idx_high = min(idx_low + 1, len(video.frames) - 1)
                ratio = idx - idx_low
                if ratio == 0 or idx_low == idx_high:
                    new_frames.append(video.frames[idx_low])
                else:
                    interpolated = (1 - ratio) * video.frames[idx_low] + ratio * video.frames[idx_high]
                    new_frames.append(interpolated.astype(np.uint8))
            video.frames = np.array(new_frames, dtype=np.uint8)
        else:
            # Simple frame selection (faster, good for speedup)
            frame_indices = np.round(source_indices).astype(int)
            frame_indices = np.clip(frame_indices, 0, n_frames - 1)
            video.frames = video.frames[frame_indices]

        # Handle audio adjustment
        target_duration = len(video.frames) / video.fps

        if video.audio is not None and not video.audio.is_silent:
            if self.adjust_audio:
                # Time-stretch audio to match video speed
                effective_speed = self.speed if self.end_speed is None else (self.speed + self.end_speed) / 2
                video.audio = video.audio.time_stretch(effective_speed)

            # Ensure audio duration matches video duration
            video.audio = video.audio.fit_to_duration(target_duration)
        elif video.audio is not None:
            # Silent audio - just adjust duration
            video.audio = video.audio.fit_to_duration(target_duration)

        return video


class PictureInPicture(Transformation):
    """Places a smaller video on top of the main video (picture-in-picture).

    Useful for reaction videos, facecam overlays, tutorials with presenter
    inset, and side-by-side commentary.
    """

    def __init__(
        self,
        overlay: Video,
        position: tuple[float, float] = (0.7, 0.7),
        scale: float = 0.25,
        border_width: int = 0,
        border_color: tuple[int, int, int] = (255, 255, 255),
        corner_radius: int = 0,
        opacity: float = 1.0,
        audio_mode: Literal["main", "overlay", "mix"] = "main",
        audio_mix: tuple[float, float] = (1.0, 1.0),
    ):
        """Initialize picture-in-picture transform.

        Args:
            overlay: The video to show as the small inset.
            position: Center of the inset as normalized (x, y). (0, 0) = top-left
                corner, (1, 1) = bottom-right corner.
            scale: Inset width as a fraction of the main video width.
                0.25 = 25% of main width.
            border_width: Border thickness in pixels. 0 = no border.
            border_color: Border color as [R, G, B], each 0-255.
            corner_radius: Rounded corner radius in pixels. 0 = square corners.
            opacity: Inset transparency. 0 = invisible, 1 = fully opaque.
            audio_mode: "main" keeps only the main audio, "overlay" uses only
                the inset audio, "mix" blends both tracks together.
            audio_mix: Volume levels as [main, overlay] when audio_mode is "mix".
                1.0 = original level. Values above 1.0 amplify (may clip).
        """
        if not 0 <= position[0] <= 1 or not 0 <= position[1] <= 1:
            raise ValueError("Position must be normalized values in range [0, 1]!")
        if not 0 < scale <= 1:
            raise ValueError("Scale must be in range (0, 1]!")
        if border_width < 0:
            raise ValueError("Border width must be non-negative!")
        if corner_radius < 0:
            raise ValueError("Corner radius must be non-negative!")
        if not 0 <= opacity <= 1:
            raise ValueError("Opacity must be in range [0, 1]!")
        if audio_mode not in ("main", "overlay", "mix"):
            raise ValueError(f"audio_mode must be 'main', 'overlay', or 'mix', got '{audio_mode}'")
        if audio_mix[0] < 0 or audio_mix[1] < 0:
            raise ValueError("audio_mix factors must be non-negative")

        self.overlay = overlay
        self.position = position
        self.scale = scale
        self.border_width = border_width
        self.border_color = border_color
        self.corner_radius = corner_radius
        self.opacity = opacity
        self.audio_mode = audio_mode
        self.audio_mix = audio_mix

    def _create_rounded_mask(self, width: int, height: int) -> np.ndarray:
        """Create a mask with rounded corners."""
        if self.corner_radius <= 0:
            return np.ones((height, width), dtype=np.float32)

        radius = min(self.corner_radius, width // 2, height // 2)
        mask = np.ones((height, width), dtype=np.float32)

        # Vectorized equivalent of the previous per-pixel corner loop.
        corners = [
            (0, radius, 0, radius, radius, radius),  # top-left
            (0, radius, width - radius, width, width - radius, radius),  # top-right
            (height - radius, height, 0, radius, radius, height - radius),  # bottom-left
            (height - radius, height, width - radius, width, width - radius, height - radius),  # bottom-right
        ]
        radius_sq = float(radius * radius)
        for y_start, y_end, x_start, x_end, center_x, center_y in corners:
            yy, xx = np.ogrid[y_start:y_end, x_start:x_end]
            outside = ((xx - center_x) ** 2 + (yy - center_y) ** 2) > radius_sq
            mask[y_start:y_end, x_start:x_end][outside] = 0.0

        return mask

    def _add_border(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add border around the overlay frame."""
        if self.border_width <= 0:
            return frame

        h, w = frame.shape[:2]
        bordered = frame.copy()

        # Create border by drawing on edges
        border_color = np.array(self.border_color, dtype=np.uint8)

        # Top and bottom borders
        bordered[: self.border_width, :] = border_color
        bordered[-self.border_width :, :] = border_color

        # Left and right borders
        bordered[:, : self.border_width] = border_color
        bordered[:, -self.border_width :] = border_color

        # If we have rounded corners, apply mask to border too
        if self.corner_radius > 0:
            bordered = bordered * mask[:, :, None].astype(np.uint8)

        return bordered

    def apply(self, video: Video) -> Video:
        """Apply picture-in-picture overlay to video.

        Args:
            video: Main video to apply overlay on.

        Returns:
            Video with overlay applied.
        """
        main_h, main_w = video.frame_shape[:2]
        n_main_frames = len(video.frames)
        n_overlay_frames = len(self.overlay.frames)

        # Calculate overlay dimensions
        overlay_w = int(main_w * self.scale)
        # Maintain aspect ratio of overlay video
        overlay_aspect = self.overlay.metadata.width / self.overlay.metadata.height
        overlay_h = int(overlay_w / overlay_aspect)

        # Calculate position (position is center of overlay)
        pos_x = int(self.position[0] * main_w - overlay_w / 2)
        pos_y = int(self.position[1] * main_h - overlay_h / 2)

        # Clamp position to keep overlay within bounds
        pos_x = max(0, min(pos_x, main_w - overlay_w))
        pos_y = max(0, min(pos_y, main_h - overlay_h))

        # Resize overlay frames once
        log(f"Resizing overlay to {overlay_w}x{overlay_h}...")
        resized_overlay_frames = np.asarray(
            [
                cv2.resize(frame, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
                for frame in progress_iter(self.overlay.frames, desc="Resizing overlay")
            ],
            dtype=np.uint8,
        )

        # Create rounded corner mask if needed
        mask = self._create_rounded_mask(overlay_w, overlay_h)
        alpha = mask[:, :, None] * self.opacity

        # Apply border to overlay frames
        if self.border_width > 0:
            resized_overlay_frames = np.asarray(
                [self._add_border(frame, mask) for frame in resized_overlay_frames],
                dtype=np.uint8,
            )

        log("Applying picture-in-picture...")
        new_frames = []
        for i in progress_iter(range(n_main_frames), desc="Picture-in-picture"):
            main_frame = video.frames[i].copy()

            # Get overlay frame (loop if overlay is shorter)
            overlay_idx = i % n_overlay_frames
            overlay_frame = resized_overlay_frames[overlay_idx]

            # Extract the region where overlay will be placed
            region = main_frame[pos_y : pos_y + overlay_h, pos_x : pos_x + overlay_w]

            # Apply mask and opacity
            if self.corner_radius > 0 or self.opacity < 1.0:
                # Blend with mask and opacity
                blended = (overlay_frame * alpha + region * (1 - alpha)).astype(np.uint8)
            else:
                blended = overlay_frame

            # Place the blended overlay back
            main_frame[pos_y : pos_y + overlay_h, pos_x : pos_x + overlay_w] = blended
            new_frames.append(main_frame)

        video.frames = np.array(new_frames, dtype=np.uint8)

        # Handle audio based on audio_mode
        main_duration = len(video.frames) / video.fps
        video.audio = self._handle_audio(video.audio, main_duration)

        return video

    def _handle_audio(self, main_audio: "Audio", target_duration: float) -> "Audio":
        """Handle audio based on audio_mode setting."""

        if self.audio_mode == "main":
            # Keep main video audio (current behavior)
            return main_audio.fit_to_duration(target_duration)

        elif self.audio_mode == "overlay":
            # Use only overlay video audio (looped if necessary)
            return self._prepare_overlay_audio(target_duration)

        else:  # mix
            # Mix both audio tracks with volume factors
            scaled_main = main_audio.scale_volume(self.audio_mix[0])
            overlay_audio = self._prepare_overlay_audio(target_duration)
            scaled_overlay = overlay_audio.scale_volume(self.audio_mix[1])

            # Ensure sample rates match
            if scaled_main.metadata.sample_rate != scaled_overlay.metadata.sample_rate:
                scaled_overlay = scaled_overlay.resample(scaled_main.metadata.sample_rate)

            # Mix using overlay at position 0
            return scaled_main.overlay(scaled_overlay, position=0.0)

    def _prepare_overlay_audio(self, target_duration: float) -> "Audio":
        """Prepare overlay audio, looping if shorter than target duration."""
        from videopython.base.audio import Audio

        overlay_audio = self.overlay.audio

        if overlay_audio is None or overlay_audio.is_silent:
            return Audio.create_silent(
                target_duration,
                stereo=True,
                sample_rate=44100,
            )

        # Loop overlay audio if shorter than main video
        if overlay_audio.metadata.duration_seconds < target_duration:
            # Calculate how many times to loop
            loops_needed = int(np.ceil(target_duration / overlay_audio.metadata.duration_seconds))

            # Concatenate copies
            looped_audio = overlay_audio
            for _ in range(loops_needed - 1):
                looped_audio = looped_audio.concat(overlay_audio)

            # Slice to exact duration
            return looped_audio.slice(0, target_duration)
        else:
            # Slice to match main video duration
            return overlay_audio.slice(0, target_duration)


class Reverse(Transformation):
    """Plays the video backwards, with optional audio reversal."""

    def __init__(self, reverse_audio: bool = True):
        """Initialize reverse transform.

        Args:
            reverse_audio: If true, reverse the audio track along with the
                video. Set to false to keep original audio over reversed footage.
        """
        self.reverse_audio = reverse_audio

    def apply(self, video: Video) -> Video:
        video.frames = video.frames[::-1].copy()
        if self.reverse_audio and video.audio is not None:
            video.audio.data = np.flip(video.audio.data, axis=0).copy()
        return video


class FreezeFrame(Transformation):
    """Pauses video at a specific moment by holding a single frame.

    The frozen frame is inserted into the timeline, making the video longer
    (or replacing existing frames in "replace" mode).
    """

    def __init__(
        self,
        timestamp: float,
        duration: float = 2.0,
        position: Literal["before", "after", "replace"] = "after",
    ):
        """Initialize freeze frame transform.

        Args:
            timestamp: Time in seconds at which to capture the frame.
            duration: How long to hold the frozen frame, in seconds.
            position: Where to place the frozen frames. "after" inserts them
                after the timestamp, "before" inserts them before it,
                "replace" swaps out existing video at that point.
        """
        if timestamp < 0:
            raise ValueError(f"timestamp must be >= 0, got {timestamp}")
        if duration <= 0:
            raise ValueError(f"duration must be > 0, got {duration}")
        self.timestamp = timestamp
        self.duration = duration
        self.position = position

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
        elif self.position == "replace":
            replace_end = min(frame_idx + freeze_count, len(video.frames))
            video.frames = np.concatenate([video.frames[:frame_idx], frozen, video.frames[replace_end:]], axis=0)

        # Rebuild audio to match new video duration
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
            elif self.position == "replace":
                replace_end_sample = min(timestamp_sample + silence_samples, len(video.audio.data))
                video.audio.data = np.concatenate(
                    [video.audio.data[:timestamp_sample], silence, video.audio.data[replace_end_sample:]], axis=0
                )

            video.audio = video.audio.fit_to_duration(target_duration)

        return video


class SilenceRemoval(Transformation):
    """Cuts or fast-forwards through silent gaps between speech.

    Uses word-level transcription timestamps to identify silent sections
    and either removes them entirely or speeds them up. Requires a
    transcription to be available.
    """

    def __init__(
        self,
        min_silence_duration: float = 1.0,
        padding: float = 0.15,
        mode: Literal["cut", "speed_up"] = "cut",
        speed_factor: float = 3.0,
    ):
        """Initialize silence removal transform.

        Args:
            min_silence_duration: Ignore silences shorter than this many
                seconds. Keeps natural pauses intact.
            padding: Seconds of breathing room to keep around each speech
                boundary so cuts don't feel abrupt.
            mode: "cut" removes silent sections entirely, "speed_up" plays
                them at a faster speed instead.
            speed_factor: How fast to play silent sections when mode is
                "speed_up". 3.0 = three times normal speed.
        """
        if min_silence_duration <= 0:
            raise ValueError(f"min_silence_duration must be > 0, got {min_silence_duration}")
        if padding < 0:
            raise ValueError(f"padding must be >= 0, got {padding}")
        if speed_factor <= 1.0:
            raise ValueError(f"speed_factor must be > 1.0, got {speed_factor}")
        self.min_silence_duration = min_silence_duration
        self.padding = padding
        self.mode = mode
        self.speed_factor = speed_factor

    def apply(self, video: Video, transcription: Transcription | None = None) -> Video:  # type: ignore[override]
        """Apply silence removal to video.

        Args:
            video: Input video.
            transcription: Word-level transcription (Transcription object).
                Required -- raises ValueError if not provided.
        """
        if transcription is None:
            raise ValueError(
                "SilenceRemoval requires transcription data. "
                "Pass it via VideoEdit.run(context={'transcription': ...}) or directly to apply()."
            )

        words = transcription.words
        if not words:
            return video

        # Build speech ranges from word timestamps (with padding)
        speech_ranges: list[tuple[float, float]] = []
        for word in words:
            start = max(0, word.start - self.padding)
            end = min(video.total_seconds, word.end + self.padding)
            if speech_ranges and start <= speech_ranges[-1][1]:
                speech_ranges[-1] = (speech_ranges[-1][0], max(speech_ranges[-1][1], end))
            else:
                speech_ranges.append((start, end))

        # Identify silence gaps
        silence_ranges: list[tuple[float, float]] = []
        prev_end = 0.0
        for s_start, s_end in speech_ranges:
            if s_start - prev_end >= self.min_silence_duration:
                silence_ranges.append((prev_end, s_start))
            prev_end = s_end
        # Trailing silence
        if video.total_seconds - prev_end >= self.min_silence_duration:
            silence_ranges.append((prev_end, video.total_seconds))

        if not silence_ranges:
            return video

        if self.mode == "cut":
            return self._apply_cut(video, silence_ranges)
        else:
            return self._apply_speed_up(video, silence_ranges)

    def _apply_cut(self, video: Video, silence_ranges: list[tuple[float, float]]) -> Video:
        """Cut silent sections out of the video."""
        keep_ranges: list[tuple[int, int]] = []
        prev_frame = 0
        for s_start, s_end in silence_ranges:
            cut_start = round(s_start * video.fps)
            cut_end = round(s_end * video.fps)
            if cut_start > prev_frame:
                keep_ranges.append((prev_frame, cut_start))
            prev_frame = cut_end
        if prev_frame < len(video.frames):
            keep_ranges.append((prev_frame, len(video.frames)))

        if not keep_ranges:
            return video

        frame_segments = [video.frames[start:end] for start, end in keep_ranges]
        video.frames = np.concatenate(frame_segments, axis=0)

        if video.audio is not None:
            sample_rate = video.audio.metadata.sample_rate
            audio_segments = []
            for start_f, end_f in keep_ranges:
                a_start = round((start_f / video.fps) * sample_rate)
                a_end = round((end_f / video.fps) * sample_rate)
                audio_segments.append(video.audio.data[a_start:a_end])
            video.audio.data = np.concatenate(audio_segments, axis=0)
            video.audio = video.audio.fit_to_duration(len(video.frames) / video.fps)

        return video

    def _apply_speed_up(self, video: Video, silence_ranges: list[tuple[float, float]]) -> Video:
        """Speed up silent sections instead of cutting them."""
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
            n_frames = end_f - start_f
            if n_frames <= 0:
                continue
            if speed == 1.0:
                frame_parts.append(video.frames[start_f:end_f])
            else:
                target_count = max(1, round(n_frames / speed))
                indices = np.linspace(start_f, end_f - 1, target_count).astype(int)
                frame_parts.append(video.frames[indices])

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
                    target_len = max(1, round(len(chunk) / speed))
                    indices = np.linspace(0, len(chunk) - 1, target_len).astype(int)
                    audio_parts.append(chunk[indices])
            video.audio.data = np.concatenate(audio_parts, axis=0)
            video.audio = video.audio.fit_to_duration(len(video.frames) / video.fps)

        return video

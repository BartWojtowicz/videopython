from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from multiprocessing import Pool
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
from tqdm import tqdm

from videopython.base.video import Video

# Minimum frames before using multiprocessing (Pool overhead isn't worth it below this)
MIN_FRAMES_FOR_MULTIPROCESSING = 100

if TYPE_CHECKING:
    from videopython.base.audio import Audio

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

    def __init__(self, width: int | None = None, height: int | None = None):
        """Initialize resizer.

        Args:
            width: Target width in pixels, or None to maintain aspect ratio.
            height: Target height in pixels, or None to maintain aspect ratio.
        """
        self.width = width
        self.height = height
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

        print(f"Resizing video to: {new_width}x{new_height}!")
        n_frames = len(video.frames)

        if n_frames >= MIN_FRAMES_FOR_MULTIPROCESSING:
            with Pool() as pool:
                frames_copy = pool.starmap(
                    self._resize_frame,
                    [(frame, new_width, new_height) for frame in video.frames],
                )
        else:
            frames_copy = [self._resize_frame(frame, new_width, new_height) for frame in video.frames]

        video.frames = np.array(frames_copy)
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
        for i in tqdm(range(len(new_frame_indices) - 1)):
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
            print(f"Downsampling video from {video.fps} to {self.fps} FPS.")
            video = self._downsample(video)
        else:
            print(f"Upsampling video from {video.fps} to {self.fps} FPS.")
            video = self._upsample(video)
        return video


class CropMode(Enum):
    CENTER = "center"
    CUSTOM = "custom"  # Use x, y coordinates for positioning


class Crop(Transformation):
    """Crops video to specified dimensions.

    Supports both pixel values (int) and normalized coordinates (float 0-1).
    When using normalized coordinates, values are converted to pixels based on
    the input video dimensions.
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
            width: Target crop width. If int, interpreted as pixels.
                If float in range (0, 1], interpreted as normalized (e.g., 0.5 = 50% of width).
            height: Target crop height. If int, interpreted as pixels.
                If float in range (0, 1], interpreted as normalized (e.g., 0.5 = 50% of height).
            x: Left edge X coordinate. If int, pixels. If float in [0, 1], normalized.
                Only used when mode is not CENTER. Defaults to 0.
            y: Top edge Y coordinate. If int, pixels. If float in [0, 1], normalized.
                Only used when mode is not CENTER. Defaults to 0.
            mode: Crop mode, defaults to center crop.
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
    """Changes video playback speed with optional smooth ramping.

    Speed > 1.0 = faster playback (fewer frames)
    Speed < 1.0 = slower playback (more frames, with interpolation)
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
            speed: Playback speed multiplier (2.0 = 2x faster, 0.5 = half speed).
            end_speed: If provided, smoothly ramp from speed to end_speed over duration.
            interpolate: Whether to interpolate frames when slowing down (default True).
            adjust_audio: Whether to time-stretch audio to match video speed (default True).
                If False, audio will be sliced/padded to match new video duration.
                Note: For speed ramps, the average speed is used for audio time-stretching.
        """
        if speed <= 0:
            raise ValueError("Speed must be positive!")
        if end_speed is not None and end_speed <= 0:
            raise ValueError("End speed must be positive!")

        self.speed = speed
        self.end_speed = end_speed
        self.interpolate = interpolate
        self.adjust_audio = adjust_audio

    def _interpolate_frame(self, frames: np.ndarray, index: float) -> np.ndarray:
        """Interpolate between two frames at a fractional index."""
        idx_low = int(index)
        idx_high = min(idx_low + 1, len(frames) - 1)
        ratio = index - idx_low

        if ratio == 0 or idx_low == idx_high:
            return frames[idx_low]

        # Linear interpolation between frames
        return ((1 - ratio) * frames[idx_low] + ratio * frames[idx_high]).astype(np.uint8)

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
            print(f"Applying {self.speed}x speed change...")
            new_frame_count = int(n_frames / self.speed)

            if new_frame_count == 0:
                raise ValueError(f"Speed {self.speed}x would result in 0 frames!")

            # Map new frame indices to original frame positions
            source_indices = np.linspace(0, n_frames - 1, new_frame_count)
        else:
            # Speed ramp: smoothly transition from speed to end_speed
            print(f"Applying speed ramp from {self.speed}x to {self.end_speed}x...")

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
            for idx in tqdm(source_indices, desc="Interpolating frames"):
                new_frames.append(self._interpolate_frame(video.frames, idx))
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
    """Overlays a smaller video on top of a main video.

    Commonly used for reaction videos, gaming streams with facecam,
    tutorials with presenter overlay, and news broadcasts.
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
            overlay: Video to overlay on the main video.
            position: Normalized (x, y) position where (0, 0) is top-left, (1, 1) is bottom-right.
                The position refers to the center of the overlay.
            scale: Size of overlay relative to main video width (0.25 = 25% of main video width).
            border_width: Border width in pixels (default 0, no border).
            border_color: Border color as RGB tuple (default white).
            corner_radius: Radius for rounded corners in pixels (default 0, square corners).
            opacity: Overlay transparency from 0 (invisible) to 1 (opaque), default 1.0.
            audio_mode: How to handle audio.
                - "main": Keep only main video audio (default, current behavior).
                - "overlay": Use only overlay video audio.
                - "mix": Mix both audio tracks.
            audio_mix: Volume factors for mixing as (main_factor, overlay_factor).
                Only used when audio_mode="mix". Default (1.0, 1.0) gives equal mix.
                Values < 1.0 reduce volume, > 1.0 increase (may clip).
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
        mask = np.ones((height, width), dtype=np.float32)

        if self.corner_radius <= 0:
            return mask

        radius = min(self.corner_radius, width // 2, height // 2)

        # Create rounded corners using circles
        for corner_y, corner_x in [(0, 0), (0, width), (height, 0), (height, width)]:
            # Determine the center of the corner circle
            center_y = radius if corner_y == 0 else height - radius
            center_x = radius if corner_x == 0 else width - radius

            # Create coordinate grids for this corner region
            y_start = 0 if corner_y == 0 else height - radius
            y_end = radius if corner_y == 0 else height
            x_start = 0 if corner_x == 0 else width - radius
            x_end = radius if corner_x == 0 else width

            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if dist > radius:
                        mask[y, x] = 0

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
            # Only show border where mask is 1
            for c in range(3):
                bordered[:, :, c] = bordered[:, :, c] * mask

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
        print(f"Resizing overlay to {overlay_w}x{overlay_h}...")
        resized_overlay_frames = []
        for frame in tqdm(self.overlay.frames, desc="Resizing overlay"):
            resized = cv2.resize(frame, (overlay_w, overlay_h), interpolation=cv2.INTER_AREA)
            resized_overlay_frames.append(resized)

        # Create rounded corner mask if needed
        mask = self._create_rounded_mask(overlay_w, overlay_h)

        # Apply border to overlay frames
        if self.border_width > 0:
            resized_overlay_frames = [self._add_border(frame, mask) for frame in resized_overlay_frames]

        print("Applying picture-in-picture...")
        new_frames = []
        for i in tqdm(range(n_main_frames)):
            main_frame = video.frames[i].copy()

            # Get overlay frame (loop if overlay is shorter)
            overlay_idx = i % n_overlay_frames
            overlay_frame = resized_overlay_frames[overlay_idx]

            # Extract the region where overlay will be placed
            region = main_frame[pos_y : pos_y + overlay_h, pos_x : pos_x + overlay_w]

            # Apply mask and opacity
            if self.corner_radius > 0 or self.opacity < 1.0:
                # Blend with mask and opacity
                mask_3d = mask[:, :, np.newaxis]
                alpha = mask_3d * self.opacity
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

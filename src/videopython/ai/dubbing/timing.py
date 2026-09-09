"""Timing synchronization for dubbed audio segments."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import replace
from functools import lru_cache
from typing import Literal

import numpy as np

from videopython.ai.dubbing.models import TimingAdjustment
from videopython.audio import Audio, AudioMetadata

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _speech_stretch_method() -> Literal["atempo", "rubberband"]:
    """Prefer Rubber Band when installed; retain compatibility with core ffmpeg."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, check=True)
    if any(len(fields := line.split()) > 1 and fields[1] == "rubberband" for line in result.stdout.splitlines()):
        return "rubberband"
    logger.warning("FFmpeg has no rubberband filter; dubbing falls back to atempo time stretching")
    return "atempo"


class TimingSynchronizer:
    """Synchronizes dubbed audio segments to match original timing.

    Adjusts the speed of dubbed audio segments to fit within the timing
    constraints of the original speech, preserving complete utterances. Speeds
    beyond the preferred range are reported because they can sound unnatural.
    """

    def __init__(self, max_speed: float = 1.1):
        """Set the preferred maximum speed; larger speeds preserve speech when needed."""
        if not np.isfinite(max_speed) or max_speed < 1.0:
            raise ValueError("max_speed must be finite and at least 1.0")
        self.max_speed = max_speed

    def synchronize_segment(
        self,
        audio: Audio,
        target_duration: float,
        segment_index: int = 0,
    ) -> tuple[Audio, TimingAdjustment]:
        """Synchronize a single audio segment to a target duration.

        Args:
            audio: The audio segment to synchronize.
            target_duration: Target duration in seconds.
            segment_index: Index of this segment (for tracking).

        Returns:
            Tuple of (synchronized audio, timing adjustment info).
        """
        original_duration = audio.metadata.duration_seconds

        if original_duration <= 0 or target_duration <= 0:
            # Empty audio or zero-length target, return as-is
            return audio, TimingAdjustment(
                segment_index=segment_index,
                original_duration=original_duration,
                target_duration=target_duration,
                actual_duration=original_duration,
                speed_factor=1.0,
                excessive_speed=False,
            )

        # Calculate required speed factor
        required_speed = original_duration / target_duration

        # Source-word phrase anchors carry the delivery rhythm. Do not stretch
        # a shorter translation to fill silence; only accelerate an overrun.
        speed_factor = max(1.0, required_speed)

        # Apply time stretch
        if abs(speed_factor - 1.0) > 0.01:
            synchronized_audio = audio.time_stretch(speed_factor, method=_speech_stretch_method())
        else:
            synchronized_audio = audio
            speed_factor = 1.0

        # atempo has a small sample-count error. Fit its *whole* output to the
        # window instead of slicing off the ending. This residual resampling can
        # shift pitch slightly; the main duration change above preserves pitch.
        actual_duration = synchronized_audio.metadata.duration_seconds
        if actual_duration > target_duration:
            frames = max(1, int(target_duration * synchronized_audio.metadata.sample_rate))
            data = synchronized_audio.data
            positions = np.linspace(0, len(data) - 1, frames)
            source_positions = np.arange(len(data))
            fitted = (
                np.interp(positions, source_positions, data)
                if data.ndim == 1
                else np.column_stack([np.interp(positions, source_positions, channel) for channel in data.T])
            )
            speed_factor *= len(data) / frames
            metadata = replace(
                synchronized_audio.metadata,
                frame_count=frames,
                duration_seconds=frames / synchronized_audio.metadata.sample_rate,
            )
            synchronized_audio = Audio(fitted.astype(np.float32), metadata)
            actual_duration = metadata.duration_seconds

        excessive_speed = speed_factor > self.max_speed + 0.01
        if excessive_speed:
            logger.warning(
                "Dubbed turn %d requires %.2fx speed (preferred maximum %.2fx)",
                segment_index,
                speed_factor,
                self.max_speed,
            )

        return synchronized_audio, TimingAdjustment(
            segment_index=segment_index,
            original_duration=original_duration,
            target_duration=target_duration,
            actual_duration=actual_duration,
            speed_factor=speed_factor,
            excessive_speed=excessive_speed,
        )

    def synchronize_segments(
        self,
        audio_segments: list[Audio],
        target_durations: list[float],
    ) -> tuple[list[Audio], list[TimingAdjustment]]:
        """Synchronize multiple audio segments to their target durations.

        Args:
            audio_segments: List of audio segments to synchronize.
            target_durations: List of target durations (same length as audio_segments).

        Returns:
            Tuple of (synchronized audio segments, timing adjustments).

        Raises:
            ValueError: If lengths don't match.
        """
        if len(audio_segments) != len(target_durations):
            raise ValueError(
                f"Length mismatch: {len(audio_segments)} segments vs {len(target_durations)} target durations"
            )

        synchronized = []
        adjustments = []

        for i, (audio, target_duration) in enumerate(zip(audio_segments, target_durations)):
            synced_audio, adjustment = self.synchronize_segment(audio, target_duration, i)
            synchronized.append(synced_audio)
            adjustments.append(adjustment)

        return synchronized, adjustments

    def assemble_with_timing(
        self,
        audio_segments: list[Audio],
        start_times: list[float],
        total_duration: float,
    ) -> Audio:
        """Assemble synchronized segments into a single track with proper timing.

        Creates a track where each segment starts at its specified time,
        with silence filling gaps.

        Args:
            audio_segments: List of audio segments (already synchronized).
            start_times: Start time for each segment in seconds.
            total_duration: Total duration of the output track.

        Returns:
            Assembled audio track with segments at correct positions.

        Raises:
            ValueError: If lengths don't match or timing is invalid.
        """
        if len(audio_segments) != len(start_times):
            raise ValueError(f"Length mismatch: {len(audio_segments)} segments vs {len(start_times)} start times")

        for start_time in start_times:
            if start_time < 0:
                raise ValueError(f"Invalid start time: {start_time}")

        if not audio_segments:
            return Audio.create_silent(total_duration, stereo=False)

        # Single-pass assembler: allocate one mono float32 buffer and add each
        # segment in place at its start sample. The previous implementation
        # called Audio.overlay() per segment, which allocates np.zeros and
        # copies the full track on every call — O(N * total_samples) memory
        # traffic. For long dubs (thousands of segments) this loop dominated
        # wall time and peak RAM.
        sample_rate = audio_segments[0].metadata.sample_rate
        base_samples = max(int(total_duration * sample_rate), 0)

        # Pre-normalize each segment to (mono, target sample rate) and compute
        # placement bounds so the output buffer is sized to fit any segment
        # that runs past total_duration (mirrors Audio.overlay's extend-on-OOB
        # behavior so we don't silently truncate speech).
        normalized: list[tuple[int, np.ndarray]] = []
        end_sample = base_samples
        for audio, start_time in zip(audio_segments, start_times):
            if audio.metadata.sample_rate != sample_rate:
                audio = audio.resample(sample_rate)
            if audio.metadata.channels > 1:
                audio = audio.to_mono()
            start_sample = int(np.ceil(start_time * sample_rate))
            seg_data = audio.data
            normalized.append((start_sample, seg_data))
            end_sample = max(end_sample, start_sample + len(seg_data))

        output = np.zeros(end_sample, dtype=np.float32)
        for start_sample, seg_data in normalized:
            stop = start_sample + len(seg_data)
            # Fade only the edges, without copying or cutting the whole phrase.
            fade_samples = min(round(0.005 * sample_rate), len(seg_data) // 2)
            if fade_samples:
                ramp = np.linspace(0, 1, fade_samples, dtype=np.float32)
                output[start_sample : start_sample + fade_samples] += seg_data[:fade_samples] * ramp
                output[start_sample + fade_samples : stop - fade_samples] += seg_data[fade_samples:-fade_samples]
                output[stop - fade_samples : stop] += seg_data[-fade_samples:] * ramp[::-1]
            else:
                output[start_sample:stop] += seg_data

        # Single post-mix peak guard, equivalent to Audio.overlay's per-call
        # rescale collapsed into one pass. For non-overlapping dub segments
        # this is a no-op; only the rare overlap case touches it.
        max_amplitude = float(np.max(np.abs(output))) if output.size else 0.0
        if max_amplitude > 1.0:
            output /= max_amplitude

        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=audio_segments[0].metadata.sample_width,
            duration_seconds=end_sample / sample_rate,
            frame_count=end_sample,
        )
        return Audio(output, metadata)

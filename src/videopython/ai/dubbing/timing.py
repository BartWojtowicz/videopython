"""Timing synchronization for dubbed audio segments."""

from __future__ import annotations

from dataclasses import dataclass

from videopython.base.audio import Audio


@dataclass
class TimingAdjustment:
    """Information about a timing adjustment made to a segment.

    Attributes:
        segment_index: Index of the segment that was adjusted.
        original_duration: Original duration of the segment in seconds.
        target_duration: Target duration to fit into.
        actual_duration: Actual duration after adjustment.
        speed_factor: Speed factor applied (> 1 means sped up).
        was_truncated: Whether the segment had to be truncated.
    """

    segment_index: int
    original_duration: float
    target_duration: float
    actual_duration: float
    speed_factor: float
    was_truncated: bool


class TimingSynchronizer:
    """Synchronizes dubbed audio segments to match original timing.

    Adjusts the speed of dubbed audio segments to fit within the timing
    constraints of the original speech while maintaining natural-sounding speech.
    """

    # Speed limits for natural-sounding speech
    MIN_SPEED: float = 0.8  # Slowest allowed (20% slower)
    MAX_SPEED: float = 1.3  # Fastest allowed (30% faster)

    def __init__(
        self,
        min_speed: float | None = None,
        max_speed: float | None = None,
        gap_threshold: float = 0.1,
    ):
        """Initialize the timing synchronizer.

        Args:
            min_speed: Minimum speed factor (default: 0.8).
            max_speed: Maximum speed factor (default: 1.3).
            gap_threshold: Minimum gap between segments in seconds (default: 0.1).
        """
        self.min_speed = min_speed if min_speed is not None else self.MIN_SPEED
        self.max_speed = max_speed if max_speed is not None else self.MAX_SPEED
        self.gap_threshold = gap_threshold

        if self.min_speed <= 0:
            raise ValueError("min_speed must be positive")
        if self.max_speed <= self.min_speed:
            raise ValueError("max_speed must be greater than min_speed")

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

        if original_duration <= 0:
            # Empty audio, return as-is
            return audio, TimingAdjustment(
                segment_index=segment_index,
                original_duration=original_duration,
                target_duration=target_duration,
                actual_duration=original_duration,
                speed_factor=1.0,
                was_truncated=False,
            )

        # Calculate required speed factor
        required_speed = original_duration / target_duration

        # Clamp to acceptable range
        clamped_speed = max(self.min_speed, min(self.max_speed, required_speed))

        # Check if we need to truncate
        was_truncated = False
        if required_speed > self.max_speed:
            # Even at max speed, audio is too long - will need truncation
            was_truncated = True

        # Apply time stretch
        if abs(clamped_speed - 1.0) > 0.01:
            synchronized_audio = audio.time_stretch(clamped_speed)
        else:
            synchronized_audio = audio

        # Truncate if still too long
        actual_duration = synchronized_audio.metadata.duration_seconds
        if actual_duration > target_duration:
            synchronized_audio = synchronized_audio.slice(0, target_duration)
            actual_duration = target_duration
            was_truncated = True

        return synchronized_audio, TimingAdjustment(
            segment_index=segment_index,
            original_duration=original_duration,
            target_duration=target_duration,
            actual_duration=actual_duration,
            speed_factor=clamped_speed,
            was_truncated=was_truncated,
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

        if not audio_segments:
            return Audio.create_silent(total_duration, stereo=False)

        # Determine sample rate from first segment
        sample_rate = audio_segments[0].metadata.sample_rate

        # Create base silent track
        output = Audio.create_silent(total_duration, stereo=False, sample_rate=sample_rate)

        # Overlay each segment at its start time
        for audio, start_time in zip(audio_segments, start_times):
            if start_time < 0:
                raise ValueError(f"Invalid start time: {start_time}")

            # Resample if needed
            if audio.metadata.sample_rate != sample_rate:
                audio = audio.resample(sample_rate)

            # Convert to mono if needed
            if audio.metadata.channels > 1:
                audio = audio.to_mono()

            # Overlay at position
            output = output.overlay(audio, position=start_time)

        return output

    def check_overlaps(
        self,
        start_times: list[float],
        durations: list[float],
    ) -> list[tuple[int, int, float]]:
        """Check for overlapping segments.

        Args:
            start_times: Start time for each segment.
            durations: Duration of each segment.

        Returns:
            List of overlapping pairs as (index1, index2, overlap_duration).
        """
        if len(start_times) != len(durations):
            raise ValueError("Length mismatch between start_times and durations")

        overlaps = []
        n = len(start_times)

        for i in range(n):
            end_i = start_times[i] + durations[i]
            for j in range(i + 1, n):
                # Check if segment j starts before segment i ends
                if start_times[j] < end_i and start_times[j] >= start_times[i]:
                    overlap = end_i - start_times[j]
                    if overlap > self.gap_threshold:
                        overlaps.append((i, j, overlap))
                # Check if segment i starts before segment j ends
                elif start_times[i] < start_times[j] + durations[j] and start_times[i] >= start_times[j]:
                    overlap = start_times[j] + durations[j] - start_times[i]
                    if overlap > self.gap_threshold:
                        overlaps.append((j, i, overlap))

        return overlaps

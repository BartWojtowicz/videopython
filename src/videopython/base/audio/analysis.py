"""Audio analysis data structures.

This module provides dataclasses for representing audio analysis results
including level measurements, silence detection, and segment classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AudioSegmentType(Enum):
    """Classification of audio segment content."""

    SILENCE = "silence"
    SPEECH = "speech"
    MUSIC = "music"
    NOISE = "noise"


@dataclass
class AudioLevels:
    """Audio level measurements for a segment.

    Attributes:
        rms: Root mean square (average loudness), 0.0 to 1.0
        peak: Maximum absolute amplitude, 0.0 to 1.0
        db_rms: RMS level in decibels (relative to full scale)
        db_peak: Peak level in decibels (relative to full scale)

    Example:
        >>> audio = Audio.from_path("audio.mp3")
        >>> levels = audio.get_levels()
        >>> print(f"Peak: {levels.db_peak:.1f} dB, RMS: {levels.db_rms:.1f} dB")
    """

    rms: float
    peak: float
    db_rms: float
    db_peak: float


@dataclass
class SilentSegment:
    """Represents a detected silent segment.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        duration: Duration in seconds
        avg_level: Average RMS level during the segment

    Example:
        >>> silent_segments = audio.detect_silence(threshold_db=-40.0)
        >>> for seg in silent_segments:
        ...     print(f"Silence: {seg.start:.2f}s - {seg.end:.2f}s ({seg.duration:.2f}s)")
    """

    start: float
    end: float
    duration: float
    avg_level: float


@dataclass
class AudioSegment:
    """A classified segment of audio.

    Attributes:
        start: Start time in seconds
        end: End time in seconds
        segment_type: Classification of the segment content
        confidence: Confidence score for the classification (0.0 to 1.0)
        levels: Audio level measurements for the segment

    Example:
        >>> segments = audio.classify_segments(segment_length=2.0)
        >>> for seg in segments:
        ...     print(f"{seg.start:.1f}-{seg.end:.1f}s: {seg.segment_type.value} ({seg.confidence:.0%})")
    """

    start: float
    end: float
    segment_type: AudioSegmentType
    confidence: float
    levels: AudioLevels

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start

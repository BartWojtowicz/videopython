"""Data models for video dubbing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from videopython.base.audio import Audio
from videopython.base.text.transcription import Transcription, TranscriptionSegment

if TYPE_CHECKING:
    from videopython.ai.dubbing.quality import TranscriptQuality
    from videopython.ai.dubbing.timing import TimingAdjustment


# Speed factors within this band of 1.0 are treated as a "clean" timing
# adjustment (no perceptible compression/stretch). Heuristic threshold for
# the TimingSummary classification only.
CLEAN_SPEED_TOLERANCE = 0.01


@dataclass
class TranslatedSegment:
    """A segment of translated text with timing information.

    Attributes:
        original_segment: The original transcription segment.
        translated_text: The translated text.
        source_lang: Source language code (e.g., "en").
        target_lang: Target language code (e.g., "es").
        speaker: Speaker identifier if available.
        start: Start time in seconds.
        end: End time in seconds.
    """

    original_segment: TranscriptionSegment
    translated_text: str
    source_lang: str
    target_lang: str
    speaker: str | None = None
    start: float = 0.0
    end: float = 0.0

    def __post_init__(self) -> None:
        """Set timing from original segment if not provided."""
        if self.start == 0.0 and self.end == 0.0:
            self.start = self.original_segment.start
            self.end = self.original_segment.end
        if self.speaker is None:
            self.speaker = self.original_segment.speaker

    @property
    def original_text(self) -> str:
        """Get the original text from the segment."""
        return self.original_segment.text

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start


@dataclass
class SeparatedAudio:
    """Audio separated into different components.

    Attributes:
        vocals: Isolated vocal/speech track.
        background: Combined background audio (music + effects).
        music: Isolated music track (if available).
        effects: Isolated sound effects track (if available).
        original: The original unseparated audio.
    """

    vocals: Audio
    background: Audio
    original: Audio
    music: Audio | None = None
    effects: Audio | None = None

    @property
    def has_detailed_separation(self) -> bool:
        """Check if music and effects are separated."""
        return self.music is not None and self.effects is not None


@dataclass
class TimingSummary:
    """Aggregate stats over per-segment timing adjustments.

    Surfaces how aggressively the timing synchronizer had to compress or
    truncate dubbed segments to fit the source's spoken regions. High
    truncation rates indicate translation produced text too long for the
    source duration.
    """

    total_segments: int
    clean_count: int
    stretched_count: int
    truncated_count: int
    mean_speed_factor: float
    max_truncation_seconds: float

    @classmethod
    def from_adjustments(cls, adjustments: list[TimingAdjustment]) -> TimingSummary:
        """Aggregate a list of TimingAdjustments into a TimingSummary."""
        total = len(adjustments)
        if total == 0:
            return cls(
                total_segments=0,
                clean_count=0,
                stretched_count=0,
                truncated_count=0,
                mean_speed_factor=1.0,
                max_truncation_seconds=0.0,
            )

        clean = 0
        stretched = 0
        truncated = 0
        speed_sum = 0.0
        max_truncation = 0.0
        for adj in adjustments:
            speed_sum += adj.speed_factor
            if adj.was_truncated:
                truncated += 1
                truncation = adj.original_duration - adj.actual_duration
                if truncation > max_truncation:
                    max_truncation = truncation
            elif abs(adj.speed_factor - 1.0) <= CLEAN_SPEED_TOLERANCE:
                clean += 1
            else:
                stretched += 1

        return cls(
            total_segments=total,
            clean_count=clean,
            stretched_count=stretched,
            truncated_count=truncated,
            mean_speed_factor=speed_sum / total,
            max_truncation_seconds=max_truncation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_segments": self.total_segments,
            "clean_count": self.clean_count,
            "stretched_count": self.stretched_count,
            "truncated_count": self.truncated_count,
            "mean_speed_factor": self.mean_speed_factor,
            "max_truncation_seconds": self.max_truncation_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimingSummary:
        """Create TimingSummary from dictionary."""
        return cls(
            total_segments=data["total_segments"],
            clean_count=data["clean_count"],
            stretched_count=data["stretched_count"],
            truncated_count=data["truncated_count"],
            mean_speed_factor=data["mean_speed_factor"],
            max_truncation_seconds=data["max_truncation_seconds"],
        )


@dataclass
class DubbingResult:
    """Result of a video dubbing operation.

    Attributes:
        dubbed_audio: The final dubbed audio track.
        translated_segments: List of translated segments with timing.
        source_transcription: Original transcription of the source audio.
        source_lang: Detected or specified source language.
        target_lang: Target language for dubbing.
        separated_audio: Separated audio components (if preserve_background=True).
        voice_samples: Dictionary mapping speaker IDs to voice sample Audio.
        timing_summary: Aggregate stats over per-segment timing adjustments.
        transcript_quality: Heuristic quality assessment of the transcription
            (None when the pipeline returned early on an empty transcription).
    """

    dubbed_audio: Audio
    translated_segments: list[TranslatedSegment]
    source_transcription: Transcription
    source_lang: str
    target_lang: str
    separated_audio: SeparatedAudio | None = None
    voice_samples: dict[str, Audio] = field(default_factory=dict)
    timing_summary: TimingSummary | None = None
    transcript_quality: TranscriptQuality | None = None

    @property
    def num_segments(self) -> int:
        """Number of translated segments."""
        return len(self.translated_segments)

    @property
    def total_duration(self) -> float:
        """Total duration of the dubbed audio."""
        return self.dubbed_audio.metadata.duration_seconds

    def get_segments_by_speaker(self) -> dict[str, list[TranslatedSegment]]:
        """Group translated segments by speaker.

        Returns:
            Dictionary mapping speaker IDs to their segments.
        """
        segments_by_speaker: dict[str, list[TranslatedSegment]] = {}
        for segment in self.translated_segments:
            speaker = segment.speaker or "unknown"
            if speaker not in segments_by_speaker:
                segments_by_speaker[speaker] = []
            segments_by_speaker[speaker].append(segment)
        return segments_by_speaker


@dataclass
class RevoiceResult:
    """Result of a voice replacement operation.

    Attributes:
        revoiced_audio: The final audio with new speech.
        text: The text that was spoken.
        separated_audio: Separated audio components (if preserve_background=True).
        voice_sample: Voice sample used for cloning.
        original_duration: Duration of the original audio.
        speech_duration: Duration of the generated speech.
    """

    revoiced_audio: Audio
    text: str
    separated_audio: SeparatedAudio | None = None
    voice_sample: Audio | None = None
    original_duration: float = 0.0
    speech_duration: float = 0.0

    @property
    def total_duration(self) -> float:
        """Total duration of the revoiced audio."""
        return self.revoiced_audio.metadata.duration_seconds

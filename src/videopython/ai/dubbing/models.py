"""Data models for video dubbing."""

from __future__ import annotations

from dataclasses import dataclass, field

from videopython.base.audio import Audio
from videopython.base.text.transcription import Transcription, TranscriptionSegment


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
    """

    dubbed_audio: Audio
    translated_segments: list[TranslatedSegment]
    source_transcription: Transcription
    source_lang: str
    target_lang: str
    separated_audio: SeparatedAudio | None = None
    voice_samples: dict[str, Audio] = field(default_factory=dict)

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

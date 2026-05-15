"""Data models for video dubbing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, model_validator

from videopython.ai.dubbing.quality import TranscriptQuality
from videopython.audio import Audio
from videopython.base.transcription import Transcription, TranscriptionSegment

if TYPE_CHECKING:
    from videopython.ai.dubbing.timing import TimingAdjustment


# Speed factors within this band of 1.0 are treated as a "clean" timing
# adjustment (no perceptible compression/stretch). Heuristic threshold for
# the TimingSummary classification only.
CLEAN_SPEED_TOLERANCE = 0.01


# TranscriptionSegment and Transcription still live in videopython.base as
# plain dataclasses with hand-rolled to_dict/from_dict. Bridge them at
# the field boundary so the dubbing cache wire format stays identical.
def _validate_transcription_segment(value: Any) -> Any:
    if value is None or isinstance(value, TranscriptionSegment):
        return value
    return TranscriptionSegment.from_dict(value)


def _serialize_with_to_dict(value: Any) -> Any:
    return value.to_dict() if value is not None else None


_TranscriptionSegmentField = Annotated[
    TranscriptionSegment,
    BeforeValidator(_validate_transcription_segment),
    PlainSerializer(_serialize_with_to_dict, return_type=dict, when_used="always"),
]


class Expressiveness(BaseModel):
    """Chatterbox ``generate()`` knobs derived from source-segment prosody.

    ``None`` on any field means "let Chatterbox use its own default" --
    avoids pinning the dub against future Chatterbox default changes.

    Attributes:
        exaggeration: Emotional intensity. Chatterbox default ``0.5``;
            ``0.7+`` produces dramatic output.
        cfg_weight: Classifier-free guidance weight. Chatterbox default
            ``0.5``; lower values (~``0.3``) slow pacing.
        temperature: Sampling temperature. Chatterbox default ``0.8``.
    """

    model_config = ConfigDict(frozen=True)

    exaggeration: float | None = None
    cfg_weight: float | None = None
    temperature: float | None = None

    def as_kwargs(self) -> dict[str, float]:
        """Knobs as a dict, dropping ``None`` entries.

        Suitable for ``**``-expansion into Chatterbox.
        """
        return {
            name: value
            for name, value in (
                ("exaggeration", self.exaggeration),
                ("cfg_weight", self.cfg_weight),
                ("temperature", self.temperature),
            )
            if value is not None
        }


class TranslatedSegment(BaseModel):
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    original_segment: _TranscriptionSegmentField
    translated_text: str
    source_lang: str
    target_lang: str
    speaker: str | None = None
    start: float = 0.0
    end: float = 0.0

    @model_validator(mode="after")
    def _default_timing_from_segment(self) -> TranslatedSegment:
        # ``start == end == 0.0`` is the dataclass-era sentinel for "use the
        # original segment's timing." Preserved so legacy callers (and the
        # dub cache wire format) keep working.
        if self.start == 0.0 and self.end == 0.0:
            self.start = self.original_segment.start
            self.end = self.original_segment.end
        if self.speaker is None:
            self.speaker = self.original_segment.speaker
        return self

    @property
    def original_text(self) -> str:
        """Get the original text from the segment."""
        return self.original_segment.text

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start


class SeparatedAudio(BaseModel):
    """Audio separated into different components.

    Attributes:
        vocals: Isolated vocal/speech track.
        background: Combined background audio (music + effects).
        music: Isolated music track (if available).
        effects: Isolated sound effects track (if available).
        original: The original unseparated audio.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vocals: Audio
    background: Audio
    original: Audio
    music: Audio | None = None
    effects: Audio | None = None

    @property
    def has_detailed_separation(self) -> bool:
        """Check if music and effects are separated."""
        return self.music is not None and self.effects is not None


class TimingSummary(BaseModel):
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


class DubbingResult(BaseModel):
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
        translation_failures: Indices of segments where translation failed
            entirely. Used by Qwen3Translator when both the primary call and
            the per-segment Marian fallback fail; those segments are dubbed
            with empty text. Empty list under MarianTranslator (Marian has
            no failure mode that drops segments).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dubbed_audio: Audio
    translated_segments: list[TranslatedSegment]
    source_transcription: Transcription
    source_lang: str
    target_lang: str
    separated_audio: SeparatedAudio | None = None
    voice_samples: dict[str, Audio] = Field(default_factory=dict)
    timing_summary: TimingSummary | None = None
    transcript_quality: TranscriptQuality | None = None
    translation_failures: list[int] = Field(default_factory=list)

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


class RevoiceResult(BaseModel):
    """Result of a voice replacement operation.

    Attributes:
        revoiced_audio: The final audio with new speech.
        text: The text that was spoken.
        separated_audio: Separated audio components (if preserve_background=True).
        voice_sample: Voice sample used for cloning.
        original_duration: Duration of the original audio.
        speech_duration: Duration of the generated speech.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

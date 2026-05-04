"""Cheap heuristics over a Whisper transcription to flag degenerate output.

Surfaces three failure modes seen in production where Demucs/translation/TTS
would otherwise spend minutes producing a useless dub:

- Dominant-phrase cascade — one phrase repeats across most segments. The
  classic Whisper failure on ambient music / outro screens
  ("Thank you for watching").
- Low decoder confidence — median per-segment ``avg_logprob`` is poor.
- Silent input misread as speech — total speech duration is tiny relative
  to the clip's wall-clock duration (only meaningful on long inputs).

Each check raises a flag; a recommendation is derived from how many fired.
Threshold constants live at module scope so production data can re-tune them
without touching code structure.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from videopython.base.text.transcription import Transcription


# Tuned conservatively to favor "warn" over "reject"; first-week production
# data may move them.
DOMINANT_PHRASE_FRACTION_THRESHOLD = 0.70
LOW_LOGPROB_MEDIAN_THRESHOLD = -1.5
LOW_SPEECH_FRACTION_THRESHOLD = 0.05
SHORT_CLIP_SECONDS = 30.0  # below this, speech-fraction is too unstable to trust


Recommendation = Literal["ok", "warn", "reject"]


_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_phrase(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


@dataclass
class TranscriptQuality:
    """Quality assessment of a Whisper transcription.

    Attributes:
        recommendation: ``"ok"`` (continue), ``"warn"`` (continue, log), or
            ``"reject"`` (caller should refuse to dub if strict_quality).
        dominant_phrase: The repeating phrase that triggered the dominance
            flag, or None when the flag didn't fire.
        dominant_phrase_fraction: Character-count share of the most common
            normalized segment phrase. 0.0 when no segments.
        median_avg_logprob: Median of ``avg_logprob`` across segments that
            carry it; None when no segment had a logprob (e.g. SRT-loaded).
        speech_fraction: Sum of segment durations divided by the audio's
            wall-clock duration.
        flags: Human-readable list of which checks fired.
    """

    recommendation: Recommendation
    dominant_phrase: str | None
    dominant_phrase_fraction: float
    median_avg_logprob: float | None
    speech_fraction: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation": self.recommendation,
            "dominant_phrase": self.dominant_phrase,
            "dominant_phrase_fraction": self.dominant_phrase_fraction,
            "median_avg_logprob": self.median_avg_logprob,
            "speech_fraction": self.speech_fraction,
            "flags": list(self.flags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptQuality:
        return cls(
            recommendation=data["recommendation"],
            dominant_phrase=data.get("dominant_phrase"),
            dominant_phrase_fraction=data.get("dominant_phrase_fraction", 0.0),
            median_avg_logprob=data.get("median_avg_logprob"),
            speech_fraction=data.get("speech_fraction", 0.0),
            flags=list(data.get("flags", [])),
        )


class GarbageTranscriptError(RuntimeError):
    """Raised by the dubbing pipeline when ``strict_quality=True`` and the
    transcript heuristic returns ``recommendation="reject"``.

    The triggering :class:`TranscriptQuality` is attached as ``quality`` so
    callers can introspect the flags without re-running the pipeline.
    """

    def __init__(self, message: str, quality: TranscriptQuality):
        super().__init__(message)
        self.quality = quality


def assess_transcript(
    transcription: Transcription,
    audio_duration_seconds: float,
) -> TranscriptQuality:
    """Run the three quality checks and return a recommendation.

    See module docstring for what each check looks for.
    """
    segments = list(transcription.segments)

    # Dominant-phrase share by character count.
    dominant_phrase: str | None = None
    dominant_fraction = 0.0
    if segments:
        normalized = [_normalize_phrase(s.text) for s in segments]
        char_counts: Counter[str] = Counter()
        total_chars = 0
        for phrase in normalized:
            if not phrase:
                continue
            n = len(phrase)
            char_counts[phrase] += n
            total_chars += n
        if total_chars > 0 and char_counts:
            most_common_phrase, most_common_chars = char_counts.most_common(1)[0]
            dominant_fraction = most_common_chars / total_chars
            dominant_phrase = most_common_phrase

    # Median avg_logprob across segments that carry it.
    logprobs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
    median_logprob = statistics.median(logprobs) if logprobs else None

    # Speech fraction = sum of segment durations / audio duration.
    speech_seconds = sum(max(0.0, s.end - s.start) for s in segments)
    speech_fraction = speech_seconds / audio_duration_seconds if audio_duration_seconds > 0 else 0.0

    flags: list[str] = []
    dominance_flag = dominant_fraction >= DOMINANT_PHRASE_FRACTION_THRESHOLD
    if dominance_flag:
        flags.append(f"dominant phrase {dominant_fraction:.0%}: {dominant_phrase!r}")

    logprob_flag = median_logprob is not None and median_logprob < LOW_LOGPROB_MEDIAN_THRESHOLD
    if logprob_flag:
        flags.append(f"median avg_logprob {median_logprob:.2f} below {LOW_LOGPROB_MEDIAN_THRESHOLD}")

    # Speech-fraction is unstable on short clips; skip it there.
    speech_flag = audio_duration_seconds > SHORT_CLIP_SECONDS and speech_fraction < LOW_SPEECH_FRACTION_THRESHOLD
    if speech_flag:
        flags.append(f"speech fraction {speech_fraction:.1%} below {LOW_SPEECH_FRACTION_THRESHOLD:.0%}")

    # Reject only when dominance + at least one other flag fires; legitimate
    # repetitive content (chants, lyric clips) should warn, not reject.
    recommendation: Recommendation
    if dominance_flag and (logprob_flag or speech_flag):
        recommendation = "reject"
    elif flags:
        recommendation = "warn"
    else:
        recommendation = "ok"

    return TranscriptQuality(
        recommendation=recommendation,
        dominant_phrase=dominant_phrase if dominance_flag else None,
        dominant_phrase_fraction=dominant_fraction,
        median_avg_logprob=median_logprob,
        speech_fraction=speech_fraction,
        flags=flags,
    )

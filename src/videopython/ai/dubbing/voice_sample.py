"""Per-speaker voice-sample extraction with quality gating."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from videopython.audio import Audio

if TYPE_CHECKING:
    from videopython.base.transcription import Transcription, TranscriptionSegment

logger = logging.getLogger(__name__)

# Voice-sample quality gating thresholds. Tuned conservatively to favor
# accepting real-world dialogue over rejecting it; failures fall back to
# the longest segment with a WARNING log so we can re-tune from production
# data instead of guessing.
PEAK_CLIP_THRESHOLD = 0.99
MIN_VOCAL_BG_RMS_RATIO = 1.5
VOICE_SAMPLE_TARGET_DURATION = 6.0


def extract(
    vocal_audio: Audio,
    background_audio: Audio | None,
    transcription: Transcription,
    min_duration: float = 3.0,
    max_duration: float = 10.0,
) -> dict[str, Audio]:
    """Extract a per-speaker voice sample with quality gating.

    Picks the highest-scored segment per speaker after rejecting clipped
    slices (peak >= ``PEAK_CLIP_THRESHOLD``) and slices where Demucs left
    the background louder than the vocals
    (``vocal_rms / bg_rms < MIN_VOCAL_BG_RMS_RATIO``). When the
    background track isn't available (e.g. ``revoice`` after
    ``low_memory`` dropped it), the RMS check is skipped silently.

    Falls back to the longest available segment with a WARNING log when
    every candidate is rejected, so the dub continues with the best
    sample we have rather than silently dropping the speaker.
    """
    voice_samples: dict[str, Audio] = {}

    segments_by_speaker: dict[str, list[TranscriptionSegment]] = {}
    for segment in transcription.segments:
        speaker = segment.speaker or "speaker_0"
        if speaker not in segments_by_speaker:
            segments_by_speaker[speaker] = []
        segments_by_speaker[speaker].append(segment)

    for speaker, segments in segments_by_speaker.items():
        chosen, fallback_reason = pick(speaker, segments, vocal_audio, background_audio, min_duration)

        if chosen is None:
            logger.warning("No usable voice-sample segment for speaker %r (no candidates)", speaker)
            continue

        if fallback_reason is not None:
            logger.warning(
                "Voice-sample quality fallback for speaker %r (%d candidates): %s -- using longest segment",
                speaker,
                len(segments),
                fallback_reason,
            )

        start = chosen.start
        end = min(chosen.end, start + max_duration)
        sliced = vocal_audio.slice(start, end)
        # Audio.slice returns a numpy view into the source. Copy so the
        # short voice sample doesn't keep the full vocals array (~1.3 GB
        # for 2h sources) alive across translate + TTS.
        voice_samples[speaker] = Audio(sliced.data.copy(), sliced.metadata)

    return voice_samples


def pick(
    speaker: str,
    segments: list[TranscriptionSegment],
    vocal_audio: Audio,
    background_audio: Audio | None,
    min_duration: float,
) -> tuple[TranscriptionSegment | None, str | None]:
    """Score eligible segments and pick the best one for ``speaker``.

    Returns ``(segment, fallback_reason)``. ``fallback_reason`` is None
    when scoring picked a segment cleanly; non-None when every candidate
    was rejected and the longest segment was used instead.
    """
    if not segments:
        return None, None

    eligible = [s for s in segments if (s.end - s.start) >= min_duration]

    rejection_reasons: list[str] = []
    scored: list[tuple[float, TranscriptionSegment]] = []
    for segment in eligible:
        result, reason = score(segment, vocal_audio, background_audio)
        if result is None:
            rejection_reasons.append(reason or "rejected")
        else:
            scored.append((result, segment))

    if scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1], None

    # All eligible segments rejected (or none met the min duration).
    # Fall back to the longest segment overall so the speaker still
    # gets a clone reference.
    longest = max(segments, key=lambda s: s.end - s.start)
    if eligible:
        reason = ", ".join(sorted(set(rejection_reasons)))
    else:
        reason = f"no segment >= {min_duration:.1f}s"
    return longest, reason


def score(
    segment: TranscriptionSegment,
    vocal_audio: Audio,
    background_audio: Audio | None,
) -> tuple[float | None, str | None]:
    """Return ``(score, reason)`` for a candidate segment.

    ``score`` is ``None`` when the segment is rejected; ``reason`` carries
    the rejection cause so the fallback logger can summarize.
    """
    vocal_slice = vocal_audio.slice(segment.start, segment.end)
    if vocal_slice.data.size == 0:
        return None, "empty slice"

    peak = float(np.max(np.abs(vocal_slice.data)))
    if peak >= PEAK_CLIP_THRESHOLD:
        return None, "clipped"

    vocal_rms = float(np.sqrt(np.mean(vocal_slice.data**2)))

    if background_audio is not None:
        bg_slice = background_audio.slice(segment.start, segment.end)
        if bg_slice.data.size > 0:
            bg_rms = float(np.sqrt(np.mean(bg_slice.data**2)))
            if bg_rms > 0 and (vocal_rms / bg_rms) < MIN_VOCAL_BG_RMS_RATIO:
                return None, "background-dominated"

    duration = segment.end - segment.start
    duration_penalty = abs(duration - VOICE_SAMPLE_TARGET_DURATION)
    return vocal_rms - 0.05 * duration_penalty, None

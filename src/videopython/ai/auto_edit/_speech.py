from __future__ import annotations

import math

from videopython.base.transcription import Transcription, TranscriptionWord

from .models import SpeechCandidateConfig


def speech_passages(
    transcription: Transcription | None, config: SpeechCandidateConfig
) -> list[list[TranscriptionWord]]:
    if transcription is None or any(segment.text.strip() and not segment.words for segment in transcription.segments):
        return []
    words = sorted(transcription.words, key=lambda word: (word.start, word.end))
    if not words:
        return []
    if any(not math.isfinite(w.start) or not math.isfinite(w.end) or w.start < 0 or w.end < w.start for w in words):
        raise ValueError("Speech candidates require finite word times with 0 <= start <= end")

    spans: list[list[TranscriptionWord]] = []
    first = 0
    end = words[0].end
    for index, word in enumerate(words):
        end = max(end, word.end)
        following = words[index + 1] if index + 1 < len(words) else None
        if following is not None and following.start < end:
            continue
        terminal = word.word.rstrip().rstrip("\"'”’»)]}").endswith((".", "?", "!", "…", "。", "？", "！"))
        pause = following is not None and following.start - end >= config.pause_duration
        if terminal or pause:
            spans.append(words[first : index + 1])
            first = index + 1
            if following is not None:
                end = following.end

    passages: list[list[TranscriptionWord]] = []
    pending: list[TranscriptionWord] = []
    for span in spans:
        start = span[0].start
        end = max(word.end for word in span)
        if end - start > config.max_duration:
            pending = []
            continue
        if pending and (
            start - max(word.end for word in pending) >= config.pause_duration
            or end - pending[0].start > config.max_duration
        ):
            pending = []
        pending.extend(span)
        if end - pending[0].start >= config.min_duration:
            passages.append(pending)
            pending = []
    return passages

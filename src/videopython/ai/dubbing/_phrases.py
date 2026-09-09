import math
import re

from videopython.base.transcription import TranscriptionSegment

_MAX_SECONDS = 8.0
_MIN_WORDS = 4
_ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "e.g.", "i.e.", "np.", "itd.", "itp."}


def timed_phrases(segments: list[TranscriptionSegment]) -> tuple[list[TranscriptionSegment], list[int]]:
    """Return phrases and their source indices, preserving words and speaker ownership.

    Prefer sentence ends, then pauses/clauses. Bound long runs at word boundaries,
    avoiding tiny fragments which are poor translation and voice-cloning prompts.
    Missing, partial or inconsistent word timing leaves the source segment intact;
    text alone cannot establish trustworthy phrase timestamps.
    """
    phrases: list[TranscriptionSegment] = []
    parents: list[int] = []
    for parent, segment in enumerate(segments):
        words = segment.words
        valid = (
            len(words) >= 2 * _MIN_WORDS
            and "".join(segment.text.split()) == "".join("".join(w.word.split()) for w in words)
            and all(
                math.isfinite(w.start)
                and math.isfinite(w.end)
                and segment.start <= w.start <= w.end <= segment.end
                and w.speaker in (None, segment.speaker)
                for w in words
            )
            and all(a.start <= b.start and a.end <= b.end for a, b in zip(words, words[1:]))
        )
        cuts: list[int] = []
        if valid:
            start = 0
            while len(words) - start >= 2 * _MIN_WORDS:
                strong: list[int] = []
                weak: list[int] = []
                bounded: list[int] = []
                for end in range(start + _MIN_WORDS, len(words) - _MIN_WORDS + 1):
                    left, right = words[end - 1], words[end]
                    span = left.end - words[start].start
                    if span > _MAX_SECONDS:
                        break
                    if span < 1.5 or left.end > right.start:
                        continue
                    ending = left.word.rstrip().rstrip("\"'”’)]}")
                    bounded.append(end)
                    if ending.lower() not in _ABBREVIATIONS and ending.endswith((".", "!", "?", "…", "。", "！", "？")):
                        strong.append(end)
                    elif span >= 2.5 and (
                        ending.endswith((",", ";", ":", "，", "；", "：")) or right.start - left.end >= 0.4
                    ):
                        weak.append(end)
                # A complete sentence beats an earlier comma: do not detach
                # subordinate clauses when the sentence fits the phrase budget.
                if strong:
                    cut = strong[0]
                elif words[-1].end - words[start].start <= _MAX_SECONDS:
                    break
                elif weak or bounded:
                    cut = (weak or bounded)[-1]
                else:
                    break
                cuts.append(cut)
                start = cut
        if not cuts:
            phrases.append(segment)
            parents.append(parent)
            continue
        # Align cuts to the source text so token boundaries do not add spaces.
        text_offsets = [0, *(match.end() for match in re.finditer(r"\S", segment.text))]
        text_start = 0
        text_chars = 0
        start = 0
        for end in [*cuts, len(words)]:
            phrase_words = words[start:end]
            text_chars += sum(len("".join(word.word.split())) for word in phrase_words)
            text_end = text_offsets[text_chars]
            phrases.append(
                segment.model_copy(
                    update={
                        "start": phrase_words[0].start,
                        "end": phrase_words[-1].end,
                        "text": segment.text[text_start:text_end].strip(),
                        "words": list(phrase_words),
                    }
                )
            )
            parents.append(parent)
            start = end
            text_start = text_end
    return phrases, parents

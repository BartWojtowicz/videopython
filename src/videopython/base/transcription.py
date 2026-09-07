from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["Transcription", "TranscriptionSegment", "TranscriptionWord"]

# Sentence-ending punctuation used by ``Transcription.capitalize_sentences``.
_SENTENCE_TERMINATORS = (".", "!", "?", "…")

# Trailing characters stripped before checking for a sentence terminator
# (closing quotes/brackets and whitespace), so ``end."`` still ends a sentence.
_TRAILING_WRAPPERS = "\"')]}»”’ "


class TranscriptionWord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    word: str
    speaker: str | None = None


class TranscriptionSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: float
    end: float
    text: str
    words: list[TranscriptionWord]
    speaker: str | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None

    @classmethod
    def from_words(
        cls,
        words: list[TranscriptionWord],
        *,
        speaker: str | None = None,
        avg_logprob: float | None = None,
        no_speech_prob: float | None = None,
        compression_ratio: float | None = None,
    ) -> TranscriptionSegment:
        """Build a segment spanning ``words``, deriving start/end/text from them.

        ``words`` must be non-empty: ``start``/``end`` come from the first/last
        word and ``text`` is the words joined by single spaces. Speaker and the
        confidence fields are passed through so callers re-segmenting *within* a
        known source segment can preserve them; callers regrouping words across
        segments (where these are ambiguous) simply omit them, leaving ``None``.
        The ``words`` list is copied, so the result never aliases the caller's.
        """
        if not words:
            raise ValueError("from_words requires a non-empty word list")
        return cls(
            start=words[0].start,
            end=words[-1].end,
            text=" ".join(w.word for w in words),
            words=list(words),
            speaker=speaker,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            compression_ratio=compression_ratio,
        )


class Transcription(BaseModel):
    """A timed transcription grouped into segments."""

    model_config = ConfigDict(extra="forbid")

    segments: list[TranscriptionSegment]
    language: str | None = None

    def __init__(
        self,
        *,
        segments: list[TranscriptionSegment] | None = None,
        words: list[TranscriptionWord] | None = None,
        language: str | None = None,
        **data: Any,
    ) -> None:
        if (segments is None) == (words is None):
            raise ValueError("Exactly one of 'segments' or 'words' must be provided")
        if words is not None:
            segments = self._words_to_segments(words)
        super().__init__(segments=segments, language=language, **data)

    @property
    def words(self) -> list[TranscriptionWord]:
        """Return all words from all segments."""
        all_words = []
        for segment in self.segments:
            all_words.extend(segment.words)
        return all_words

    @property
    def speakers(self) -> set[str]:
        """Return the speaker identifiers used by the transcription."""
        return {segment.speaker for segment in self.segments if segment.speaker is not None}

    @staticmethod
    def _words_to_segments(words: list[TranscriptionWord]) -> list[TranscriptionSegment]:
        """Group words into segments based on speaker changes."""
        if not words:
            return []

        current_speaker = words[0].speaker
        current_words: list[TranscriptionWord] = []
        segments = []

        for word in words:
            if current_speaker == word.speaker:
                current_words.append(word)
            else:
                segments.append(TranscriptionSegment.from_words(current_words, speaker=current_speaker))
                current_speaker = word.speaker
                current_words = [word]

        if current_words:
            segments.append(TranscriptionSegment.from_words(current_words, speaker=current_speaker))

        return segments

    def speaker_stats(self) -> dict[str, float]:
        """Calculate speaking time percentage for each speaker.

        Returns:
            Dictionary mapping speaker names to their percentage of total speaking time
        """
        all_words = []
        for segment in self.segments:
            all_words.extend(segment.words)

        speaking_stats: dict[str, float] = {speaker: 0.0 for speaker in self.speakers}
        total_speaking_time = 0.0

        for word in all_words:
            if word.speaker is not None:
                speak_time = word.end - word.start
                total_speaking_time += speak_time
                speaking_stats[word.speaker] += speak_time

        if total_speaking_time > 0:
            for speaker in speaking_stats:
                speaking_stats[speaker] /= total_speaking_time

        return speaking_stats

    def offset(self, time: float) -> Transcription:
        """Return a new Transcription with all timings offset by the provided time value."""
        offset_segments = []

        for segment in self.segments:
            offset_words = [
                TranscriptionWord(start=w.start + time, end=w.end + time, word=w.word, speaker=w.speaker)
                for w in segment.words
            ]
            offset_segments.append(
                segment.model_copy(
                    update={"start": segment.start + time, "end": segment.end + time, "words": offset_words}
                )
            )

        return Transcription(segments=offset_segments, language=self.language)

    def standardize_segments(self, *, time: float | None = None, num_words: int | None = None) -> Transcription:
        """Return a new Transcription with standardized segments.

        Segments are also split on speaker changes so that each segment contains
        words from a single speaker.

        Args:
            time: Maximum duration in seconds for each segment
            num_words: Maximum number of words per segment

        Raises:
            ValueError: If both time and num_words are provided or if neither is provided
        """
        if (time is None) == (num_words is None):
            raise ValueError("Exactly one of 'time' or 'num_words' must be provided")

        if time is not None and time <= 0:
            raise ValueError("Time must be positive")

        if num_words is not None and num_words <= 0:
            raise ValueError("Number of words must be positive")

        # Collect all words from all segments
        all_words: list[TranscriptionWord] = []
        for segment in self.segments:
            all_words.extend(segment.words)

        if not all_words:
            return Transcription(segments=[], language=self.language)

        standardized_segments: list[TranscriptionSegment] = []

        def _flush(words: list[TranscriptionWord]) -> None:
            if not words:
                return
            # Words here are regrouped across original segments, so the source
            # segments' confidence fields no longer apply -- left as None.
            standardized_segments.append(TranscriptionSegment.from_words(words, speaker=words[0].speaker))

        if time is not None:
            current_words: list[TranscriptionWord] = []

            for word in all_words:
                if not current_words:
                    current_words = [word]
                elif word.speaker != current_words[0].speaker or word.end - current_words[0].start > time:
                    _flush(current_words)
                    current_words = [word]
                else:
                    current_words.append(word)

            _flush(current_words)

        elif num_words is not None:
            current_words = []

            for word in all_words:
                if not current_words:
                    current_words = [word]
                elif word.speaker != current_words[0].speaker or len(current_words) >= num_words:
                    _flush(current_words)
                    current_words = [word]
                else:
                    current_words.append(word)

            _flush(current_words)

        return Transcription(segments=standardized_segments, language=self.language)

    def capitalize_sentences(self) -> Transcription:
        """Return a new Transcription with sentence-start capitalization.

        The first letter of the first spoken word and of every word that
        follows sentence-ending punctuation (``.``, ``!``, ``?``, ``…``) is
        upper-cased. Remaining characters are left untouched, so acronyms and
        proper nouns from the source transcription are preserved. Timing,
        speaker, and language are carried through unchanged.

        Abbreviation detection is intentionally not attempted: a token like
        ``"U.S."`` is treated as a sentence end. This heuristic is adequate
        for burned-in subtitles and avoids a brittle abbreviation list.
        """
        capitalized_segments: list[TranscriptionSegment] = []
        start_of_sentence = True

        for segment in self.segments:
            new_words: list[TranscriptionWord] = []
            for word in segment.words:
                token = word.word
                if start_of_sentence:
                    idx = next((i for i, ch in enumerate(token) if ch.isalpha()), None)
                    if idx is not None:
                        token = token[:idx] + token[idx].upper() + token[idx + 1 :]
                        start_of_sentence = False
                if token.rstrip(_TRAILING_WRAPPERS).endswith(_SENTENCE_TERMINATORS):
                    start_of_sentence = True
                new_words.append(TranscriptionWord(start=word.start, end=word.end, word=token, speaker=word.speaker))

            capitalized_segments.append(
                segment.model_copy(update={"text": " ".join(w.word for w in new_words), "words": new_words})
            )

        return Transcription(segments=capitalized_segments, language=self.language)

    def chunk_segments(self, max_words: int) -> Transcription:
        """Return a new Transcription splitting each segment into smaller cues.

        Each segment is split into consecutive groups of at most ``max_words``
        words, using that group's own first/last word timings. Unlike
        :meth:`standardize_segments`, words are never merged across the
        original segments, so silence gaps between segments are preserved and
        subtitles do not linger over pauses. Speaker, confidence, and language
        metadata are carried through unchanged.

        Args:
            max_words: Maximum number of words per output segment.

        Raises:
            ValueError: If ``max_words`` is not positive.
        """
        if max_words <= 0:
            raise ValueError("max_words must be positive")

        chunked_segments: list[TranscriptionSegment] = []
        for segment in self.segments:
            words = segment.words
            if not words:
                chunked_segments.append(segment.model_copy(update={"words": list(segment.words)}))
                continue
            for i in range(0, len(words), max_words):
                group = words[i : i + max_words]
                # Splitting *within* one source segment -- its confidence
                # fields still apply, so carry them through.
                chunked_segments.append(
                    TranscriptionSegment.from_words(
                        group,
                        speaker=segment.speaker,
                        avg_logprob=segment.avg_logprob,
                        no_speech_prob=segment.no_speech_prob,
                        compression_ratio=segment.compression_ratio,
                    )
                )

        return Transcription(segments=chunked_segments, language=self.language)

    def slice(self, start: float, end: float) -> Transcription | None:
        """Return a new Transcription containing only words within the time range.

        Slices at word-level granularity: words that overlap with the time range
        are included, and new segments are reconstructed from the included words.

        Args:
            start: Start time in seconds (inclusive)
            end: End time in seconds (exclusive)

        Returns:
            New Transcription with words/segments in the time range, or None if no words overlap
        """
        if start >= end:
            return None

        # Collect all words that overlap with the time range
        overlapping_words: list[TranscriptionWord] = []
        for segment in self.segments:
            for word in segment.words:
                # Include word if it overlaps with our time range
                if word.end > start and word.start < end:
                    overlapping_words.append(word)

        if not overlapping_words:
            return None

        # Reconstruct segments from the overlapping words
        # Group consecutive words by speaker to form segments
        sliced_segments: list[TranscriptionSegment] = []
        current_speaker = overlapping_words[0].speaker
        current_words: list[TranscriptionWord] = []

        for word in overlapping_words:
            if word.speaker == current_speaker:
                current_words.append(word)
            else:
                # Finish current segment (speaker is ambiguous across the
                # original segments these words came from -- confidence omitted)
                if current_words:
                    sliced_segments.append(TranscriptionSegment.from_words(current_words, speaker=current_speaker))
                # Start new segment
                current_speaker = word.speaker
                current_words = [word]

        # Add final segment
        if current_words:
            sliced_segments.append(TranscriptionSegment.from_words(current_words, speaker=current_speaker))

        return Transcription(segments=sliced_segments, language=self.language)

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _parse_srt_time(timestamp: str) -> float:
        """Parse SRT timestamp (HH:MM:SS,mmm) to seconds."""
        hours, minutes, rest = timestamp.strip().split(":")
        seconds, millis = rest.split(",")
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000

    def to_srt(self) -> str:
        """Export transcription as an SRT subtitle string."""
        blocks = []
        for i, segment in enumerate(self.segments, start=1):
            start = self._format_srt_time(segment.start)
            end = self._format_srt_time(segment.end)
            blocks.append(f"{i}\n{start} --> {end}\n{segment.text}")
        return "\n\n".join(blocks) + "\n" if blocks else ""

    @classmethod
    def from_srt(cls, srt: str) -> Transcription:
        """Parse an SRT string into a Transcription.

        Each SRT block becomes a segment with a single word spanning the full
        segment duration (word-level timing is not available in SRT).

        Args:
            srt: SRT-formatted string.

        Returns:
            Transcription with one segment per SRT block.
        """
        segments: list[TranscriptionSegment] = []
        blocks = [b.strip() for b in srt.strip().split("\n\n") if b.strip()]

        for block in blocks:
            lines = block.split("\n")
            # SRT block: index, timestamp line, one or more text lines
            if len(lines) < 3:
                continue
            timestamp_line = lines[1]
            start_str, end_str = timestamp_line.split("-->")
            start = cls._parse_srt_time(start_str)
            end = cls._parse_srt_time(end_str)
            text = "\n".join(lines[2:]).strip()

            words = [TranscriptionWord(start=start, end=end, word=text)]
            segments.append(TranscriptionSegment(start=start, end=end, text=text, words=words))

        return cls(segments=segments)

    def save_srt(self, path: str | Path) -> None:
        """Write transcription to an SRT file.

        Args:
            path: Output file path.
        """
        Path(path).write_text(self.to_srt(), encoding="utf-8")

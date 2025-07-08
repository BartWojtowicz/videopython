from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptionWord:
    start: float
    end: float
    word: str


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    words: list[TranscriptionWord]


@dataclass
class Transcription:
    segments: list[TranscriptionSegment]

    def offset(self, time: float) -> Transcription:
        """Return a new Transcription with all timings offset by the provided time value."""
        offset_segments = []

        for segment in self.segments:
            offset_words = []
            for word in segment.words:
                offset_words.append(TranscriptionWord(start=word.start + time, end=word.end + time, word=word.word))

            offset_segments.append(
                TranscriptionSegment(
                    start=segment.start + time, end=segment.end + time, text=segment.text, words=offset_words
                )
            )

        return Transcription(segments=offset_segments)

    def standardize_segments(self, *, time: float | None = None, num_words: int | None = None) -> Transcription:
        """Return a new Transcription with standardized segments.

        Args:
            time: Maximum duration in seconds for each segment
            num_words: Exact number of words per segment

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
        all_words = []
        for segment in self.segments:
            all_words.extend(segment.words)

        if not all_words:
            return Transcription(segments=[])

        standardized_segments = []

        if time is not None:
            # Group words by time constraint
            current_words = []
            current_start = None

            for word in all_words:
                if current_start is None:
                    current_start = word.start
                    current_words = [word]
                elif word.end - current_start <= time:
                    current_words.append(word)
                else:
                    # Create segment from current words
                    if current_words:
                        segment_text = " ".join(w.word for w in current_words)
                        standardized_segments.append(
                            TranscriptionSegment(
                                start=current_start,
                                end=current_words[-1].end,
                                text=segment_text,
                                words=current_words.copy(),
                            )
                        )

                    # Start new segment
                    current_start = word.start
                    current_words = [word]

            # Add final segment
            if current_words:
                segment_text = " ".join(w.word for w in current_words)
                standardized_segments.append(
                    TranscriptionSegment(
                        start=current_start,  # type: ignore
                        end=current_words[-1].end,
                        text=segment_text,
                        words=current_words.copy(),
                    )
                )
        elif num_words is not None:
            # Group words by word count constraint
            for i in range(0, len(all_words), num_words):
                segment_words = all_words[i : i + num_words]
                segment_text = " ".join(w.word for w in segment_words)
                standardized_segments.append(
                    TranscriptionSegment(
                        start=segment_words[0].start, end=segment_words[-1].end, text=segment_text, words=segment_words
                    )
                )

        return Transcription(segments=standardized_segments)

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Transcription", "TranscriptionSegment", "TranscriptionWord"]


@dataclass
class TranscriptionWord:
    start: float
    end: float
    word: str
    speaker: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "word": self.word,
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TranscriptionWord:
        """Create TranscriptionWord from dictionary."""
        return cls(
            start=data["start"],
            end=data["end"],
            word=data["word"],
            speaker=data.get("speaker"),
        )


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str
    words: list[TranscriptionWord]
    speaker: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "speaker": self.speaker,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TranscriptionSegment:
        """Create TranscriptionSegment from dictionary."""
        return cls(
            start=data["start"],
            end=data["end"],
            text=data["text"],
            words=[TranscriptionWord.from_dict(w) for w in data["words"]],
            speaker=data.get("speaker"),
        )


class Transcription:
    def __init__(
        self,
        segments: list[TranscriptionSegment] | None = None,
        words: list[TranscriptionWord] | None = None,
    ):
        """Initialize Transcription from either segments or words.

        Args:
            segments: Pre-constructed segments (backward compatible)
            words: Words to group into segments by speaker (for diarization)

        Raises:
            ValueError: If both or neither arguments are provided
        """
        if (segments is None) == (words is None):
            raise ValueError("Exactly one of 'segments' or 'words' must be provided")

        if segments is not None:
            self.segments = segments
            self.speakers = {s.speaker for s in segments if s.speaker is not None}
        else:
            self.segments = self._words_to_segments(words)  # type: ignore
            self.speakers = {w.speaker for w in words if w.speaker is not None}  # type: ignore

    @property
    def words(self) -> list[TranscriptionWord]:
        """Return all words from all segments."""
        all_words = []
        for segment in self.segments:
            all_words.extend(segment.words)
        return all_words

    def _words_to_segments(self, words: list[TranscriptionWord]) -> list[TranscriptionSegment]:
        """Group words into segments based on speaker changes."""
        if not words:
            return []

        current_speaker = words[0].speaker
        current_words = []
        segment_start = words[0].start
        segments = []

        for word in words:
            if current_speaker == word.speaker:
                current_words.append(word)
            else:
                segment_text = " ".join(w.word for w in current_words)
                segments.append(
                    TranscriptionSegment(
                        start=segment_start,
                        end=current_words[-1].end,
                        text=segment_text.strip(),
                        words=current_words.copy(),
                        speaker=current_speaker,
                    )
                )
                current_speaker = word.speaker
                current_words = [word]
                segment_start = word.start

        if current_words:
            segment_text = " ".join(w.word for w in current_words)
            segments.append(
                TranscriptionSegment(
                    start=segment_start,
                    end=current_words[-1].end,
                    text=segment_text.strip(),
                    words=current_words.copy(),
                    speaker=current_speaker,
                )
            )

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
            offset_words = []
            for word in segment.words:
                offset_words.append(
                    TranscriptionWord(
                        start=word.start + time, end=word.end + time, word=word.word, speaker=word.speaker
                    )
                )

            offset_segments.append(
                TranscriptionSegment(
                    start=segment.start + time,
                    end=segment.end + time,
                    text=segment.text,
                    words=offset_words,
                    speaker=segment.speaker,
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
                # Finish current segment
                if current_words:
                    segment_text = " ".join(w.word for w in current_words)
                    sliced_segments.append(
                        TranscriptionSegment(
                            start=current_words[0].start,
                            end=current_words[-1].end,
                            text=segment_text,
                            words=current_words.copy(),
                            speaker=current_speaker,
                        )
                    )
                # Start new segment
                current_speaker = word.speaker
                current_words = [word]

        # Add final segment
        if current_words:
            segment_text = " ".join(w.word for w in current_words)
            sliced_segments.append(
                TranscriptionSegment(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=segment_text,
                    words=current_words.copy(),
                    speaker=current_speaker,
                )
            )

        return Transcription(segments=sliced_segments)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transcription:
        """Create Transcription from dictionary."""
        return cls(segments=[TranscriptionSegment.from_dict(s) for s in data["segments"]])

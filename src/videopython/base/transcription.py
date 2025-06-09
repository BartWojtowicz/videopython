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

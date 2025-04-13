from dataclasses import dataclass


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class Transcription:
    segments: list[TranscriptionSegment]

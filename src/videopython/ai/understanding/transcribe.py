from typing import Literal

import whisper

from videopython.base.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video


class VideoTranscription:
    def __init__(self, model_name: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "small") -> None:
        self.model = whisper.load_model(name=model_name)

    def transcribe_video(self, video: Video) -> Transcription:
        """Transcribes video to text.

        Args:
            video: Video to transcribe.

        Returns:
            List of dictionaries with segments of text and their start and end times.
        """
        if video.audio.is_silent:
            return Transcription(segments=[])

        audio = video.audio.to_mono()
        audio = audio.resample(whisper.audio.SAMPLE_RATE)
        audio_data = audio.data

        transcription = self.model.transcribe(audio=audio_data, word_timestamps=True)

        transcription_segments = []
        for segment in transcription["segments"]:
            transcription_words = [
                TranscriptionWord(word=word["word"], start=float(word["start"]), end=float(word["end"]))
                for word in segment["words"]
            ]
            transcription_segment = TranscriptionSegment(
                start=segment["start"], end=segment["end"], text=segment["text"], words=transcription_words
            )
            transcription_segments.append(transcription_segment)

        result = Transcription(segments=transcription_segments)
        return result

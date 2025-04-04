from typing import Literal

import whisper

from videopython.base.video import Video


class Transcription:
    def __init__(self, model_name: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "small") -> None:
        self.model = whisper.load_model(name=model_name)

    def transcribe_video(self, video: Video) -> list[dict]:
        """Transcribes video to text.

        Args:
            video: Video to transcribe.

        Returns:
            List of dictionaries with segments of text and their start and end times.
        """
        if video.audio.is_silent():
            return []

        audio = video.audio.to_mono()
        audio = audio.resample(whisper.audio.SAMPLE_RATE)
        audio_data = audio.data

        transcription = self.model.transcribe(audio_data, word_timestamps=True)

        result = [
            {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
            for segment in transcription["segments"]
        ]

        return result

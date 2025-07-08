from typing import Literal, Union

import whisper
from soundpython import Audio

from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video


class CreateTranscription:
    """Unified transcription service for both audio and video."""

    def __init__(self, model_name: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "small") -> None:
        self.model = whisper.load_model(name=model_name)

    def _process_transcription_result(self, transcription_result: dict) -> Transcription:
        """Process raw transcription result into Transcription object.

        Args:
            transcription_result: Raw result from whisper model

        Returns:
            Processed Transcription object
        """
        transcription_segments = []
        for segment in transcription_result["segments"]:
            transcription_words = [
                TranscriptionWord(word=word["word"], start=float(word["start"]), end=float(word["end"]))
                for word in segment["words"]
            ]
            transcription_segment = TranscriptionSegment(
                start=segment["start"], end=segment["end"], text=segment["text"], words=transcription_words
            )
            transcription_segments.append(transcription_segment)

        return Transcription(segments=transcription_segments)

    def transcribe(self, media: Union[Audio, Video]) -> Transcription:
        """Transcribe audio or video to text.

        Args:
            media: Audio or Video to transcribe.

        Returns:
            Transcription object with segments of text and their timestamps.
        """
        if isinstance(media, Video):
            # Handle video transcription
            if media.audio.is_silent:
                return Transcription(segments=[])

            audio = media.audio.to_mono().resample(whisper.audio.SAMPLE_RATE)
            transcription_result = self.model.transcribe(audio=audio.data, word_timestamps=True)

        elif isinstance(media, Audio):
            # Handle audio transcription
            if media.is_silent:
                return Transcription(segments=[])

            audio = media.to_mono().resample(whisper.audio.SAMPLE_RATE)
            transcription_result = self.model.transcribe(audio=audio.data, word_timestamps=True)

        else:
            raise TypeError(f"Unsupported media type: {type(media)}. Expected Audio or Video.")

        return self._process_transcription_result(transcription_result)

from typing import Literal, Union

import whisper
from soundpython import Audio

from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video


class AudioToText:
    """Unified transcription service for both audio and video."""

    def __init__(
        self,
        model_name: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "small",
        enable_diarization: bool = False,
        device: str = "cpu",
        compute_type: str = "float32",
    ) -> None:
        """Initialize the audio-to-text transcriber.

        Args:
            model_name: Whisper model to use for transcription (default: "small")
            enable_diarization: Enable speaker diarization (default: False)
            device: Device to use for computation (default: "cpu")
            compute_type: Compute type for model (default: "float32")
        """
        self.model_name = model_name
        self.enable_diarization = enable_diarization
        self.device = device
        self.compute_type = compute_type

        if enable_diarization:
            import whisperx  # type: ignore

            self.model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
        else:
            self.model = whisper.load_model(name=model_name)

    def _process_transcription_result(self, transcription_result: dict) -> Transcription:
        """Process raw transcription result into Transcription object (without diarization).

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

    def _process_whisperx_result(self, whisperx_result: dict, audio_data) -> Transcription:
        """Process whisperx result with diarization into Transcription object.

        Args:
            whisperx_result: Raw result from whisperx transcription
            audio_data: Audio data for diarization

        Returns:
            Processed Transcription object with speaker information
        """
        import whisperx  # type: ignore

        # Step 2: Align the whisper output to get word timestamps
        model_a, metadata = whisperx.load_align_model(language_code=whisperx_result["language"], device=self.device)
        aligned_result = whisperx.align(
            whisperx_result["segments"], model_a, metadata, audio_data, self.device, return_char_alignments=False
        )

        # Step 3: Assign speaker labels
        diarize_model = whisperx.diarize.DiarizationPipeline(device=self.device)
        diarize_segments = diarize_model(audio_data)
        result_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned_result)

        # Step 4: Transform the whisperx result to our Transcription class
        words = []
        for item in result_with_speakers["word_segments"]:
            words.append(
                TranscriptionWord(
                    word=item["word"],
                    start=item["start"],
                    end=item["end"],
                    speaker=item.get("speaker", None),
                )
            )

        return Transcription(words=words)

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

        elif isinstance(media, Audio):
            # Handle audio transcription
            if media.is_silent:
                return Transcription(segments=[])
            audio = media.to_mono().resample(whisper.audio.SAMPLE_RATE)

        else:
            raise TypeError(f"Unsupported media type: {type(media)}. Expected Audio or Video.")

        if self.enable_diarization:
            # Step 1: Run transcription with whisperx
            # whisperx expects audio as numpy array
            audio_data = audio.data
            transcription_result = self.model.transcribe(audio_data)

            # Process with diarization pipeline
            return self._process_whisperx_result(transcription_result, audio_data)
        else:
            # Use standard whisper transcription without diarization
            transcription_result = self.model.transcribe(audio=audio.data, word_timestamps=True)
            return self._process_transcription_result(transcription_result)

"""Audio understanding with multi-backend support."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from videopython.ai.backends import AudioToTextBackend, UnsupportedBackendError, get_api_key
from videopython.ai.config import get_default_backend
from videopython.base.audio import Audio
from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video


class AudioToText:
    """Transcription service for audio and video."""

    SUPPORTED_BACKENDS: list[str] = ["local", "openai", "gemini"]

    def __init__(
        self,
        backend: AudioToTextBackend | None = None,
        model_name: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "small",
        enable_diarization: bool = False,
        device: str = "cpu",
        compute_type: str = "float32",
        api_key: str | None = None,
    ):
        """Initialize the audio-to-text transcriber.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            model_name: Whisper model for local backend.
            enable_diarization: Enable speaker diarization (local backend only).
            device: Device for local backend ('cuda' or 'cpu').
            compute_type: Compute type for local backend.
            api_key: API key for cloud backends. If None, reads from environment.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("audio_to_text")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: AudioToTextBackend = resolved_backend  # type: ignore[assignment]
        self.model_name = model_name
        self.enable_diarization = enable_diarization
        self.device = device
        self.compute_type = compute_type
        self.api_key = api_key

        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local Whisper model."""
        if self.enable_diarization:
            import whisperx  # type: ignore

            self._model = whisperx.load_model(self.model_name, device=self.device, compute_type=self.compute_type)
        else:
            import whisper

            self._model = whisper.load_model(name=self.model_name)

    def _process_transcription_result(self, transcription_result: dict) -> Transcription:
        """Process raw transcription result into Transcription object."""
        transcription_segments = []
        for segment in transcription_result["segments"]:
            transcription_words = [
                TranscriptionWord(word=word["word"], start=float(word["start"]), end=float(word["end"]))
                for word in segment.get("words", [])
            ]
            transcription_segment = TranscriptionSegment(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
                words=transcription_words,
            )
            transcription_segments.append(transcription_segment)

        return Transcription(segments=transcription_segments)

    def _process_whisperx_result(self, whisperx_result: dict, audio_data) -> Transcription:
        """Process whisperx result with diarization."""
        import whisperx  # type: ignore

        model_a, metadata = whisperx.load_align_model(language_code=whisperx_result["language"], device=self.device)
        aligned_result = whisperx.align(
            whisperx_result["segments"],
            model_a,
            metadata,
            audio_data,
            self.device,
            return_char_alignments=False,
        )

        diarize_model = whisperx.diarize.DiarizationPipeline(device=self.device)
        diarize_segments = diarize_model(audio_data)
        result_with_speakers = whisperx.assign_word_speakers(diarize_segments, aligned_result)

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

    def _transcribe_local(self, audio: Audio) -> Transcription:
        """Transcribe using local Whisper model."""
        import whisper

        if self._model is None:
            self._init_local()

        audio_mono = audio.to_mono().resample(whisper.audio.SAMPLE_RATE)

        if self.enable_diarization:
            audio_data = audio_mono.data
            transcription_result = self._model.transcribe(audio_data)
            return self._process_whisperx_result(transcription_result, audio_data)
        else:
            transcription_result = self._model.transcribe(audio=audio_mono.data, word_timestamps=True)
            return self._process_transcription_result(transcription_result)

    def _transcribe_openai(self, audio: Audio) -> Transcription:
        """Transcribe using OpenAI Whisper API."""
        from openai import OpenAI

        api_key = get_api_key("openai", self.api_key)
        client = OpenAI(api_key=api_key)

        # Convert audio to file-like object (WAV format)
        # Save to temp file first, then read into BytesIO
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio.save(f.name)
            temp_path = f.name

        audio_bytes = io.BytesIO(Path(temp_path).read_bytes())
        audio_bytes.name = "audio.wav"
        Path(temp_path).unlink()  # Clean up temp file

        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

        # Convert OpenAI response to Transcription
        segments = []
        for segment in response.segments or []:
            words = []
            # OpenAI may include words in segment
            for word in getattr(response, "words", []) or []:
                if segment.start <= word.start < segment.end:
                    words.append(
                        TranscriptionWord(
                            word=word.word,
                            start=word.start,
                            end=word.end,
                        )
                    )

            segments.append(
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    words=words,
                )
            )

        return Transcription(segments=segments)

    def _transcribe_gemini(self, audio: Audio) -> Transcription:
        """Transcribe using Google Gemini."""
        import google.generativeai as genai

        api_key = get_api_key("gemini", self.api_key)
        genai.configure(api_key=api_key)

        # Save audio to temp file (Gemini needs file path or bytes)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio.save(f.name)
            temp_path = f.name

        model = genai.GenerativeModel("gemini-2.0-flash")

        try:
            # Upload audio file
            audio_file = genai.upload_file(temp_path)

            response = model.generate_content(
                [
                    audio_file,
                    "Transcribe this audio. Return only the transcription text, nothing else.",
                ]
            )
            transcription_text = response.text
        finally:
            os.unlink(temp_path)

        # Gemini doesn't provide timestamps, create a single segment
        return Transcription(
            segments=[
                TranscriptionSegment(
                    start=0.0,
                    end=audio.metadata.duration_seconds,
                    text=transcription_text.strip(),
                    words=[],
                )
            ]
        )

    def transcribe(self, media: Audio | Video) -> Transcription:
        """Transcribe audio or video to text.

        Args:
            media: Audio or Video to transcribe.

        Returns:
            Transcription object with segments of text and their timestamps.
        """
        if isinstance(media, Video):
            if media.audio.is_silent:
                return Transcription(segments=[])
            audio = media.audio
        elif isinstance(media, Audio):
            if media.is_silent:
                return Transcription(segments=[])
            audio = media
        else:
            raise TypeError(f"Unsupported media type: {type(media)}. Expected Audio or Video.")

        if self.backend == "local":
            return self._transcribe_local(audio)
        elif self.backend == "openai":
            return self._transcribe_openai(audio)
        elif self.backend == "gemini":
            return self._transcribe_gemini(audio)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)

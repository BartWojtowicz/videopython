"""Audio understanding with multi-backend support."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from videopython.ai.backends import AudioClassifierBackend, AudioToTextBackend, UnsupportedBackendError, get_api_key
from videopython.ai.config import get_default_backend
from videopython.base.audio import Audio
from videopython.base.description import AudioClassification, AudioEvent
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
        import torch.serialization
        import whisperx  # type: ignore
        from omegaconf import DictConfig, ListConfig, OmegaConf

        # PyTorch 2.6+ defaults weights_only=True which breaks pyannote's omegaconf serialization
        torch.serialization.add_safe_globals([DictConfig, ListConfig, OmegaConf])

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


class AudioClassifier:
    """Audio event and sound classification using AST (Audio Spectrogram Transformer).

    Detects and classifies sounds, music, and audio events with timestamps.
    Uses AST trained on AudioSet with 527 sound classes.
    """

    SUPPORTED_BACKENDS: list[str] = ["local"]
    SUPPORTED_MODELS: list[str] = ["MIT/ast-finetuned-audioset-10-10-0.4593"]
    AST_SAMPLE_RATE: int = 16000
    # AST processes ~10 second chunks, we use sliding window for longer audio
    AST_CHUNK_SECONDS: float = 10.0
    AST_HOP_SECONDS: float = 5.0  # 50% overlap between chunks

    def __init__(
        self,
        backend: AudioClassifierBackend | None = None,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        confidence_threshold: float = 0.3,
        top_k: int = 10,
        device: str = "cpu",
    ):
        """Initialize the audio classifier.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            model_name: HuggingFace model ID for AST.
            confidence_threshold: Minimum confidence to include an event.
            top_k: Maximum number of classes to consider per time frame.
            device: Device for local backend ('cuda', 'mps', or 'cpu').
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("audio_classifier")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Supported: {self.SUPPORTED_MODELS}")

        self.backend: AudioClassifierBackend = resolved_backend  # type: ignore[assignment]
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.device = device

        self._model: Any = None
        self._processor: Any = None
        self._labels: list[str] = []

    def _init_local(self) -> None:
        """Initialize local AST model from HuggingFace."""
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self._processor = ASTFeatureExtractor.from_pretrained(self.model_name)
        self._model = ASTForAudioClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        # Get labels from model config (527 AudioSet classes)
        self._labels = [self._model.config.id2label[i] for i in range(len(self._model.config.id2label))]

    def _merge_events(self, events: list[AudioEvent], gap_threshold: float = 0.5) -> list[AudioEvent]:
        """Merge consecutive events of the same class.

        Args:
            events: List of audio events sorted by start time.
            gap_threshold: Maximum gap in seconds to merge events.

        Returns:
            List of merged audio events.
        """
        if not events:
            return []

        # Sort by label, then by start time
        events_by_label: dict[str, list[AudioEvent]] = {}
        for event in events:
            if event.label not in events_by_label:
                events_by_label[event.label] = []
            events_by_label[event.label].append(event)

        merged = []
        for label, label_events in events_by_label.items():
            sorted_events = sorted(label_events, key=lambda e: e.start)
            current = sorted_events[0]

            for next_event in sorted_events[1:]:
                # If events are close enough, merge them
                if next_event.start - current.end <= gap_threshold:
                    current = AudioEvent(
                        start=current.start,
                        end=next_event.end,
                        label=label,
                        confidence=max(current.confidence, next_event.confidence),
                    )
                else:
                    merged.append(current)
                    current = next_event

            merged.append(current)

        # Sort final list by start time
        return sorted(merged, key=lambda e: e.start)

    def _classify_local(self, audio: Audio) -> AudioClassification:
        """Classify audio using local AST model with sliding window for temporal events."""
        import numpy as np
        import torch

        if self._model is None:
            self._init_local()

        # Resample to AST expected sample rate (16kHz mono)
        audio_processed = audio.to_mono().resample(self.AST_SAMPLE_RATE)
        audio_data = audio_processed.data.astype(np.float32)

        # Calculate chunk and hop sizes in samples
        chunk_samples = int(self.AST_CHUNK_SECONDS * self.AST_SAMPLE_RATE)
        hop_samples = int(self.AST_HOP_SECONDS * self.AST_SAMPLE_RATE)
        total_samples = len(audio_data)

        # Process audio in overlapping chunks for temporal resolution
        all_chunk_probs = []
        chunk_times = []

        # If audio is shorter than one chunk, just process it directly
        if total_samples <= chunk_samples:
            chunks = [(0, audio_data)]
        else:
            chunks = []
            start = 0
            while start < total_samples:
                end = min(start + chunk_samples, total_samples)
                chunk = audio_data[start:end]
                # Pad short final chunk
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                chunks.append((start, chunk))
                start += hop_samples

        for start_sample, chunk in chunks:
            start_time = start_sample / self.AST_SAMPLE_RATE

            # Process through AST
            inputs = self._processor(
                chunk,
                sampling_rate=self.AST_SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0]  # Remove batch dimension
                probs = torch.sigmoid(logits).cpu().numpy()

            all_chunk_probs.append(probs)
            chunk_times.append(start_time)

        # Convert to numpy array for easier processing
        chunk_probs_array = np.array(all_chunk_probs)  # shape: (num_chunks, num_classes)

        # Extract events from chunk-level predictions
        events = []
        for chunk_idx, (start_time, probs) in enumerate(zip(chunk_times, chunk_probs_array)):
            end_time = start_time + self.AST_CHUNK_SECONDS

            # Get top-k classes for this chunk
            top_indices = np.argsort(probs)[-self.top_k :][::-1]

            for class_idx in top_indices:
                confidence = float(probs[class_idx])
                if confidence >= self.confidence_threshold:
                    label = self._labels[class_idx]
                    events.append(
                        AudioEvent(
                            start=start_time,
                            end=min(end_time, total_samples / self.AST_SAMPLE_RATE),
                            label=label,
                            confidence=confidence,
                        )
                    )

        # Merge consecutive events of the same class
        merged_events = self._merge_events(events)

        # Calculate clip-level predictions (average across all chunks)
        clip_preds = np.mean(chunk_probs_array, axis=0)
        top_clip_indices = np.argsort(clip_preds)[-self.top_k :][::-1]
        clip_predictions = {
            self._labels[idx]: float(clip_preds[idx])
            for idx in top_clip_indices
            if clip_preds[idx] >= self.confidence_threshold
        }

        return AudioClassification(events=merged_events, clip_predictions=clip_predictions)

    def classify(self, media: Audio | Video) -> AudioClassification:
        """Classify audio events in audio or video.

        Args:
            media: Audio or Video to classify.

        Returns:
            AudioClassification with detected events and timestamps.
        """
        if isinstance(media, Video):
            if media.audio.is_silent:
                return AudioClassification(events=[], clip_predictions={})
            audio = media.audio
        elif isinstance(media, Audio):
            if media.is_silent:
                return AudioClassification(events=[], clip_predictions={})
            audio = media
        else:
            raise TypeError(f"Unsupported media type: {type(media)}. Expected Audio or Video.")

        if self.backend == "local":
            return self._classify_local(audio)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)

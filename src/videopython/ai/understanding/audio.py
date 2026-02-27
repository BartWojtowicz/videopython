"""Audio understanding using local models."""

from __future__ import annotations

from typing import Any, Literal

from videopython.ai._device import log_device_initialization, select_device
from videopython.base.audio import Audio
from videopython.base.description import AudioClassification, AudioEvent
from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video


class AudioToText:
    """Transcription service for audio and video using local Whisper models."""

    def __init__(
        self,
        model_name: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "small",
        enable_diarization: bool = False,
        device: str | None = None,
        compute_type: str = "float32",
    ):
        self.model_name = model_name
        self.enable_diarization = enable_diarization
        self.device = select_device(device, mps_allowed=True)
        log_device_initialization(
            "AudioToText",
            requested_device=device,
            resolved_device=self.device,
        )
        self.compute_type = compute_type
        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local Whisper model."""
        if self.enable_diarization:
            import whisperx  # type: ignore

            self._model = whisperx.load_model(self.model_name, device=self.device, compute_type=self.compute_type)
        else:
            import whisper

            self._model = whisper.load_model(name=self.model_name, device=self.device)

    def _process_transcription_result(self, transcription_result: dict) -> Transcription:
        """Process raw transcription result into a Transcription object."""
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

        transcription_result = self._model.transcribe(audio=audio_mono.data, word_timestamps=True)
        return self._process_transcription_result(transcription_result)

    def transcribe(self, media: Audio | Video) -> Transcription:
        """Transcribe audio or video to text."""
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

        return self._transcribe_local(audio)


class AudioClassifier:
    """Audio event and sound classification using AST."""

    SUPPORTED_MODELS: list[str] = ["MIT/ast-finetuned-audioset-10-10-0.4593"]
    AST_SAMPLE_RATE: int = 16000
    AST_CHUNK_SECONDS: float = 10.0
    AST_HOP_SECONDS: float = 5.0

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        confidence_threshold: float = 0.3,
        top_k: int = 10,
        device: str | None = None,
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Supported: {self.SUPPORTED_MODELS}")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.device = select_device(device, mps_allowed=True)
        log_device_initialization(
            "AudioClassifier",
            requested_device=device,
            resolved_device=self.device,
        )

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

        self._labels = [self._model.config.id2label[i] for i in range(len(self._model.config.id2label))]

    def _merge_events(self, events: list[AudioEvent], gap_threshold: float = 0.5) -> list[AudioEvent]:
        """Merge consecutive events of the same class."""
        if not events:
            return []

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

        return sorted(merged, key=lambda e: e.start)

    def _classify_local(self, audio: Audio) -> AudioClassification:
        """Classify audio using local AST model with sliding window."""
        import numpy as np
        import torch

        if self._model is None:
            self._init_local()

        audio_processed = audio.to_mono().resample(self.AST_SAMPLE_RATE)
        audio_data = audio_processed.data.astype(np.float32)

        chunk_samples = int(self.AST_CHUNK_SECONDS * self.AST_SAMPLE_RATE)
        hop_samples = int(self.AST_HOP_SECONDS * self.AST_SAMPLE_RATE)
        total_samples = len(audio_data)

        all_chunk_probs = []
        chunk_times = []

        if total_samples <= chunk_samples:
            chunks = [(0, audio_data)]
        else:
            chunks = []
            start = 0
            while start < total_samples:
                end = min(start + chunk_samples, total_samples)
                chunk = audio_data[start:end]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                chunks.append((start, chunk))
                start += hop_samples

        for start_sample, chunk in chunks:
            start_time = start_sample / self.AST_SAMPLE_RATE

            inputs = self._processor(
                chunk,
                sampling_rate=self.AST_SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0]
                probs = torch.sigmoid(logits).cpu().numpy()

            all_chunk_probs.append(probs)
            chunk_times.append(start_time)

        chunk_probs_array = np.array(all_chunk_probs)

        events = []
        for start_time, probs in zip(chunk_times, chunk_probs_array):
            end_time = start_time + self.AST_CHUNK_SECONDS
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

        merged_events = self._merge_events(events)

        clip_preds = np.mean(chunk_probs_array, axis=0)
        top_clip_indices = np.argsort(clip_preds)[-self.top_k :][::-1]
        clip_predictions = {
            self._labels[idx]: float(clip_preds[idx])
            for idx in top_clip_indices
            if clip_preds[idx] >= self.confidence_threshold
        }

        return AudioClassification(events=merged_events, clip_predictions=clip_predictions)

    def classify(self, media: Audio | Video) -> AudioClassification:
        """Classify audio events in audio or video."""
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

        return self._classify_local(audio)

"""Audio generation using local models."""

from __future__ import annotations

from typing import Any

from videopython.ai._device import log_device_initialization, select_device
from videopython.base.audio import Audio, AudioMetadata


class TextToSpeech:
    """Generates speech audio from text using local models.

    Supports Bark (`base`, `small`) for general TTS and Qwen3-TTS (`qwen3`)
    for multilingual voice cloning.
    """

    SUPPORTED_LOCAL_MODELS: list[str] = ["base", "small", "qwen3"]

    # Qwen3-TTS language code to name mapping
    QWEN3_LANGUAGES: dict[str, str] = {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
    }

    def __init__(
        self,
        model_size: str = "base",
        voice: str | None = None,
        device: str | None = None,
        language: str = "en",
    ):
        if model_size not in self.SUPPORTED_LOCAL_MODELS:
            raise ValueError(f"model_size must be one of {self.SUPPORTED_LOCAL_MODELS}, got '{model_size}'")

        self.model_size = model_size
        self.voice = voice
        self.device = device
        self.language = language
        self._model: Any = None
        self._processor: Any = None
        self._qwen3_model: Any = None

    def _init_local(self) -> None:
        """Initialize local Bark model."""
        from transformers import AutoModel, AutoProcessor

        requested_device = self.device
        device = select_device(self.device, mps_allowed=False)

        model_name = "suno/bark" if self.model_size == "base" else "suno/bark-small"
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        log_device_initialization(
            "TextToSpeech",
            requested_device=requested_device,
            resolved_device=device,
        )

    def _init_qwen3(self) -> None:
        """Initialize Qwen3-TTS model for voice cloning."""
        from videopython.ai.generation._qwen3_tts_compat import apply_patches

        apply_patches()

        from qwen_tts import Qwen3TTSModel  # type: ignore[import-untyped]

        requested_device = self.device
        device = select_device(self.device, mps_allowed=False)

        self._qwen3_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map=device,
        )
        self.device = device
        log_device_initialization(
            "TextToSpeech",
            requested_device=requested_device,
            resolved_device=device,
        )

    def _generate_local(self, text: str, voice_preset: str | None) -> Audio:
        """Generate speech using Bark."""
        import torch

        if self._model is None:
            self._init_local()

        inputs = self._processor(text=[text], return_tensors="pt", voice_preset=voice_preset)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            speech_values = self._model.generate(**inputs, do_sample=True)

        audio_data = speech_values.cpu().float().numpy().squeeze()
        sample_rate = self._model.generation_config.sample_rate

        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=len(audio_data) / sample_rate,
            frame_count=len(audio_data),
        )
        return Audio(audio_data, metadata)

    def _generate_qwen3(self, text: str, voice_sample: Audio) -> Audio:
        """Generate speech using Qwen3-TTS with voice cloning."""
        import tempfile
        from pathlib import Path

        import numpy as np

        if self._qwen3_model is None:
            self._init_qwen3()

        language = self.QWEN3_LANGUAGES.get(self.language, "English")

        # Save voice sample to temp file for ref_audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            voice_sample.save(f.name)
            ref_path = Path(f.name)

        try:
            wavs, sr = self._qwen3_model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=str(ref_path),
                x_vector_only_mode=True,
            )
        finally:
            ref_path.unlink()

        audio_data = wavs[0] if isinstance(wavs[0], np.ndarray) else wavs[0].cpu().float().numpy()
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()

        metadata = AudioMetadata(
            sample_rate=sr,
            channels=1,
            sample_width=2,
            duration_seconds=len(audio_data) / sr,
            frame_count=len(audio_data),
        )
        return Audio(audio_data, metadata)

    def generate_audio(
        self,
        text: str,
        voice_preset: str | None = None,
        voice_sample: Audio | None = None,
    ) -> Audio:
        """Generate speech audio from text."""
        effective_voice = voice_preset or self.voice

        if self.model_size == "qwen3" or voice_sample is not None:
            if voice_sample is None:
                raise ValueError(
                    "voice_sample is required for Qwen3-TTS voice cloning. "
                    "Provide an Audio sample of the voice to clone."
                )
            return self._generate_qwen3(text, voice_sample)

        return self._generate_local(text, effective_voice)


class TextToMusic:
    """Generates music from text descriptions using MusicGen."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._processor: Any = None
        self._model: Any = None
        self._device: str | None = None

    def _init_local(self) -> None:
        """Initialize local MusicGen model."""
        import os

        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        requested_device = self.device
        self._device = select_device(self.device, mps_allowed=True)

        model_name = "facebook/musicgen-small"
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        self._model.to(self._device)
        self.device = self._device
        log_device_initialization(
            "TextToMusic",
            requested_device=requested_device,
            resolved_device=self._device,
        )

    def generate_audio(self, text: str, max_new_tokens: int = 256) -> Audio:
        """Generate music audio from text description."""
        if self._model is None:
            self._init_local()

        inputs = self._processor(text=[text], padding=True, return_tensors="pt")
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
        audio_values = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        sampling_rate = self._model.config.audio_encoder.sampling_rate

        audio_data = audio_values[0, 0].cpu().float().numpy()

        metadata = AudioMetadata(
            sample_rate=sampling_rate,
            channels=1,
            sample_width=2,
            duration_seconds=len(audio_data) / sampling_rate,
            frame_count=len(audio_data),
        )
        return Audio(audio_data, metadata)

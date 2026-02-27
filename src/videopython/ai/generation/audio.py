"""Audio generation using local models."""

from __future__ import annotations

from typing import Any

from videopython.ai._device import select_device
from videopython.base.audio import Audio, AudioMetadata


class TextToSpeech:
    """Generates speech audio from text using local models.

    Supports Bark (`base`, `small`) and XTTS voice cloning (`xtts`).
    """

    SUPPORTED_LOCAL_MODELS: list[str] = ["base", "small", "xtts"]

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
        self._xtts_model: Any = None

    def _init_local(self) -> None:
        """Initialize local Bark model."""
        from transformers import AutoModel, AutoProcessor

        device = select_device(self.device, mps_allowed=False)

        model_name = "suno/bark" if self.model_size == "base" else "suno/bark-small"
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def _patch_xtts_load_audio(self) -> None:
        """Patch XTTS load_audio to avoid torchcodec dependency issues."""
        import TTS.tts.models.xtts as xtts_module

        def load_audio_soundfile(audiopath: str, sampling_rate: int):
            import soundfile as sf  # type: ignore[import-untyped]
            import torch
            import torchaudio.functional as F  # type: ignore[import-untyped]

            audio_np, sr = sf.read(audiopath, dtype="float32")

            audio = torch.from_numpy(audio_np)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            else:
                audio = audio.T

            if audio.size(0) != 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if sr != sampling_rate:
                audio = F.resample(audio, sr, sampling_rate)

            return audio

        xtts_module.load_audio = load_audio_soundfile

    def _init_xtts(self) -> None:
        """Initialize XTTS-v2 model for voice cloning."""
        from TTS.api import TTS

        self._patch_xtts_load_audio()

        device = select_device(self.device, mps_allowed=False)

        self._xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.device = device

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

    def _generate_xtts(self, text: str, voice_sample: Audio) -> Audio:
        """Generate speech using XTTS-v2 with voice cloning."""
        import tempfile
        from pathlib import Path

        import numpy as np

        if self._xtts_model is None:
            self._init_xtts()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            voice_sample.save(f.name)
            speaker_wav_path = Path(f.name)

        try:
            audio_list = self._xtts_model.tts(
                text=text,
                speaker_wav=str(speaker_wav_path),
                language=self.language,
            )
            audio_data = np.array(audio_list, dtype=np.float32)
            sample_rate = 24000

            metadata = AudioMetadata(
                sample_rate=sample_rate,
                channels=1,
                sample_width=2,
                duration_seconds=len(audio_data) / sample_rate,
                frame_count=len(audio_data),
            )
            return Audio(audio_data, metadata)
        finally:
            speaker_wav_path.unlink()

    def generate_audio(
        self,
        text: str,
        voice_preset: str | None = None,
        voice_sample: Audio | None = None,
    ) -> Audio:
        """Generate speech audio from text."""
        effective_voice = voice_preset or self.voice

        if self.model_size == "xtts" or voice_sample is not None:
            if voice_sample is None:
                raise ValueError(
                    "voice_sample is required for XTTS voice cloning. Provide an Audio sample of the voice to clone."
                )
            return self._generate_xtts(text, voice_sample)

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

        self._device = select_device(self.device, mps_allowed=True)

        model_name = "facebook/musicgen-small"
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        self._model.to(self._device)

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

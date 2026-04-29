"""Audio generation using local models."""

from __future__ import annotations

from typing import Any

from videopython.ai._device import log_device_initialization, release_device_memory, select_device
from videopython.base.audio import Audio, AudioMetadata


class TextToSpeech:
    """Generates speech audio from text using Chatterbox Multilingual.

    Backed by Chatterbox Multilingual (Resemble AI). When ``voice_sample`` is
    provided to ``generate_audio``, the model clones that voice; otherwise it
    falls back to Chatterbox's built-in default speaker.
    """

    SAMPLE_RATE: int = 24000

    def __init__(
        self,
        voice: Audio | None = None,
        device: str | None = None,
        language: str = "en",
    ):
        self.voice = voice
        self.device = device
        self.language = language
        self._model: Any = None

    def _init_model(self) -> None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore[import-untyped]

        requested_device = self.device
        device = select_device(self.device, mps_allowed=False)

        self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self.device = device
        log_device_initialization(
            "TextToSpeech",
            requested_device=requested_device,
            resolved_device=device,
        )

    def generate_audio(
        self,
        text: str,
        voice_sample: Audio | None = None,
    ) -> Audio:
        """Generate speech audio from text.

        Args:
            text: Text to synthesize.
            voice_sample: Optional voice sample to clone. Falls back to the
                instance's ``voice`` and then to Chatterbox's default speaker.
        """
        import tempfile
        from pathlib import Path

        import numpy as np

        if self._model is None:
            self._init_model()

        effective_sample = voice_sample or self.voice
        speaker_wav_path: Path | None = None

        if effective_sample is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                effective_sample.save(f.name)
                speaker_wav_path = Path(f.name)

        try:
            wav = self._model.generate(
                text=text,
                language_id=self.language,
                audio_prompt_path=str(speaker_wav_path) if speaker_wav_path else None,
            )

            audio_data = wav.cpu().float().numpy().squeeze()
            if audio_data.ndim == 0:
                audio_data = np.array([audio_data], dtype=np.float32)

            metadata = AudioMetadata(
                sample_rate=self.SAMPLE_RATE,
                channels=1,
                sample_width=2,
                duration_seconds=len(audio_data) / self.SAMPLE_RATE,
                frame_count=len(audio_data),
            )
            return Audio(audio_data, metadata)
        finally:
            if speaker_wav_path is not None:
                speaker_wav_path.unlink()

    def unload(self) -> None:
        """Release the TTS model so the next generate_audio() re-initializes.

        Used by low-memory dubbing to free VRAM between pipeline stages.
        """
        self._model = None
        release_device_memory(self.device)


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

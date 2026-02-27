"""Audio source separation using local Demucs models."""

from __future__ import annotations

from typing import Any

from videopython.ai._device import log_device_initialization, select_device
from videopython.ai.dubbing.models import SeparatedAudio
from videopython.base.audio import Audio, AudioMetadata


class AudioSeparator:
    """Separates audio into vocals and background components using Demucs."""

    SUPPORTED_MODELS: list[str] = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"]
    STEM_NAMES = ["drums", "bass", "other", "vocals"]
    STEM_NAMES_6S = ["drums", "bass", "other", "vocals", "guitar", "piano"]

    def __init__(self, model_name: str = "htdemucs", device: str | None = None):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Supported: {self.SUPPORTED_MODELS}")

        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local Demucs model."""
        from demucs.pretrained import get_model

        requested_device = self.device
        device = select_device(self.device, mps_allowed=False)

        self._model = get_model(self.model_name)
        self._model.to(device)
        self._model.eval()
        self.device = device
        log_device_initialization(
            "AudioSeparator",
            requested_device=requested_device,
            resolved_device=device,
        )

    def _separate_local(self, audio: Audio) -> SeparatedAudio:
        """Separate audio using local Demucs model."""
        import numpy as np
        import torch
        from demucs.apply import apply_model

        if self._model is None:
            self._init_local()

        target_sr = self._model.samplerate

        if audio.metadata.channels == 1:
            audio = audio._to_stereo()

        if audio.metadata.sample_rate != target_sr:
            audio = audio.resample(target_sr)

        audio_data = audio.data
        if audio_data.ndim == 1:
            audio_data = np.stack([audio_data, audio_data])
        elif audio_data.ndim == 2:
            audio_data = audio_data.T

        wav = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        wav = wav.to(self.device)

        with torch.no_grad():
            sources = apply_model(self._model, wav, device=self.device)

        sources_np = sources[0].cpu().numpy()

        stem_names = self.STEM_NAMES_6S if self.model_name == "htdemucs_6s" else self.STEM_NAMES

        stems: dict[str, Audio] = {}
        for i, name in enumerate(stem_names):
            stem_data = sources_np[i].T

            metadata = AudioMetadata(
                sample_rate=target_sr,
                channels=2,
                sample_width=2,
                duration_seconds=stem_data.shape[0] / target_sr,
                frame_count=stem_data.shape[0],
            )
            stems[name] = Audio(stem_data.astype(np.float32), metadata)

        vocals = stems["vocals"]

        non_vocal_stems = [stems[name] for name in stem_names if name != "vocals"]
        background_data = np.zeros_like(vocals.data)
        for stem in non_vocal_stems:
            background_data += stem.data

        max_val = np.max(np.abs(background_data))
        if max_val > 1.0:
            background_data = background_data / max_val

        background = Audio(background_data.astype(np.float32), vocals.metadata)

        music_stems = ["drums", "bass", "other"]
        if self.model_name == "htdemucs_6s":
            music_stems.extend(["guitar", "piano"])

        music_data = np.zeros_like(vocals.data)
        for name in music_stems:
            if name in stems:
                music_data += stems[name].data

        max_val = np.max(np.abs(music_data))
        if max_val > 1.0:
            music_data = music_data / max_val

        music = Audio(music_data.astype(np.float32), vocals.metadata)

        return SeparatedAudio(
            vocals=vocals,
            background=background,
            original=audio,
            music=music,
            effects=None,
        )

    def separate(self, audio: Audio) -> SeparatedAudio:
        """Separate audio into vocals and background components."""
        return self._separate_local(audio)

    def extract_vocals(self, audio: Audio) -> Audio:
        """Convenience method to extract only vocals from audio."""
        return self.separate(audio).vocals

    def extract_background(self, audio: Audio) -> Audio:
        """Convenience method to extract only background from audio."""
        return self.separate(audio).background

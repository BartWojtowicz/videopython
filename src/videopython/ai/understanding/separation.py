"""Audio source separation using local Demucs models."""

from __future__ import annotations

from typing import Any

from videopython.ai._device import log_device_initialization, release_device_memory, select_device
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
        """Separate audio using local Demucs model.

        Keeps the input tensor on CPU and passes ``device=self.device`` to
        ``apply_model`` so per-chunk compute runs on GPU while the full
        ``(stems, channels, samples)`` output is stored in CPU RAM. For long
        sources this is the difference between OOM-on-GPU and running cleanly:
        a 2h stereo @ 44.1kHz output is ~10 GB — too big for an 8 GB card but
        comfortable on a 32 GB host.
        """
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

        with torch.no_grad():
            sources = apply_model(self._model, wav, device=self.device)

        sources_np = sources[0].cpu().numpy()
        del sources

        stem_names = self.STEM_NAMES_6S if self.model_name == "htdemucs_6s" else self.STEM_NAMES
        vocals_idx = stem_names.index("vocals")
        non_vocal_indices = [i for i in range(len(stem_names)) if i != vocals_idx]

        vocals_data = sources_np[vocals_idx].T
        background_data = sources_np[non_vocal_indices].sum(axis=0).T
        del sources_np

        max_val = np.max(np.abs(background_data))
        if max_val > 1.0:
            background_data /= max_val

        metadata = AudioMetadata(
            sample_rate=target_sr,
            channels=2,
            sample_width=2,
            duration_seconds=vocals_data.shape[0] / target_sr,
            frame_count=vocals_data.shape[0],
        )
        vocals = Audio(np.ascontiguousarray(vocals_data, dtype=np.float32), metadata)
        background = Audio(np.ascontiguousarray(background_data, dtype=np.float32), metadata)

        return SeparatedAudio(
            vocals=vocals,
            background=background,
            original=audio,
            music=None,
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

    def unload(self) -> None:
        """Release the Demucs model so the next separate() re-initializes.

        Used by low-memory dubbing to free VRAM between pipeline stages.
        """
        self._model = None
        release_device_memory(self.device)

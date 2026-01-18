"""Audio source separation using Demucs."""

from __future__ import annotations

from typing import Any

from videopython.ai.backends import AudioSeparatorBackend, UnsupportedBackendError
from videopython.ai.config import get_default_backend
from videopython.ai.dubbing.models import SeparatedAudio
from videopython.base.audio import Audio, AudioMetadata


class AudioSeparator:
    """Separates audio into different components (vocals, music, effects).

    Uses Demucs for high-quality source separation, isolating vocals
    from background music and sound effects.

    Example:
        >>> from videopython.ai.understanding.separation import AudioSeparator
        >>> from videopython.base.audio import Audio
        >>>
        >>> separator = AudioSeparator()
        >>> audio = Audio.from_path("audio.mp3")
        >>> separated = separator.separate(audio)
        >>> separated.vocals.save("vocals.wav")
        >>> separated.background.save("background.wav")
    """

    SUPPORTED_BACKENDS: list[str] = ["local"]
    SUPPORTED_MODELS: list[str] = ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra"]

    # Demucs outputs these stems
    STEM_NAMES = ["drums", "bass", "other", "vocals"]
    STEM_NAMES_6S = ["drums", "bass", "other", "vocals", "guitar", "piano"]

    def __init__(
        self,
        backend: AudioSeparatorBackend | None = None,
        model_name: str = "htdemucs",
        device: str | None = None,
    ):
        """Initialize the audio separator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            model_name: Demucs model to use:
                - 'htdemucs': Default hybrid transformer model (4 stems)
                - 'htdemucs_ft': Fine-tuned version for better vocals
                - 'htdemucs_6s': 6-stem version (adds guitar, piano)
                - 'mdx_extra': MDX-Net based model
            device: Device for local backend ('cuda', 'mps', 'cpu').
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("audio_separator")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Supported: {self.SUPPORTED_MODELS}")

        self.backend: AudioSeparatorBackend = resolved_backend  # type: ignore[assignment]
        self.model_name = model_name
        self.device = device

        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local Demucs model."""
        import torch
        from demucs.pretrained import get_model

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            # MPS has limitations with large conv operations in Demucs
            # (Output channels > 65536 not supported), so use CPU
            else:
                device = "cpu"

        self._model = get_model(self.model_name)
        self._model.to(device)
        self._model.eval()
        self.device = device

    def _separate_local(self, audio: Audio) -> SeparatedAudio:
        """Separate audio using local Demucs model."""
        import numpy as np
        import torch
        from demucs.apply import apply_model

        if self._model is None:
            self._init_local()

        # Demucs expects stereo audio at model's sample rate
        target_sr = self._model.samplerate

        # Convert to stereo if mono
        if audio.metadata.channels == 1:
            audio = audio._to_stereo()

        # Resample if needed
        if audio.metadata.sample_rate != target_sr:
            audio = audio.resample(target_sr)

        # Prepare tensor: (batch, channels, samples)
        audio_data = audio.data
        if audio_data.ndim == 1:
            audio_data = np.stack([audio_data, audio_data])
        elif audio_data.ndim == 2:
            audio_data = audio_data.T  # (samples, channels) -> (channels, samples)

        # Add batch dimension
        wav = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
        wav = wav.to(self.device)

        # Apply separation
        with torch.no_grad():
            sources = apply_model(self._model, wav, device=self.device)

        # sources shape: (batch, stems, channels, samples)
        sources_np = sources[0].cpu().numpy()  # Remove batch dimension

        # Determine stem names based on model
        if self.model_name == "htdemucs_6s":
            stem_names = self.STEM_NAMES_6S
        else:
            stem_names = self.STEM_NAMES

        # Create stem dictionary
        stems = {}
        for i, name in enumerate(stem_names):
            stem_data = sources_np[i]  # (channels, samples)
            stem_data = stem_data.T  # (samples, channels)

            metadata = AudioMetadata(
                sample_rate=target_sr,
                channels=2,
                sample_width=2,
                duration_seconds=stem_data.shape[0] / target_sr,
                frame_count=stem_data.shape[0],
            )
            stems[name] = Audio(stem_data.astype(np.float32), metadata)

        # Create vocals track
        vocals = stems["vocals"]

        # Create background by mixing all non-vocal stems
        non_vocal_stems = [stems[name] for name in stem_names if name != "vocals"]
        background_data = np.zeros_like(vocals.data)
        for stem in non_vocal_stems:
            background_data += stem.data

        # Normalize to prevent clipping
        max_val = np.max(np.abs(background_data))
        if max_val > 1.0:
            background_data = background_data / max_val

        background = Audio(background_data.astype(np.float32), vocals.metadata)

        # Create music track (drums + bass + other, or more for 6s model)
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
            effects=None,  # Demucs doesn't separate sound effects
        )

    def separate(self, audio: Audio) -> SeparatedAudio:
        """Separate audio into vocals and background components.

        Args:
            audio: Audio to separate.

        Returns:
            SeparatedAudio with isolated vocal and background tracks.

        Raises:
            UnsupportedBackendError: If backend is not supported.
        """
        if self.backend == "local":
            return self._separate_local(audio)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)

    def extract_vocals(self, audio: Audio) -> Audio:
        """Convenience method to extract only vocals from audio.

        Args:
            audio: Audio to extract vocals from.

        Returns:
            Audio containing only the vocal track.
        """
        separated = self.separate(audio)
        return separated.vocals

    def extract_background(self, audio: Audio) -> Audio:
        """Convenience method to extract only background from audio.

        Args:
            audio: Audio to extract background from.

        Returns:
            Audio containing only the background (music, effects).
        """
        separated = self.separate(audio)
        return separated.background

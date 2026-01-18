"""Audio generation with multi-backend support."""

from __future__ import annotations

from typing import Any

from videopython.ai.backends import (
    TextToMusicBackend,
    TextToSpeechBackend,
    UnsupportedBackendError,
    get_api_key,
)
from videopython.ai.config import get_default_backend
from videopython.base.audio import Audio, AudioMetadata


class TextToSpeech:
    """Generates speech audio from text.

    Supports multiple backends and models:
    - local (Bark): General TTS with emotion support
    - local (XTTS): Voice cloning TTS with speaker reference
    - openai: OpenAI TTS API
    - elevenlabs: ElevenLabs TTS API

    Example:
        >>> # Standard TTS
        >>> tts = TextToSpeech(backend="local")
        >>> audio = tts.generate_audio("Hello world")
        >>>
        >>> # Voice cloning with XTTS
        >>> tts = TextToSpeech(backend="local", model_size="xtts")
        >>> voice_sample = Audio.from_path("speaker_sample.wav")
        >>> audio = tts.generate_audio("Hello world", voice_sample=voice_sample)
    """

    SUPPORTED_BACKENDS: list[str] = ["local", "openai", "elevenlabs"]
    SUPPORTED_LOCAL_MODELS: list[str] = ["base", "small", "xtts"]

    def __init__(
        self,
        backend: TextToSpeechBackend | None = None,
        model_size: str = "base",
        voice: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
        language: str = "en",
    ):
        """Initialize text-to-speech generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            model_size: Model size for local backend ('base', 'small', or 'xtts').
                'xtts' enables voice cloning with Coqui XTTS-v2.
            voice: Voice to use (backend-specific).
            api_key: API key for cloud backends. If None, reads from environment.
            device: Device for local backend ('cuda', 'mps', or 'cpu').
            language: Language code for XTTS (e.g., 'en', 'es', 'fr').
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("text_to_speech")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: TextToSpeechBackend = resolved_backend  # type: ignore[assignment]
        self.model_size = model_size
        self.voice = voice
        self.api_key = api_key
        self.device = device
        self.language = language
        self._model: Any = None
        self._processor: Any = None
        self._xtts_model: Any = None

    def _init_local(self) -> None:
        """Initialize local Bark model."""
        import torch
        from transformers import AutoModel, AutoProcessor

        if self.model_size not in ["base", "small"]:
            raise ValueError(f"model_size must be 'base' or 'small', got '{self.model_size}'")

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            # MPS has limitations with large conv operations in Bark's encodec
            # (Output channels > 65536 not supported), so use CPU
            else:
                device = "cpu"

        model_name = "suno/bark" if self.model_size == "base" else "suno/bark-small"
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    def _init_xtts(self) -> None:
        """Initialize XTTS-v2 model for voice cloning."""
        import torch
        from TTS.api import TTS

        # Monkey-patch XTTS load_audio to use soundfile instead of torchaudio
        # This avoids torchcodec dependency issues
        self._patch_xtts_load_audio()

        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                # XTTS may have issues with MPS, fall back to CPU
                device = "cpu"
            else:
                device = "cpu"

        # Load XTTS-v2 model
        self._xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        self.device = device

    def _patch_xtts_load_audio(self) -> None:
        """Patch XTTS load_audio to use soundfile instead of torchaudio.

        This avoids issues with torchcodec on some systems.
        """
        import TTS.tts.models.xtts as xtts_module

        def load_audio_soundfile(audiopath: str, sampling_rate: int):
            import soundfile as sf  # type: ignore[import-untyped]
            import torch
            import torchaudio.functional as F  # type: ignore[import-untyped]

            # Load audio with soundfile
            audio_np, sr = sf.read(audiopath, dtype="float32")

            # Convert to torch tensor [channels, samples]
            audio = torch.from_numpy(audio_np)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            else:
                # [samples, channels] -> [channels, samples]
                audio = audio.T

            # Stereo to mono if needed
            if audio.size(0) != 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Resample if needed
            if sr != sampling_rate:
                audio = F.resample(audio, sr, sampling_rate)

            return audio

        # Replace the load_audio function in the xtts module
        xtts_module.load_audio = load_audio_soundfile

    def _generate_local(
        self,
        text: str,
        voice_preset: str | None,
    ) -> Audio:
        """Generate speech using local Bark model."""
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

    def _generate_xtts(
        self,
        text: str,
        voice_sample: Audio,
    ) -> Audio:
        """Generate speech using XTTS-v2 with voice cloning.

        Args:
            text: Text to synthesize.
            voice_sample: Audio sample of the voice to clone (3-10 seconds recommended).

        Returns:
            Generated speech audio in the cloned voice.
        """
        import tempfile
        from pathlib import Path

        import numpy as np

        if self._xtts_model is None:
            self._init_xtts()

        # XTTS requires a file path for the speaker reference
        # Save the voice sample to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            voice_sample.save(f.name)
            speaker_wav_path = Path(f.name)

        try:
            # Generate speech with voice cloning
            # tts_to_file returns path, tts returns list of audio samples
            audio_list = self._xtts_model.tts(
                text=text,
                speaker_wav=str(speaker_wav_path),
                language=self.language,
            )

            # Convert to numpy array
            audio_data = np.array(audio_list, dtype=np.float32)

            # XTTS outputs at 24kHz
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
            # Clean up temp file
            speaker_wav_path.unlink()

    def _generate_openai(self, text: str) -> Audio:
        """Generate speech using OpenAI TTS."""
        import numpy as np
        from openai import OpenAI

        api_key = get_api_key("openai", self.api_key)
        client = OpenAI(api_key=api_key)

        voice = self.voice or "alloy"
        response = client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,  # type: ignore
            input=text,
            response_format="pcm",
        )

        # OpenAI returns raw PCM at 24kHz, 16-bit, mono
        audio_bytes = response.read()
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sample_rate = 24000

        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=len(audio_data) / sample_rate,
            frame_count=len(audio_data),
        )
        return Audio(audio_data, metadata)

    def _generate_elevenlabs(self, text: str) -> Audio:
        """Generate speech using ElevenLabs."""
        import numpy as np
        from elevenlabs import ElevenLabs

        api_key = get_api_key("elevenlabs", self.api_key)
        client = ElevenLabs(api_key=api_key)

        voice = self.voice or "Sarah"

        # Resolve voice name to ID if needed (voice IDs are 20+ chars)
        if len(voice) < 20:
            voices = client.voices.get_all()
            voice_id = None
            for v in voices.voices:
                if v.name and voice.lower() in v.name.lower():
                    voice_id = v.voice_id
                    break
            if voice_id is None:
                raise ValueError(f"Voice '{voice}' not found. Use a voice ID or valid name.")
            voice = voice_id

        # Generate audio - returns generator
        audio_chunks = []
        for chunk in client.text_to_speech.convert(
            voice_id=voice,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_24000",
        ):
            audio_chunks.append(chunk)

        audio_bytes = b"".join(audio_chunks)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        sample_rate = 24000

        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=len(audio_data) / sample_rate,
            frame_count=len(audio_data),
        )
        return Audio(audio_data, metadata)

    def generate_audio(
        self,
        text: str,
        voice_preset: str | None = None,
        voice_sample: Audio | None = None,
    ) -> Audio:
        """Generate speech audio from text.

        Args:
            text: Text to synthesize. For local backend, can include emotion markers
                  like [laughs], [sighs].
            voice_preset: Voice preset (backend-specific). For local backend, use
                          IDs like "v2/en_speaker_0".
            voice_sample: Audio sample for voice cloning (XTTS only).
                          Should be 3-10 seconds of clean speech.

        Returns:
            Generated speech audio.
        """
        effective_voice = voice_preset or self.voice

        if self.backend == "local":
            # Use XTTS if model_size is "xtts" or voice_sample is provided
            if self.model_size == "xtts" or voice_sample is not None:
                if voice_sample is None:
                    raise ValueError(
                        "voice_sample is required for XTTS voice cloning. "
                        "Provide an Audio sample of the voice to clone."
                    )
                return self._generate_xtts(text, voice_sample)
            else:
                return self._generate_local(text, effective_voice)
        elif self.backend == "openai":
            return self._generate_openai(text)
        elif self.backend == "elevenlabs":
            return self._generate_elevenlabs(text)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)


class TextToMusic:
    """Generates music from text descriptions."""

    SUPPORTED_BACKENDS: list[str] = ["local"]

    def __init__(
        self,
        backend: TextToMusicBackend | None = None,
    ):
        """Initialize text-to-music generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("text_to_music")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: TextToMusicBackend = resolved_backend  # type: ignore[assignment]
        self._processor: Any = None
        self._model: Any = None
        self._device: str = "cpu"

    def _init_local(self) -> None:
        """Initialize local MusicGen model."""
        import os

        import torch
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        # Enable MPS fallback for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        if torch.cuda.is_available():
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        model_name = "facebook/musicgen-small"
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        self._model.to(self._device)

    def _generate_local(self, text: str, max_new_tokens: int) -> Audio:
        """Generate music using local MusicGen model."""
        if self._model is None:
            self._init_local()

        inputs = self._processor(text=[text], padding=True, return_tensors="pt")
        # Move inputs to the same device as the model
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

    def generate_audio(self, text: str, max_new_tokens: int = 256) -> Audio:
        """Generate music audio from text description.

        Args:
            text: Text description of desired music.
            max_new_tokens: Maximum length of generated audio in tokens.

        Returns:
            Generated music audio.
        """
        if self.backend == "local":
            return self._generate_local(text, max_new_tokens)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)

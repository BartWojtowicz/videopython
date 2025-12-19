"""Audio generation with multi-backend support."""

from __future__ import annotations

import asyncio

from soundpython import Audio, AudioMetadata

from videopython.ai.backends import (
    TextToMusicBackend,
    TextToSpeechBackend,
    UnsupportedBackendError,
    get_api_key,
)
from videopython.ai.config import get_default_backend, get_replicate_model


class TextToSpeech:
    """Generates speech audio from text."""

    SUPPORTED_BACKENDS: list[str] = ["local", "openai", "elevenlabs"]

    def __init__(
        self,
        backend: TextToSpeechBackend | None = None,
        model_size: str = "base",
        voice: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
    ):
        """Initialize text-to-speech generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            model_size: Model size for local backend ('base' or 'small').
            voice: Voice to use (backend-specific).
            api_key: API key for cloud backends. If None, reads from environment.
            device: Device for local backend ('cuda' or 'cpu').
        """
        if backend is None:
            backend = get_default_backend("text_to_speech")  # type: ignore

        if backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(backend, self.SUPPORTED_BACKENDS)

        self.backend: TextToSpeechBackend = backend  # type: ignore
        self.model_size = model_size
        self.voice = voice
        self.api_key = api_key
        self.device = device
        self._model = None
        self._processor = None

    def _init_local(self) -> None:
        """Initialize local Bark model."""
        import torch
        from transformers import AutoModel, AutoProcessor

        if self.model_size not in ["base", "small"]:
            raise ValueError(f"model_size must be 'base' or 'small', got '{self.model_size}'")

        device = self.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name = "suno/bark" if self.model_size == "base" else "suno/bark-small"
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device

    async def _generate_local(
        self,
        text: str,
        voice_preset: str | None,
    ) -> Audio:
        """Generate speech using local Bark model."""
        import torch

        if self._model is None:
            await asyncio.to_thread(self._init_local)

        def _run_model() -> Audio:
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

        return await asyncio.to_thread(_run_model)

    async def _generate_openai(self, text: str) -> Audio:
        """Generate speech using OpenAI TTS."""

        import numpy as np
        from openai import AsyncOpenAI

        api_key = get_api_key("openai", self.api_key)
        client = AsyncOpenAI(api_key=api_key)

        voice = self.voice or "alloy"
        response = await client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,  # type: ignore
            input=text,
            response_format="pcm",
        )

        # OpenAI returns raw PCM at 24kHz, 16-bit, mono
        audio_bytes = await response.aread()
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

    async def _generate_elevenlabs(self, text: str) -> Audio:
        """Generate speech using ElevenLabs."""
        import numpy as np
        from elevenlabs import AsyncElevenLabs

        api_key = get_api_key("elevenlabs", self.api_key)
        client = AsyncElevenLabs(api_key=api_key)

        voice = self.voice or "Rachel"

        # Generate audio
        audio_generator = await client.text_to_speech.convert(
            voice_id=voice,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format="pcm_24000",
        )

        # Collect all chunks
        audio_chunks = []
        async for chunk in audio_generator:
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

    async def generate_audio(
        self,
        text: str,
        voice_preset: str | None = None,
    ) -> Audio:
        """Generate speech audio from text.

        Args:
            text: Text to synthesize. For local backend, can include emotion markers
                  like [laughs], [sighs].
            voice_preset: Voice preset (backend-specific). For local backend, use
                          IDs like "v2/en_speaker_0".

        Returns:
            Generated speech audio.
        """
        effective_voice = voice_preset or self.voice

        if self.backend == "local":
            return await self._generate_local(text, effective_voice)
        elif self.backend == "openai":
            return await self._generate_openai(text)
        elif self.backend == "elevenlabs":
            return await self._generate_elevenlabs(text)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)


class TextToMusic:
    """Generates music from text descriptions."""

    SUPPORTED_BACKENDS: list[str] = ["local", "replicate"]

    def __init__(
        self,
        backend: TextToMusicBackend | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize text-to-music generator.

        Args:
            backend: Backend to use. If None, uses config default or 'local'.
            model: Model to use (only for 'replicate' backend).
            api_key: API key for cloud backends. If None, reads from environment.
        """
        if backend is None:
            backend = get_default_backend("text_to_music")  # type: ignore

        if backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(backend, self.SUPPORTED_BACKENDS)

        self.backend: TextToMusicBackend = backend  # type: ignore
        self.model = model
        self.api_key = api_key
        self._processor = None
        self._model = None

    def _init_local(self) -> None:
        """Initialize local MusicGen model."""
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        model_name = "facebook/musicgen-small"
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    async def _generate_local(self, text: str, max_new_tokens: int) -> Audio:
        """Generate music using local MusicGen model."""
        if self._model is None:
            await asyncio.to_thread(self._init_local)

        def _run_model() -> Audio:
            inputs = self._processor(text=[text], padding=True, return_tensors="pt")
            audio_values = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
            sampling_rate = self._model.config.audio_encoder.sampling_rate

            audio_data = audio_values[0, 0].float().numpy()

            metadata = AudioMetadata(
                sample_rate=sampling_rate,
                channels=1,
                sample_width=2,
                duration_seconds=len(audio_data) / sampling_rate,
                frame_count=len(audio_data),
            )
            return Audio(audio_data, metadata)

        return await asyncio.to_thread(_run_model)

    async def _generate_replicate(self, text: str, duration: float) -> Audio:
        """Generate music using Replicate."""
        import numpy as np
        import replicate

        api_key = get_api_key("replicate", self.api_key)
        client = replicate.Client(api_token=api_key)

        model_name = self.model or get_replicate_model("text_to_music")

        def _run_replicate() -> str:
            output = client.run(
                model_name,
                input={"prompt": text, "duration": int(duration)},
            )
            if isinstance(output, list):
                return output[0]
            return output

        audio_url = await asyncio.to_thread(_run_replicate)

        # Download and convert audio
        import httpx

        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(audio_url, timeout=60.0)
            response.raise_for_status()

            # Parse audio file (assuming WAV or similar)
            import io
            import wave

            audio_bytes = io.BytesIO(response.content)

            # Try to read as WAV
            try:
                with wave.open(audio_bytes, "rb") as wav:
                    sample_rate = wav.getframerate()
                    n_frames = wav.getnframes()
                    audio_data = np.frombuffer(wav.readframes(n_frames), dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768.0

                    metadata = AudioMetadata(
                        sample_rate=sample_rate,
                        channels=wav.getnchannels(),
                        sample_width=wav.getsampwidth(),
                        duration_seconds=n_frames / sample_rate,
                        frame_count=n_frames,
                    )
                    return Audio(audio_data, metadata)
            except wave.Error:
                # If not WAV, use soundpython to load
                from soundpython import Audio as SoundAudio

                audio_bytes.seek(0)
                return SoundAudio.from_bytes(audio_bytes.read())

    async def generate_audio(self, text: str, max_new_tokens: int = 256) -> Audio:
        """Generate music audio from text description.

        Args:
            text: Text description of desired music.
            max_new_tokens: Maximum length of generated audio in tokens (local backend).
                           For replicate, this is converted to approximate duration.

        Returns:
            Generated music audio.
        """
        if self.backend == "local":
            return await self._generate_local(text, max_new_tokens)
        elif self.backend == "replicate":
            # Convert tokens to approximate duration (MusicGen generates ~50 tokens/sec)
            duration = max_new_tokens / 50.0
            return await self._generate_replicate(text, duration)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)

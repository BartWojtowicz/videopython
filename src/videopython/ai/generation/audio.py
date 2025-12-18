import torch
from soundpython import Audio, AudioMetadata
from transformers import AutoModel, AutoProcessor, MusicgenForConditionalGeneration

MUSIC_GENERATION_MODEL_SMALL = "facebook/musicgen-small"
TTS_MODEL = "suno/bark"


class TextToSpeech:
    def __init__(
        self,
        device: str | None = None,
        model_size: str = "base",
    ):
        """Initialize text-to-speech model using Bark.

        Args:
            device: Device to run on ('cuda' or 'cpu'), defaults to auto-detect.
            model_size: Model size - 'base' (better quality) or 'small' (faster).
        """
        if model_size not in ["base", "small"]:
            raise ValueError(f"model_size must be 'base' or 'small', got '{model_size}'")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        model_name = TTS_MODEL if model_size == "base" else "suno/bark-small"

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)

    def generate_audio(
        self,
        text: str,
        voice_preset: str | None = None,
        do_sample: bool = True,
    ) -> Audio:
        """Generate speech audio from text with support for emotion markers like [laughs], [sighs].

        Args:
            text: Text to synthesize, can include emotion markers.
            voice_preset: Voice preset ID (e.g., "v2/en_speaker_0"), defaults to None.
            do_sample: Use sampling for more natural speech, defaults to True.

        Returns:
            Generated speech audio at 24 kHz.
        """
        inputs = self.processor(text=[text], return_tensors="pt", voice_preset=voice_preset)

        # Move inputs to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            speech_values = self.model.generate(**inputs, do_sample=do_sample)

        # Convert to float32 numpy array (same as TextToMusic)
        audio_data = speech_values.cpu().float().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate

        metadata = AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=len(audio_data) / sample_rate,
            frame_count=len(audio_data),
        )

        return Audio(audio_data, metadata)


class TextToMusic:
    """Generates music from text using the Musicgen model."""

    def __init__(self) -> None:
        """Initialize text-to-music generator with Musicgen small model."""
        self.processor = AutoProcessor.from_pretrained(MUSIC_GENERATION_MODEL_SMALL)
        self.model = MusicgenForConditionalGeneration.from_pretrained(MUSIC_GENERATION_MODEL_SMALL)

    def generate_audio(self, text: str, max_new_tokens: int) -> Audio:
        """Generate music audio from text description.

        Args:
            text: Text description of desired music.
            max_new_tokens: Maximum length of generated audio in tokens.

        Returns:
            Generated music audio.
        """
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        audio_values = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        sampling_rate = self.model.config.audio_encoder.sampling_rate

        # Convert to float32 and normalize to [-1, 1]
        audio_data = audio_values[0, 0].float().numpy()

        metadata = AudioMetadata(
            sample_rate=sampling_rate,
            channels=1,
            sample_width=2,
            duration_seconds=len(audio_data) / sampling_rate,
            frame_count=len(audio_data),
        )

        return Audio(audio_data, metadata)

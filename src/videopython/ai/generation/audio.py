from typing import Optional

import torch
from soundpython import Audio, AudioMetadata
from transformers import AutoModel, AutoProcessor, MusicgenForConditionalGeneration

MUSIC_GENERATION_MODEL_SMALL = "facebook/musicgen-small"
TTS_MODEL = "suno/bark"


class TextToSpeech:
    def __init__(
        self,
        device: Optional[str] = None,
        model_size: str = "base",
    ):
        """
        Initialize TextToSpeech with Bark model from Suno AI.

        Bark generates highly realistic, multilingual speech with emotions, laughter,
        and other audio effects. It doesn't require voice cloning files.

        Args:
            device: Device to run the model on ('cuda' or 'cpu'). If None, automatically
                   selects cuda if available.
            model_size: Model size to use. Options: 'base' (suno/bark) or 'small' (suno/bark-small).
                       'base' offers better quality, 'small' is faster. Defaults to 'base'.

        Example:
            tts = TextToSpeech()
            audio = tts.generate_audio("Hello world!")

            # With emotion
            audio = tts.generate_audio("Hello world! [laughs]")
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
        voice_preset: Optional[str] = None,
        do_sample: bool = True,
    ) -> Audio:
        """
        Generate speech audio from text.

        Bark supports special markup for effects:
        - [laughs], [sighs], [gasps], [clears throat]
        - CAPITALIZATION for emphasis
        - ... for hesitation
        - MAN/WOMAN for speaker change

        Args:
            text: The text to synthesize into speech. Can include emotion markers
                 like [laughs], [sighs], etc.
            voice_preset: Optional voice preset (e.g., "v2/en_speaker_0" through "v2/en_speaker_9").
                         If None, uses default voice. See Bark docs for available presets.
            do_sample: Whether to use sampling for generation. True gives more natural,
                      varied speech. False is more deterministic. Defaults to True.

        Returns:
            Audio object containing the generated speech at 24 kHz.

        Example:
            tts = TextToSpeech()

            # Simple text
            audio = tts.generate_audio("Hello, how are you?")

            # With emotion
            audio = tts.generate_audio("That's hilarious! [laughs]")

            # With voice preset
            audio = tts.generate_audio("Hello!", voice_preset="v2/en_speaker_6")
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
    def __init__(self) -> None:
        """
        Generates music from text using the Musicgen model.
        Check the license for the model before using it.
        """
        self.processor = AutoProcessor.from_pretrained(MUSIC_GENERATION_MODEL_SMALL)
        self.model = MusicgenForConditionalGeneration.from_pretrained(MUSIC_GENERATION_MODEL_SMALL)

    def generate_audio(self, text: str, max_new_tokens: int) -> Audio:
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

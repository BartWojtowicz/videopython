from kokoro import KPipeline
from soundpython import Audio, AudioMetadata
from transformers import AutoProcessor, MusicgenForConditionalGeneration

TEXT_TO_SPEECH_LANG = "a"
TEXT_TO_SPEECH_VOICE = "af_heart"
TEXT_TO_SPEECH_SAMPLE_RATE = 24000
MUSIC_GENERATION_MODEL_SMALL = "facebook/musicgen-small"


class TextToSpeech:
    def __init__(self, lang_code: str = TEXT_TO_SPEECH_LANG, voice: str = TEXT_TO_SPEECH_VOICE):
        """Initialize Kokoro text-to-speech model.

        Args:
            lang_code: Language code (default: 'a' for English American)
            voice: Voice preset (default: 'af_heart')
        """
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self.sample_rate = TEXT_TO_SPEECH_SAMPLE_RATE

    def generate_audio(self, text: str) -> Audio:
        """Generate speech audio from text.

        Args:
            text: Input text to convert to speech

        Returns:
            Audio object containing the generated speech
        """
        # Generate audio using Kokoro pipeline
        generator = self.pipeline(text, voice=self.voice)

        # Kokoro returns a generator, we take the last (complete) output
        audio_data = None
        for _, _, audio in generator:
            audio_data = audio

        if audio_data is None:
            raise RuntimeError("Failed to generate audio from text")

        metadata = AudioMetadata(
            sample_rate=self.sample_rate,
            channels=1,
            sample_width=4,
            duration_seconds=len(audio_data) / self.sample_rate,
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
            sample_width=4,
            duration_seconds=len(audio_data) / sampling_rate,
            frame_count=len(audio_data),
        )

        return Audio(audio_data, metadata)

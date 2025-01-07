import torch
from soundpython import Audio, AudioMetadata
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    MusicgenForConditionalGeneration,
    VitsModel,
)

TEXT_TO_SPEECH_MODEL = "facebook/mms-tts-eng"
MUSIC_GENERATION_MODEL_SMALL = "facebook/musicgen-small"


class TextToSpeech:
    def __init__(self):
        self.pipeline = VitsModel.from_pretrained(TEXT_TO_SPEECH_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_TO_SPEECH_MODEL)

    def generate_audio(self, text: str) -> Audio:
        tokenized = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.pipeline(**tokenized).waveform

        # Convert to float32 and normalize to [-1, 1]
        audio_data = output.T.float().numpy()

        metadata = AudioMetadata(
            sample_rate=self.pipeline.config.sampling_rate,
            channels=1,
            sample_width=4,
            duration_seconds=len(audio_data) / self.pipeline.config.sampling_rate,
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

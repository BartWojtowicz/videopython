import numpy as np
import torch
from pydub import AudioSegment
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

    def generate_audio(self, text: str) -> AudioSegment:
        tokenized = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = self.pipeline(**tokenized).waveform

        output = (output.T.float().numpy() * (2**31 - 1)).astype(np.int32)
        audio = AudioSegment(data=output, frame_rate=self.pipeline.config.sampling_rate, sample_width=4, channels=1)
        return audio


class TextToMusic:
    def __init__(self) -> None:
        """
        Generates music from text using the Musicgen model.
        Check the license for the model before using it.
        """
        self.processor = AutoProcessor.from_pretrained(MUSIC_GENERATION_MODEL_SMALL)
        self.model = MusicgenForConditionalGeneration.from_pretrained(MUSIC_GENERATION_MODEL_SMALL)

    def generate_audio(self, text: str, max_new_tokens: int) -> AudioSegment:
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        audio_values = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        output = (audio_values[0, 0].float().numpy() * (2**31 - 1)).astype(np.int32)

        audio = AudioSegment(
            data=output.tobytes(),
            frame_rate=sampling_rate,
            sample_width=4,
            channels=1,
        )
        return audio

import numpy as np
import torch
from pydub import AudioSegment
from transformers import AutoTokenizer, VitsModel

TEXT_TO_SPEECH_MODEL = "facebook/mms-tts-eng"


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

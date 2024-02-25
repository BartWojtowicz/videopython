import os
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydub import AudioSegment

from videopython.utils.common import generate_random_name


class TextToSpeech:
    def __init__(self, openai_key: str | None = None, save_audio: bool = True):
        self.client = OpenAI(api_key=openai_key)
        self._save = save_audio

    def generate_audio(
        self,
        text: str,
        voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
    ) -> AudioSegment:
        filename = generate_random_name(suffix=".mp3")
        output_path = str((Path(os.getcwd()) / filename).resolve())
        response = self.client.audio.speech.create(model="tts-1", voice=voice, input=text)
        response.stream_to_file(output_path)
        audio = AudioSegment.from_file(output_path)
        if self._save:
            print(f"Audio saved to {output_path}")
        else:
            os.remove(output_path)
        return audio

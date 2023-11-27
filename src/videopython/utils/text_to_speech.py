import random
from pathlib import Path

from openai import OpenAI

from videopython.project_config import APIkeys, LocationConfig


def text_to_speech_openai(text: str, voice: str = "alloy"):
    client = OpenAI()
    filename = f"{text[:min(len(text), 6 )].rstrip()}.mp3"
    save_dir = LocationConfig.generated_files_dir / filename
    if save_dir.exists():
        filename = f"{text[:min(len(text), 6 )].rstrip()}{str(random.randint(1,100000000))}.png"
        save_dir = LocationConfig.generated_files_dir / filename

    response = client.audio.speech.create(model="tts-1", voice=voice, input=text)
    response.stream_to_file(save_dir)

    return save_dir

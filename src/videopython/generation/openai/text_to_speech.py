import os
from pathlib import Path

from openai import OpenAI
from pydub import AudioSegment

from videopython.utils.common import generate_random_name


def text_to_speech_openai(
    text: str, voice: str = "alloy", save: bool = True, output_dir: str | None = None
) -> str | AudioSegment:
    client = OpenAI()

    filename = generate_random_name(suffix=".mp3")
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(os.getcwd())
    save_path = output_dir / filename

    response = client.audio.speech.create(model="tts-1", voice=voice, input=text)
    response.stream_to_file(save_path)

    if save:
        return str(save_path.resolve())
    else:
        audio = AudioSegment.from_mp3(str(save_path))
        save_path.unlink()
        return audio

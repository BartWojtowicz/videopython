import os
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydub import AudioSegment

from videopython.utils.common import check_path, generate_random_name


def text_to_speech_openai(
    text: str,
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] = "alloy",
    save: bool = True,
    output_dir: str | None = None,
) -> str | AudioSegment:
    client = OpenAI()

    filename = generate_random_name(suffix=".mp3")
    if output_dir:
        output_path = Path(output_dir) / filename
    else:
        output_path = Path(os.getcwd()) / filename

    output_path = check_path(str(output_path), dir_exists=True, suffix=".mp3")

    response = client.audio.speech.create(model="tts-1", voice=voice, input=text)
    response.stream_to_file(output_path)

    if save:
        return output_path
    else:
        audio = AudioSegment.from_mp3(output_path)
        Path(output_path).unlink()
        return audio

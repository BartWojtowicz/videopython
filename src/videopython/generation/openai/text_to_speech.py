from pathlib import Path
from tempfile import NamedTemporaryFile

from openai import OpenAI
from pydub import AudioSegment


def text_to_speech_openai(text: str, voice: str = "alloy") -> Path:
    client = OpenAI()

    # Create a temporary file to store the audio
    with NamedTemporaryFile(suffix=".mp3") as temp_file:
        temp_path = temp_file.name
        # Generate the audio using OpenAI API and save it to the temporary file
        response = client.audio.speech.create(model="tts-1", voice=voice, input=text)
        response.stream_to_file(temp_path)
        # Load the temporary file into a pydub AudioSegment
        audio_segment = AudioSegment.from_file(temp_path, format="mp3")

    return audio_segment

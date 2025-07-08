# About

Videopython is a minimal video generation and processing library designed with short-form videos in mind, with focus on simplicity and ease of use for both humans and AI agents.

# Setup

## Install ffmpeg
```bash
# Install with brew for MacOS:
brew install ffmpeg
# Install with apt-get for Ubuntu:
sudo apt-get install ffmpeg
```

## Install library

```bash
# Install with your favourite package manager
uv add videopython --extra ai

# pip install works as well :)
pip install videopython[ai]
```

> You can install without `[ai]` dependencies for basic video handling and processing.
> The functionalities found in `videopython.ai` won't work.

# Usage examples

## Basic video editing

```python
from videopython.base.video import Video

# Load videos and print metadata
video1 = Video.from_path("tests/test_data/small_video.mp4")
print(video1)

video2 = Video.from_path("tests/test_data/big_video.mp4")
print(video2)

# Define the transformations
from videopython.base.transforms import CutSeconds, ResampleFPS, Resize, TransformationPipeline

pipeline = TransformationPipeline(
    [CutSeconds(start=1.5, end=6.5), ResampleFPS(fps=30), Resize(width=1000, height=1000)]
)
video1 = pipeline.run(video1)
video2 = pipeline.run(video2)

# Combine videos, add audio and save
from videopython.base.transitions import FadeTransition

fade = FadeTransition(effect_time_seconds=3.0)
video = fade.apply(videos=(video1, video2))
video.add_audio_from_file("tests/test_data/test_audio.mp3")

savepath = video.save()
```

## AI powered examples

### Video Generation

> Using Nvidia A40 or better is recommended for the `videopython.ai` module.
```python
# Generate image and animate it
from videopython.ai.generation import ImageToVideo
from videopython.ai.generation import TextToImage

image = TextToImage().generate_image(prompt="Golden Retriever playing in the park")
video = ImageToVideo().generate_video(image=image, fps=24)

# Video generation directly from prompt
from videopython.ai.generation import TextToVideo
video_gen = TextToVideo()
video = video_gen.generate_video("Dogs playing in the park")
for _ in range(10):
    video += video_gen.generate_video("Dogs playing in the park")
```

### Audio generation
```python
from videopython.base.video import Video
video = Video.from_path("<PATH_TO_VIDEO>")

# Generate music on top of video
from videopython.ai.generation import TextToMusic
text_to_music = TextToMusic()
audio = text_to_music.generate_audio("Happy dogs playing together in a park", max_new_tokens=256)
video.add_audio(audio=audio)

# Add TTS on top of video
from videopython.ai.generation import TextToSpeech
text_to_speech = TextToSpeech()
audio = text_to_speech.generate_audio("Woof woof woof! Woooooof!")
video.add_audio(audio=audio)
```

### Generate and overlay subtitles
```python
from videopython.base.video import Video
video = Video.from_path("<PATH_TO_VIDEO>")

# Generate transcription with timestamps
from videopython.ai.understanding.transcribe import CreateTranscription
transcription = CreateTranscription("base").transcribe(video)
# Initialise object for overlaying. See `TranscriptionOverlay` to see detailed configuration options.
from videopython.base.text.overlay import TranscriptionOverlay
transcription_overlay = TranscriptionOverlay(font_filename="src/tests/test_data/test_font.ttf")

video = transcription_overlay.apply(video, transcription)
video.save()
```

# Development notes

## Project structure

Source code of the project can be found under `src/` directory, along with separate directories for unit tests and mypy stubs.
```
.
└── src
    ├── stubs # Contains stubs for mypy
    ├── tests # Unit tests
    └── videopython # Library code
```

----

The `videopython` library is divided into 2 separate high-level modules:
* `videopython.base`: Contains base classes for handling videos and for basic video editing. There are no imports from `videopython.ai` within the `base` module, which allows users to install light-weight base dependencies to do simple video operations.
* `videopython.ai`: Contains AI-powered functionalities for video generation. It has its own `ai` dependency group, which contains all dependencies required to run AI models.

## Running locally

We are using [uv](https://docs.astral.sh/uv/) as project and package manager. Once you clone the repo and install uv locally, you can use it to sync the dependencies.
```bash
uv sync --all-extras
```

To run the unit tests, you can simply run:
```bash
uv run pytest
```

We also use [Ruff](https://docs.astral.sh/ruff/) for linting/formatting and [mypy](https://github.com/python/mypy) as type checker.
```bash
# Run formatting
uv run ruff format
# Run linting and apply fixes
uv run ruff check --fix
# Run type checks
uv run mypy src/
```

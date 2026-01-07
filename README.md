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
pip install "videopython[ai]"
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

The `videopython.ai` module supports multiple backends - local models and cloud APIs.

### Backend Support

| Class | local | openai | gemini | elevenlabs |
|-------|-------|--------|--------|------------|
| TextToVideo | Zeroscope | - | - | - |
| ImageToVideo | SVD | - | - | - |
| TextToSpeech | Bark | TTS | - | Multilingual v2 |
| TextToMusic | MusicGen | - | - | - |
| TextToImage | SDXL | DALL-E 3 | - | - |
| ImageToText | BLIP | GPT-4o | Gemini | - |
| AudioToText | Whisper | Whisper API | Gemini | - |
| LLMSummarizer | Ollama | GPT-4o | Gemini | - |
| ObjectDetector | YOLO | GPT-4o | Gemini | - |
| TextDetector | EasyOCR | GPT-4o | Gemini | - |
| FaceDetector | OpenCV | - | - | - |
| ShotTypeClassifier | - | GPT-4o | Gemini | - |
| CameraMotionDetector | OpenCV | - | - | - |

Cloud backends require API keys via environment variables:
- `OPENAI_API_KEY` for OpenAI
- `GOOGLE_API_KEY` for Gemini
- `ELEVENLABS_API_KEY` for ElevenLabs

### Video Generation

> Using Nvidia A40 or better is recommended for local video generation.
```python
import asyncio
from videopython.ai.generation import ImageToVideo, TextToImage, TextToVideo

# Generate image with OpenAI DALL-E 3 and animate locally
image = asyncio.run(TextToImage(backend="openai").generate_image("Golden Retriever playing in the park"))
video = asyncio.run(ImageToVideo().generate_video(image=image, fps=24))

# Video generation directly from prompt (local only, requires CUDA)
video_gen = TextToVideo()
video = asyncio.run(video_gen.generate_video("Dogs playing in the park"))
```

### Audio generation
```python
import asyncio
from videopython.base.video import Video
from videopython.ai.generation import TextToMusic, TextToSpeech

video = Video.from_path("<PATH_TO_VIDEO>")

# Generate music (local MusicGen)
audio = asyncio.run(TextToMusic().generate_audio("Happy dogs playing together in a park", max_new_tokens=256))
video.add_audio(audio=audio)

# TTS with OpenAI
audio = asyncio.run(TextToSpeech(backend="openai").generate_audio("Welcome to our video!"))
video.add_audio(audio=audio)

# TTS with ElevenLabs for high-quality voices
audio = asyncio.run(TextToSpeech(backend="elevenlabs", voice="Rachel").generate_audio("This sounds amazing!"))
video.add_audio(audio=audio)
```

### Generate and overlay subtitles
```python
import asyncio
from videopython.base.video import Video
from videopython.ai.understanding import AudioToText

video = Video.from_path("<PATH_TO_VIDEO>")

# Transcribe with OpenAI Whisper API
transcriber = AudioToText(backend="openai")
transcription = asyncio.run(transcriber.transcribe(video))

# Or use local Whisper with speaker diarization
transcriber = AudioToText(backend="local", model_name="base", enable_diarization=True)
transcription = asyncio.run(transcriber.transcribe(video))
print(f"Detected speakers: {transcription.speakers}")

# Overlay subtitles on video
from videopython.base.text.overlay import TranscriptionOverlay
transcription_overlay = TranscriptionOverlay(font_filename="src/tests/test_data/test_font.ttf")

video = transcription_overlay.apply(video, transcription)
video.save()
```

### AI Video Understanding
```python
import asyncio
from videopython.base.video import Video
from videopython.ai.understanding.video import VideoAnalyzer

video = Video.from_path("<PATH_TO_VIDEO>")

# Comprehensive video analysis
analyzer = VideoAnalyzer(detection_backend="local")  # or "openai", "gemini"
result = asyncio.run(analyzer.analyze(
    video,
    detect_objects=True,  # YOLO (local) or vision API (cloud)
    detect_faces=True,    # OpenCV
    extract_colors=True,
))

for scene in result.scene_descriptions:
    print(f"Scene {scene.start:.1f}s-{scene.end:.1f}s: {scene.detected_entities}")
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

To run the unit tests:
```bash
uv run pytest src/tests/base  # Base tests (no AI, runs in CI)
uv run pytest src/tests/ai    # AI tests (requires ai extras, excluded from CI)
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

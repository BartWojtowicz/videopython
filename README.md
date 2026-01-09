# videopython

A minimal video generation and processing library designed for short-form videos, with focus on simplicity and ease of use for both humans and AI agents.

## Installation

### Install ffmpeg
```bash
# MacOS
brew install ffmpeg
# Ubuntu
sudo apt-get install ffmpeg
```

### Install library
```bash
# With AI features
uv add videopython --extra ai
# or
pip install "videopython[ai]"

# Base only (no AI dependencies)
uv add videopython
# or
pip install videopython
```

## Quick Example

```python
import asyncio
from videopython.base import Video, CutSeconds, Resize, TransformationPipeline, FadeTransition
from videopython.ai import TextToImage, ImageToVideo, AudioToText
from videopython.base.text import TranscriptionOverlay

# Load and transform videos
video1 = Video.from_path("clip1.mp4")
video2 = Video.from_path("clip2.mp4")

pipeline = TransformationPipeline([
    CutSeconds(start=0, end=5),
    Resize(width=1080, height=1920),
])
video1 = pipeline.run(video1)
video2 = pipeline.run(video2)

# Combine with fade transition
fade = FadeTransition(effect_time_seconds=1.0)
video = fade.apply((video1, video2))

# Generate and add AI content
async def add_ai_content(video):
    # Generate an image and animate it
    image = await TextToImage(backend="openai").generate_image("A sunset over mountains")
    intro = await ImageToVideo().generate_video(image=image, prompt="Sunset animation")

    # Transcribe and add subtitles
    transcription = await AudioToText(backend="openai").transcribe(video)
    overlay = TranscriptionOverlay()
    video = overlay.apply(video, transcription)

    return intro + video

video = asyncio.run(add_ai_content(video))
video.save("output.mp4")
```

## Documentation

For more examples and API reference, see the [full documentation](docs/index.md).

## AI Backend Support

Cloud backends require API keys: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ELEVENLABS_API_KEY`, `RUNWAYML_API_KEY`, `LUMAAI_API_KEY`.

| Class | local | openai | gemini | elevenlabs | luma | runway |
|-------|-------|--------|--------|------------|------|--------|
| TextToVideo | CogVideoX1.5-5B | - | - | - | Dream Machine | - |
| ImageToVideo | CogVideoX1.5-5B-I2V | - | - | - | Dream Machine | Gen-4 Turbo |
| VideoUpscaler | RealBasicVSR | - | - | - | - | - |
| TextToSpeech | Bark | TTS | - | Multilingual v2 | - | - |
| TextToMusic | MusicGen | - | - | - | - | - |
| TextToImage | SDXL | DALL-E 3 | - | - | - | - |
| ImageToText | BLIP | GPT-4o | Gemini | - | - | - |
| AudioToText | Whisper | Whisper API | Gemini | - | - | - |
| LLMSummarizer | Ollama | GPT-4o | Gemini | - | - | - |
| ObjectDetector | YOLO | GPT-4o | Gemini | - | - | - |
| TextDetector | EasyOCR | GPT-4o | Gemini | - | - | - |
| FaceDetector | OpenCV | - | - | - | - | - |
| ShotTypeClassifier | - | GPT-4o | Gemini | - | - | - |
| CameraMotionDetector | OpenCV | - | - | - | - | - |

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup instructions and contribution guidelines.

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
from videopython.base import Video, FadeTransition
from videopython.ai import TextToImage, ImageToVideo, AudioToText
from videopython.base.text import TranscriptionOverlay

# Load and transform videos using fluent API
video1 = Video.from_path("clip1.mp4").cut(0, 5).resize(1080, 1920)
video2 = Video.from_path("clip2.mp4").cut(0, 5).resize(1080, 1920)

# Combine with fade transition
video = video1.transition_to(video2, FadeTransition(effect_time_seconds=1.0))

# Generate and add AI content
def add_ai_content(video):
    # Generate an image and animate it
    image = TextToImage(backend="openai").generate_image("A sunset over mountains")
    intro = ImageToVideo().generate_video(image=image, prompt="Sunset animation")

    # Transcribe and add subtitles
    transcription = AudioToText(backend="openai").transcribe(video)
    overlay = TranscriptionOverlay()
    video = overlay.apply(video, transcription)

    return intro + video

video = add_ai_content(video)
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
| TextToSpeech | Bark | TTS | - | Multilingual v2 | - | - |
| TextToMusic | MusicGen | - | - | - | - | - |
| TextToImage | SDXL | DALL-E 3 | - | - | - | - |
| ImageToText | BLIP | GPT-4o | Gemini | - | - | - |
| AudioToText | Whisper | Whisper API | Gemini | - | - | - |
| AudioClassifier | PANNs | - | - | - | - | - |
| LLMSummarizer | Ollama | GPT-4o | Gemini | - | - | - |
| ObjectDetector | YOLO | GPT-4o | Gemini | - | - | - |
| TextDetector | EasyOCR | GPT-4o | Gemini | - | - | - |
| FaceDetector | OpenCV | - | - | - | - | - |
| ShotTypeClassifier | - | GPT-4o | Gemini | - | - | - |
| CameraMotionDetector | OpenCV | - | - | - | - | - |
| MotionAnalyzer | OpenCV | - | - | - | - | - |
| ActionRecognizer | VideoMAE | - | - | - | - | - |
| SemanticSceneDetector | TransNetV2 | - | - | - | - | - |
| VideoDubber | Local Pipeline | - | - | Dubbing API | - | - |
| TextTranslator | Helsinki-NLP | GPT-4o | Gemini | - | - | - |
| AudioSeparator | Demucs | - | - | - | - | - |

## Video Dubbing

Translate and re-voice videos while preserving speaker characteristics:

```python
from videopython.ai.dubbing import VideoDubber
from videopython.base.video import Video

# Load video
video = Video.from_path("video.mp4")

# ElevenLabs cloud dubbing (end-to-end solution)
dubber = VideoDubber(backend="elevenlabs")
dubbed_video = dubber.dub_and_replace(video, target_lang="es")

# Local pipeline with voice cloning
dubber = VideoDubber(backend="local")
result = dubber.dub(
    video,
    target_lang="es",
    preserve_background=True,  # Keep music/sfx
    voice_clone=True,          # Clone original voices with XTTS
)
dubbed_video = video.add_audio(result.dubbed_audio, overlay=False)
dubbed_video.save("dubbed_video.mp4")
```

Supported languages: English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, and 20+ more.

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup instructions and contribution guidelines.

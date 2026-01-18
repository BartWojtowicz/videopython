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

## Features

**Core Video Processing**
- Load, cut, resize, crop, and resample videos
- Fluent API for chaining transformations
- Transitions (fade, blur) and effects (zoom, vignette, ken burns)
- Audio manipulation, mixing, and analysis

**AI Generation**
- Text/image to video (CogVideoX, Luma, Runway)
- Text to speech (Bark, OpenAI, ElevenLabs)
- Text to music (MusicGen)
- Image generation (SDXL, DALL-E)

**AI Understanding**
- Speech transcription and auto-subtitles (Whisper)
- Video analysis with scene detection
- Object, face, and text detection
- Action recognition and motion analysis

**Video Dubbing**
- Translate videos into 30+ languages
- Voice cloning with XTTS or ElevenLabs
- Replace speech with custom text (revoicing)
- Background audio preservation with Demucs

See the [full documentation](https://videopython.com) for examples and API reference.

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

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup instructions and contribution guidelines.

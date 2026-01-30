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

Video processing, AI generation, understanding, dubbing, and object swapping. See [full documentation](https://videopython.com) for examples and API reference.

## AI Backend Support

Cloud backends require API keys: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ELEVENLABS_API_KEY`, `RUNWAYML_API_KEY`, `LUMAAI_API_KEY`, `REPLICATE_API_TOKEN`.

| Class | local | openai | gemini | elevenlabs | luma | runway | replicate |
|-------|-------|--------|--------|------------|------|--------|-----------|
| TextToVideo | CogVideoX1.5-5B | - | - | - | Dream Machine | - | - |
| ImageToVideo | CogVideoX1.5-5B-I2V | - | - | - | Dream Machine | Gen-4 Turbo | - |
| TextToSpeech | Bark | TTS | - | Multilingual v2 | - | - | - |
| TextToMusic | MusicGen | - | - | - | - | - | - |
| TextToImage | SDXL | DALL-E 3 | - | - | - | - | - |
| ImageToText | BLIP | GPT-4o | Gemini | - | - | - | - |
| AudioToText | Whisper | Whisper API | Gemini | - | - | - | - |
| AudioClassifier | AST | - | - | - | - | - | - |
| ObjectDetector | YOLO | GPT-4o | Gemini | - | - | - | - |
| TextDetector | EasyOCR | GPT-4o | Gemini | - | - | - | - |
| FaceDetector | OpenCV | - | - | - | - | - | - |
| ShotTypeClassifier | - | GPT-4o | Gemini | - | - | - | - |
| CameraMotionDetector | OpenCV | - | - | - | - | - | - |
| MotionAnalyzer | OpenCV | - | - | - | - | - | - |
| ActionRecognizer | VideoMAE | - | - | - | - | - | - |
| SemanticSceneDetector | TransNetV2 | - | - | - | - | - | - |
| VideoDubber | Local Pipeline | - | - | Dubbing API | - | - | - |
| TextTranslator | Helsinki-NLP | GPT-4o | Gemini | - | - | - | - |
| AudioSeparator | Demucs | - | - | - | - | - | - |
| ObjectSwapper | SAM2+SDXL | - | - | - | - | - | SAM2+SDXL |

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for setup instructions and contribution guidelines.

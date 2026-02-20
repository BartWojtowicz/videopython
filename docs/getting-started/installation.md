# Installation

## Prerequisites

videopython requires FFmpeg for video processing. Install it first:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

## Install videopython

### Basic Installation

For basic video handling and processing:

```bash
pip install videopython

# Or with uv
uv add videopython
```

### With AI Features

To use AI-powered generation, understanding, dubbing, and object swapping features:

```bash
pip install "videopython[ai]"

# Or with uv
uv add videopython --extra ai
```

!!! note "Hardware Requirements"
    Local AI models (video generation, music generation) require a GPU.
    CUDA (NVIDIA) or MPS (Apple Silicon) is supported. An NVIDIA A40 or better is recommended for video generation.

## API Keys for Cloud Backends

Cloud backends require API keys set as environment variables (only for the providers you use):

| Backend | Environment Variable |
|---------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| ElevenLabs | `ELEVENLABS_API_KEY` |
| Runway | `RUNWAYML_API_KEY` |
| Luma AI | `LUMAAI_API_KEY` |
| Replicate | `REPLICATE_API_TOKEN` |

```bash
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export ELEVENLABS_API_KEY="your-key-here"
export RUNWAYML_API_KEY="your-key-here"
export LUMAAI_API_KEY="your-key-here"
export REPLICATE_API_TOKEN="your-key-here"
```

## Choose Your Backend

Most AI classes default to `backend="local"`. You can switch per task:

| Common Task | Recommended Start | Other Backend Options |
|------------|-------------------|------------------------|
| Text-to-image (`TextToImage`) | `local` | `openai` |
| Text-to-video (`TextToVideo`) | `local` | `luma` |
| Image-to-video (`ImageToVideo`) | `local` | `luma`, `runway` |
| Text-to-speech (`TextToSpeech`) | `openai` | `local`, `elevenlabs` |
| Speech-to-text subtitles (`AudioToText`) | `openai` | `local`, `gemini` |
| Video dubbing (`VideoDubber`) | `local` | `elevenlabs` |
| Object swapping (`ObjectSwapper`) | `local` | `replicate` |

Quick rules:

- Use `local` for offline workflows and no API costs.
- Use cloud backends when you need managed inference or do not have suitable local hardware.
- Choose backend per class via the `backend` argument.

```python
from videopython.ai import TextToSpeech

tts = TextToSpeech(backend="elevenlabs")
```

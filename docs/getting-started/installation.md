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

To use AI-powered generation and understanding features:

```bash
pip install "videopython[ai]"

# Or with uv
uv add videopython --extra ai
```

!!! note "Hardware Requirements"
    Local AI models (video generation, music generation) require a CUDA-capable GPU.
    An NVIDIA A40 or better is recommended for video generation.

## API Keys for Cloud Backends

Cloud backends require API keys set as environment variables:

| Backend | Environment Variable |
|---------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| ElevenLabs | `ELEVENLABS_API_KEY` |

```bash
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export ELEVENLABS_API_KEY="your-key-here"
```


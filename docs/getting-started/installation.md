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

## Local-Only AI Runtime

`videopython.ai` runs locally and does not use cloud backend/API key configuration.

Practical setup notes:

- First AI run may download model weights.
- Prefer GPU (`cuda` or `mps`) for generation-heavy workflows.
- Use the `device` argument where supported to force placement.

```python
from videopython.ai import TextToSpeech, ImageToVideo

tts = TextToSpeech(device="cuda")
video_gen = ImageToVideo()
```

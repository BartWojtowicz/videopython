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

To use every AI-powered feature (generation, understanding, dubbing):

```bash
pip install "videopython[ai]"

# Or with uv
uv add videopython --extra ai
```

`[ai]` is the single extra and installs every AI capability — transcription,
detection/scene/VLM understanding, source separation, translation, TTS, media
generation, dubbing, and the LLM auto-editing planner. The heavy ML dependencies
still load lazily at first use (no top-level imports under `ai/`), so importing
`videopython` stays light even with `[ai]` installed.

!!! note "Dubbing TTS"
    The dubbing pipeline synthesizes speech with a local Chatterbox
    `TextToSpeech` by default. To run synthesis out of process instead, inject
    your own `SpeechBackend` into `VideoDubber` (e.g. a remote synthesizer).

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

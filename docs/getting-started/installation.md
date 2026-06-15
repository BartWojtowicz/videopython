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

`[ai]` is a convenience aggregate that installs everything. If you only need one
capability, install the matching granular extra instead — smaller, conflict-free,
and (except `[tts]`) free of chatterbox's strict version pins:

| Extra | Installs | Powers |
|-------|----------|--------|
| `[asr]` | Whisper, pyannote, silero-vad | `AudioToText` transcription + diarization |
| `[vision]` | YOLO, TransNetV2, transformers, qwen-vl-utils, imagehash | `ObjectDetector`, `FaceTracker`, `SceneVLM`, `SemanticSceneDetector`, `AudioClassifier`, `VideoAnalyzer` |
| `[separation]` | Demucs | `AudioSeparator` |
| `[translation]` | MarianMT (transformers/sentencepiece), Qwen3 (llama-cpp) | `MarianTranslator`, `Qwen3Translator` |
| `[tts]` | Chatterbox Multilingual | `TextToSpeech` (local voice-cloning synthesis) |
| `[generation]` | SDXL, CogVideoX (diffusers), MusicGen | `TextToImage`, `TextToVideo`, `ImageToVideo`, `TextToMusic` |
| `[dub]` | `asr` + `separation` + `translation` + pyloudnorm | dubbing pipeline (see note below) |

```bash
pip install "videopython[asr]"          # just transcription
pip install "videopython[dub,tts]"      # dubbing with local TTS
```

!!! note "Dubbing and TTS"
    `[dub]` deliberately excludes `chatterbox-tts` so a dubbing image
    co-resolves without chatterbox's strict torch/diffusers pins. To dub with
    the default **local** voice synthesis, also install `[tts]`
    (`pip install "videopython[dub,tts]"`). A bare `[dub]` install that reaches
    local synthesis raises a clear `ImportError` pointing at `[tts]`.
    Alternatively, inject your own `SpeechBackend` into `VideoDubber` to run
    synthesis out of process — then `[dub]` alone is enough.

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

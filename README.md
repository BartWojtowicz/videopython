# videopython

[![PyPI](https://img.shields.io/pypi/v/videopython)](https://pypi.org/project/videopython/)
[![Python](https://img.shields.io/pypi/pyversions/videopython)](https://pypi.org/project/videopython/)
[![License](https://img.shields.io/github/license/BartWojtowicz/videopython)](LICENSE)

Minimal, LLM-friendly Python library for programmatic video editing, processing, and AI video workflows.

Full documentation: [videopython.com](https://videopython.com)

> **Disclaimer:** This project started as a hand-written hobby project, but most of the code is now produced by LLM agents. Humans still drive direction, approve changes, and own design decisions.

## Installation

```bash
# Install FFmpeg first (macOS: brew install ffmpeg | Debian: apt-get install ffmpeg)
pip install videopython          # core video/audio editing
pip install "videopython[ai]"    # + local AI features (GPU recommended)
```

Python `>=3.11, <3.14`. AI features run locally — no cloud API keys required, but model weights are downloaded on first use.

## Quick Start

### JSON editing plans

A `VideoEdit` is a multi-segment plan, defined as a dict (or JSON), validated and executed against the source files:

```python
from videopython.editing import VideoEdit

edit = VideoEdit.from_dict({
    "segments": [{
        "source": "raw.mp4",
        "start": 10.0,
        "end": 20.0,
        "operations": [
            {"op": "resize", "width": 1080, "height": 1920},
            {"op": "color_adjust", "saturation": 1.15, "contrast": 1.05},
            {"op": "fade", "mode": "in", "duration": 0.5},
        ],
    }],
})
edit.validate()                  # dry-run via metadata, no frames loaded
edit.run_to_file("output.mp4")   # streams ffmpeg decode → effects → encode
```

`run_to_file()` streams ffmpeg decode → per-frame effects → encode, so memory stays bounded even for hour-long sources. Use `edit.run()` to get a `Video` back in memory instead.

### AI generation

```python
from videopython.ai import TextToImage, ImageToVideo, TextToSpeech

image = TextToImage().generate_image("A cinematic mountain sunrise")
video = ImageToVideo().generate_video(image=image)
audio = TextToSpeech().generate_audio("Welcome to videopython.")
video.add_audio(audio).save("ai_video.mp4")
```

## LLM & AI Agent Integration

Every operation is a Pydantic model whose fields ARE the JSON wire format. `VideoEdit.json_schema()` returns a JSON Schema with a discriminated union over every LLM-exposed `Operation` (server-only ops like `image_overlay` are excluded by default) — pass it straight to Anthropic tool use, OpenAI function calling, or any structured-output API. Pass `strict=True` for a provider strict-mode grammar that prevents simple bound violations at decode time.

The plan parses permissively (shape only) and owns numeric bounds at validation, so a refine loop converges fast: `edit.check(meta)` collects **every** structured `PlanError` in one pass, `edit.repair(meta)` auto-clamps the mechanical violations (window/timestamp overruns, negatives) with a reported changelog, and `edit.normalize_dimensions(meta, target)` makes heterogeneous segments concat-compatible by construction. `edit.validate()` still raises a typed `PlanValidationError` (a `ValueError` with structured `.errors`) for the single-error path.

See the [LLM Integration Guide](https://videopython.com/guides/llm-integration/) for end-to-end examples, the collect/repair/normalize refine loop, and operation discovery patterns.

## Features

- **`videopython.base`** — `Video`, `VideoMetadata`, `FrameIterator`, `ImageText`, `Transcription`, and shared result types (`BoundingBox`, `FaceTrack`, `SceneBoundary`, ...). No AI dependencies.
- **`videopython.audio`** — `Audio` with overlay, concat, normalize, time-stretch, silence detection, segment classification.
- **`videopython.editing`** — `Operation`/`Effect` foundation, `VideoEdit` plan runner with JSON Schema + streaming execution. Transforms (cut, resize, crop, fps, speed, reverse, freeze, silence removal) and effects (blur, zoom, color grading, vignette, Ken Burns, fade, overlays, animated subtitles).
- **`videopython.ai`** *(install with `[ai]`)* — generation (`TextToVideo`, `ImageToVideo`, `TextToImage`, `TextToSpeech`, `TextToMusic`), understanding (`AudioToText`, `AudioClassifier`, `SceneVLM`, `FaceTracker`, `ObjectDetector`, `SemanticSceneDetector`), the `FaceTrackingCrop` transform, the `ObjectDetectionOverlay` effect (per-frame bounding boxes + labels), and the full-pipeline `VideoAnalyzer`.
- **`videopython.ai.dubbing`** — `VideoDubber` for voice-cloned revoicing with timing sync.

## Examples

- [Social Media Clip](https://videopython.com/examples/social-clip/)
- [AI-Generated Video](https://videopython.com/examples/ai-video/)
- [Auto-Subtitles](https://videopython.com/examples/auto-subtitles/)
- [Processing Large Videos](https://videopython.com/examples/large-videos/)

## Development

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for local setup, testing, and contribution workflow.

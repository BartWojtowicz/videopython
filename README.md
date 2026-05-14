# videopython

[![PyPI](https://img.shields.io/pypi/v/videopython)](https://pypi.org/project/videopython/)
[![Python](https://img.shields.io/pypi/pyversions/videopython)](https://pypi.org/project/videopython/)
[![License](https://img.shields.io/github/license/BartWojtowicz/videopython)](LICENSE)

Minimal, LLM-friendly Python library for programmatic video editing, processing, and AI video workflows.

Full documentation: [videopython.com](https://videopython.com)

> **Disclaimer:** This project started as a hand-written hobby project, but most of the code is now produced by LLM agents. Humans still drive direction, approve changes, and own design decisions.

## Installation

### 1. Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

### 2. Install videopython

```bash
pip install videopython          # core video/audio editing
pip install "videopython[ai]"    # + local AI features (GPU recommended)
```

Python `>=3.10, <3.14`. AI features run locally - no cloud API keys required, but model weights are downloaded on first use.

## Quick Start

### Imperative editing

Every editing primitive is an `Operation` subclass — a Pydantic model
whose fields ARE the JSON wire format. Apply one to a `Video`:

```python
from videopython.base import Video, CutSeconds, Resize, Fade

video = Video.from_path("raw.mp4")
video = CutSeconds(start=10, end=25).apply(video)
video = Resize(width=1080, height=1920).apply(video)
video = Fade(mode="in", duration=0.5).apply(video)
video.save("output.mp4")
```

Concatenate clips with `+` (must share fps + dimensions):

```python
combined = video_a + video_b
```

### JSON editing plans

Define multi-segment edits as JSON — the format LLM-driven workflows
generate against. `VideoEdit.json_schema()` returns the schema:

```python
from videopython.editing import VideoEdit

plan = {
    "segments": [{
        "source": "raw.mp4",
        "start": 10.0,
        "end": 20.0,
        "operations": [
            {"op": "resize", "width": 1080, "height": 1920},
            {"op": "color_adjust", "saturation": 1.15, "contrast": 1.05},
            {"op": "fade", "mode": "in", "duration": 0.5,
             "window": {"stop": 0.5}},
        ],
    }],
}

edit = VideoEdit.from_dict(plan)
edit.validate()                  # dry-run via metadata, no frames loaded
edit.run_to_file("output.mp4")   # stream to disk, ~constant memory
```

`run_to_file()` pipes ffmpeg decode → per-frame effects → ffmpeg encode,
so memory stays bounded even for hour-long sources. Use `edit.run()`
instead if you want the result back in memory as a `Video`.

### AI generation

```python
from videopython.ai import TextToImage, ImageToVideo, TextToSpeech
from videopython.base import Resize

image = TextToImage().generate_image("A cinematic mountain sunrise")
video = ImageToVideo().generate_video(image=image)
audio = TextToSpeech().generate_audio("Welcome to videopython.")

video = Resize(width=1080, height=1920).apply(video)
video.add_audio(audio).save("ai_video.mp4")
```

## LLM & AI Agent Integration

The library is built for LLM-driven editing. Two surfaces matter:

**1. Plan schema for tool / structured-output calls.**
`VideoEdit.json_schema()` returns a JSON Schema covering segments,
`post_operations`, and a discriminated union over every registered
`Operation`. Drop it into any LLM API:

```python
from videopython.editing import VideoEdit

schema = VideoEdit.json_schema()
# Anthropic: tools=[{"name": "edit", "input_schema": schema}]
# OpenAI:    tools=[{"type": "function",
#                    "function": {"name": "edit", "parameters": schema}}]
```

Validate the LLM's output without touching the filesystem, then run it:

```python
edit = VideoEdit.from_dict(plan)
edit.validate()                  # catches bad ops, time ranges, fps mismatches
edit.run_to_file("output.mp4")
```

**2. Operation discovery for agent loops.**
Every registered op exposes its own Pydantic schema, so an agent can
introspect what's available without hardcoded lists:

```python
from videopython.base import Operation, OpCategory

for op_id, cls in Operation.registry().items():
    print(f"{op_id}: {(cls.__doc__ or '').splitlines()[0]}")

schema = Operation.get("color_adjust").model_json_schema()  # per-op schema
```

Field constraints (`minimum`, `maximum`, `enum`, `exclusiveMinimum`,
nullability) flow through to the schema, so LLMs that support
constrained generation produce valid parameters on the first try.

For ops that need side-channel data (e.g. `silence_removal` and
`add_subtitles` need a `Transcription`), pass it via `context`:

```python
edit.run(context={"transcription": my_transcription})
```

Docs: [Editing Plans](https://videopython.com/api/editing/) | [Operations](https://videopython.com/api/operations/) | [LLM Integration Guide](https://videopython.com/guides/llm-integration/)

## Features

### `videopython.base` - core editing (no AI dependencies)

| Area | Highlights |
|---|---|
| **Video I/O** | `Video`, `VideoMetadata`, `FrameIterator` - load, save, inspect |
| **Operation foundation** | `Operation`, `Effect`, `TimeRange`, `OpCategory` - Pydantic base + auto-registry + discriminated-union schema |
| **Editing plans** | `VideoEdit`, `SegmentConfig` - JSON/LLM-friendly multi-segment plans with JSON Schema generation, dry-run validation, and streaming `run_to_file` |
| **Transforms** | Cut (time/frame), resize, crop, FPS resampling, speed change, reverse, freeze frame, silence removal |
| **Effects** | Blur, zoom, color grading, vignette, Ken Burns, image overlay, fade, text overlay, volume adjust |
| **Audio** | Load/save, overlay, concat, normalize, time-stretch, silence detection, segment classification |
| **Text** | Transcription data classes, `TranscriptionOverlay` for subtitle rendering |
| **Scene detection** | Histogram-based scene boundaries (`detect`, `detect_streaming`, `detect_parallel`) |

API docs: [Core](https://videopython.com/api/index/) | [Video](https://videopython.com/api/core/video/) | [Audio](https://videopython.com/api/core/audio/) | [Editing Plans](https://videopython.com/api/editing/) | [Operations](https://videopython.com/api/operations/) | [Transforms](https://videopython.com/api/transforms/) | [Effects](https://videopython.com/api/effects/) | [Text](https://videopython.com/api/text/)

### `videopython.ai` - local AI features (install with `[ai]`)

| Area | Highlights |
|---|---|
| **Generation** | `TextToVideo`, `ImageToVideo`, `TextToImage`, `TextToSpeech`, `TextToMusic` |
| **Understanding** | `AudioToText` (transcription), `AudioClassifier`, `SceneVLM` (structured visual scene description), `FaceTracker` (per-shot face tracks) |
| **Scene detection** | `SemanticSceneDetector` (neural scene boundaries) |
| **Video analysis** | `VideoAnalyzer` - full-pipeline analysis combining multiple AI capabilities |
| **Transforms** | `FaceTrackingCrop` |
| **Dubbing** | `VideoDubber` - voice cloning and revoicing with timing sync |

API docs: [Generation](https://videopython.com/api/ai/generation/) | [Understanding](https://videopython.com/api/ai/understanding/) | [Transforms](https://videopython.com/api/ai/transforms/) | [Dubbing](https://videopython.com/api/ai/dubbing/)

## Examples

- [Social Media Clip](https://videopython.com/examples/social-clip/)
- [AI-Generated Video](https://videopython.com/examples/ai-video/)
- [Auto-Subtitles](https://videopython.com/examples/auto-subtitles/)
- [Processing Large Videos](https://videopython.com/examples/large-videos/)

## Development

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for local setup, testing, and contribution workflow.

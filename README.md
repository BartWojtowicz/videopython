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
pip install videopython              # core video/audio editing
pip install "videopython[ai]"        # + ALL local AI features (GPU recommended)
pip install "videopython[ai,mcp]"    # + MCP server for agent-driven editing
```

Python `>=3.11, <3.14`. AI features run locally — no cloud API keys required, but model weights are downloaded on first use. LLM-driven editing and scene captioning use a local [Ollama](https://ollama.com) server (`ollama pull gemma3:27b`).

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

`run_to_file()` streams ffmpeg decode → per-frame effects → encode, so memory stays bounded even for hour-long sources. If you need the frames back in memory, load the rendered file: `Video.from_path(str(edit.run_to_file("output.mp4")))`.

### Automatic editing (local LLM)

Give `AutoEditor` your clips and a brief; a local Ollama vision model selects and orders the shots, and you get back a runnable `VideoEdit`:

```python
from videopython.ai import AutoEditor, OllamaVisionLLM

editor = AutoEditor(planner=OllamaVisionLLM(model="gemma3:27b"))  # ollama pull gemma3:27b
edit = editor.edit(
    ["clip_a.mp4", "clip_b.mp4", "clip_c.mp4"],
    brief="A punchy 15-second teaser; lead with the most dynamic shot.",
)
edit.run_to_file("teaser.mp4")
```

The model picks scenes **by id** from a catalog built from scene detection + captions, so its temporal imprecision never reaches the render. See the [Automatic Editing Guide](https://videopython.com/guides/auto-editing/).

### AI generation

```python
from videopython.ai import TextToImage, ImageToVideo, TextToSpeech

image = TextToImage().generate_image("A cinematic mountain sunrise")
video = ImageToVideo().generate_video(image=image)
audio = TextToSpeech().generate_audio("Welcome to videopython.")
video.add_audio(audio).save("ai_video.mp4")
```

## LLM & AI Agent Integration

Putting an LLM in the loop works three ways:

1. **Bring your own LLM** — videopython gives your model the JSON Schema and a structured refine loop; your model authors the plans (details below).
2. **`AutoEditor`** — a local Ollama vision model is the planner (see [Automatic editing](#automatic-editing-local-llm) above).
3. **MCP server** — `videopython-mcp` exposes the pipeline as [Model Context Protocol](https://modelcontextprotocol.io) tools, so an agent like Claude drives editing with its own model. Install `[ai,mcp]`, run `videopython-mcp`, and point your MCP client at it. See the [MCP Server Guide](https://videopython.com/guides/mcp/).

For mode 1: every operation is a Pydantic model whose fields ARE the JSON wire format. `VideoEdit.json_schema()` returns a JSON Schema with a discriminated union over every LLM-exposed `Operation` (server-only ops like `image_overlay` are excluded by default) — pass it straight to Anthropic tool use, OpenAI function calling, or any structured-output API. Pass `strict=True` for a provider strict-mode grammar that prevents simple bound violations at decode time.

The plan parses permissively (shape only) and owns numeric bounds at validation, so a refine loop converges fast: `edit.check(meta)` collects **every** structured `PlanError` in one pass, `edit.repair(meta)` auto-clamps the mechanical violations (window/timestamp overruns, negatives) with a reported changelog, and `edit.normalize_dimensions(meta, target)` makes heterogeneous segments concat-compatible by construction. `edit.validate()` still raises a typed `PlanValidationError` (a `ValueError` with structured `.errors`) for the single-error path.

See the [LLM Integration Guide](https://videopython.com/guides/llm-integration/) for end-to-end examples, the collect/repair/normalize refine loop, and operation discovery patterns.

## Features

- **`videopython.base`** — `Video`, `VideoMetadata`, `FrameIterator`, `Transcription`, and shared result types (`BoundingBox`, `FaceTrack`, `SceneBoundary`, ...). No AI dependencies.
- **`videopython.audio`** — `Audio` with overlay, concat, normalize, time-stretch, silence detection, segment classification.
- **`videopython.editing`** — `Operation`/`Effect` foundation, `VideoEdit` plan runner with JSON Schema + streaming execution. Transforms (resize, crop, fps, speed, freeze, silence removal; cutting is the segment's own start/end) and effects (blur, zoom, color grading, vignette, Ken Burns, fade, overlays, animated subtitles).
- **`videopython.ai`** *(install with `[ai]`)* — generation (`TextToVideo`, `ImageToVideo`, `TextToImage`, `TextToSpeech`, `TextToMusic`), understanding (`AudioToText`, `AudioClassifier`, `SceneVLM`, `FaceTracker`, `ObjectDetector`, `SemanticSceneDetector`), the `FaceTrackingCrop` transform, the `ObjectDetectionOverlay` effect (per-frame bounding boxes + labels), and the full-pipeline `VideoAnalyzer`. Scene captioning and dub translation run on a local [Ollama](https://ollama.com) model.
- **`videopython.ai.auto_edit`** — `AutoEditor` + `OllamaVisionLLM`: plan and render an edit from sources + a one-line brief, with a local LLM selecting scenes by id from an auto-built catalog.
- **`videopython.ai.dubbing`** — `VideoDubber` for voice-cloned revoicing with timing sync.
- **`videopython.mcp`** *(install with `[mcp]`)* — `videopython-mcp`, an MCP stdio server exposing the auto-edit pipeline (analyze → catalog → validate/repair/run) so an agent drives editing.

## Examples

- [Social Media Clip](https://videopython.com/examples/social-clip/)
- [AI-Generated Video](https://videopython.com/examples/ai-video/)
- [Auto-Subtitles](https://videopython.com/examples/auto-subtitles/)
- [Processing Large Videos](https://videopython.com/examples/large-videos/)

## Development

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for local setup, testing, and contribution workflow.

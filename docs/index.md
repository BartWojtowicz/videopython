# videopython

Minimal, LLM-friendly Python library for programmatic video editing, processing, and AI video workflows.

## Quick Example

```python
from videopython import Video
from videopython.base import FadeTransition

intro = Video.from_path("intro.mp4").resize(1080, 1920)
clip = Video.from_path("raw.mp4").cut(10, 25).resize(1080, 1920).resample_fps(30)
final = intro.transition_to(clip, FadeTransition(effect_time_seconds=0.5))
final = final.add_audio_from_file("music.mp3")
final.save("output.mp4")
```

## What You Can Do

### Core editing (`videopython.base`)

- **Edit videos** - Cut, resize, crop, change speed, reverse, freeze frames, picture-in-picture
- **Combine clips** - Concatenate, split, and join with fade/blur/instant transitions
- **Apply effects** - Blur, zoom, color grading, vignette, Ken Burns, text overlay, image overlay, fade
- **Process audio** - Load, overlay, concat, normalize, time-stretch, silence detection
- **Add subtitles** - Transcription data classes and word-level subtitle rendering
- **Detect scenes** - Histogram-based scene boundary detection (single-pass or parallel)
- **LLM-driven editing** - JSON editing plans with full JSON Schema generation, dry-run validation, and an operation registry with rich constraints. See the [LLM Integration Guide](guides/llm-integration.md)

### AI features (`videopython[ai]`)

- **Generate content** - Create images, videos, speech, and music from text prompts
- **Understand video** - Transcribe audio, describe scenes, classify audio, recognize actions
- **Analyze video** - Full-pipeline analysis combining audio, visual, and temporal understanding
- **Dub and revoice** - Translate speech to 50+ languages with voice cloning
- **Swap objects** - Replace or remove objects using AI segmentation and inpainting
- **Track faces** - Face tracking crops and split-screen composites

## Installation

```bash
pip install videopython          # core editing
pip install "videopython[ai]"    # + local AI features (GPU recommended)
```

Python `>=3.10, <3.13`. AI features run locally - no cloud API keys required.

See the [Installation Guide](getting-started/installation.md) for FFmpeg setup and details.

# videopython

[![PyPI](https://img.shields.io/pypi/v/videopython)](https://pypi.org/project/videopython/)
[![Python](https://img.shields.io/pypi/pyversions/videopython)](https://pypi.org/project/videopython/)
[![License](https://img.shields.io/github/license/BartWojtowicz/videopython)](LICENSE)

Minimal, LLM-friendly Python library for programmatic video editing, processing, and AI video workflows.

Full documentation: [videopython.com](https://videopython.com)

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

Python `>=3.10, <3.13`. AI features run locally - no cloud API keys required, but model weights are downloaded on first use.

## Quick Start

### Video editing

```python
from videopython import Video
from videopython.base import FadeTransition

intro = Video.from_path("intro.mp4").resize(1080, 1920)
clip = Video.from_path("raw.mp4").cut(10, 25).resize(1080, 1920).resample_fps(30)
final = intro.transition_to(clip, FadeTransition(effect_time_seconds=0.5))
final = final.add_audio_from_file("music.mp3")
final.save("output.mp4")
```

### JSON editing plans

Define multi-segment edits as JSON - useful for LLM-driven workflows. `VideoEdit.json_schema()` returns a schema for plan generation/validation.

```python
from videopython.editing import VideoEdit

plan = {
    "segments": [{
        "source": "raw.mp4",
        "start": 10.0,
        "end": 20.0,
        "transforms": [
            {"op": "resize", "args": {"height": 1280}},
            {"op": "speed_change", "args": {"speed": 1.25}},
        ],
    }],
    "post_effects": [
        {"op": "fade", "args": {"mode": "in", "duration": 0.5}, "apply": {"start": 0.0, "stop": 0.5}},
    ],
}

edit = VideoEdit.from_dict(plan)
edit.validate()   # dry-run via metadata (no frame loading)
final = edit.run()
final.save("output.mp4")
```

### Multicam podcast editing

Switch between synchronized camera angles with transitions:

```python
from videopython.editing import MultiCamEdit, CutPoint
from videopython.base import FadeTransition

edit = MultiCamEdit(
    sources={"wide": "cam1.mp4", "closeup": "cam2.mp4"},
    audio_source="podcast_audio.aac",
    cuts=[
        CutPoint(time=0.0, camera="wide"),
        CutPoint(time=15.0, camera="closeup", transition=FadeTransition(0.5)),
        CutPoint(time=45.0, camera="wide", transition=FadeTransition(0.5)),
    ],
)
edit.run().save("podcast.mp4")
```

### AI generation

```python
from videopython.ai import TextToImage, ImageToVideo, TextToSpeech

image = TextToImage().generate_image("A cinematic mountain sunrise")
video = ImageToVideo().generate_video(image=image, fps=24).resize(1080, 1920)
audio = TextToSpeech().generate_audio("Welcome to videopython.")
video.add_audio(audio).save("ai_video.mp4")
```

## LLM & AI Agent Integration

videopython is designed to be controlled by LLMs. Every video operation exposes a machine-readable spec with descriptions, parameter types, and value constraints - all available as JSON Schema at runtime.

**Schema generation** - `VideoEdit.json_schema()` returns a complete JSON Schema describing valid edit plans. Pass it directly as a tool schema or structured-output format to any LLM API:

```python
from videopython.editing import VideoEdit

schema = VideoEdit.json_schema()
# Pass `schema` to your LLM as a function/tool definition or response format.
# The LLM generates a plan dict, then:

edit = VideoEdit.from_dict(plan)
edit.validate()   # dry-run: checks sources, time ranges, params - no frames loaded
final = edit.run()
final.save("output.mp4")
```

**Operation discovery** - the registry lets an LLM (or your code) inspect all available operations, their parameters, and constraints:

```python
from videopython.base import get_operation_specs, get_specs_by_category, OperationCategory

all_ops = get_operation_specs()                                    # all registered operations
transforms = get_specs_by_category(OperationCategory.TRANSFORMATION)  # just transforms

spec = all_ops["color_adjust"]
print(spec.description)       # LLM-friendly docstring
print(spec.to_json_schema())  # {"brightness": {"type": "number", "minimum": -1, "maximum": 1}, ...}
```

Every operation has LLM-optimized descriptions and rich constraints (`minimum`, `maximum`, `enum`, `exclusive_minimum`, etc.) so models generate valid parameters on the first try.

Docs: [Editing Plans](https://videopython.com/api/editing/) | [Operation Registry](https://videopython.com/api/registry/)

## Features

### `videopython.base` - core editing (no AI dependencies)

| Area | Highlights |
|---|---|
| **Video I/O** | `Video`, `VideoMetadata`, `FrameIterator` - load, save, inspect |
| **Editing plans** | `VideoEdit`, `SegmentConfig` - JSON/LLM-friendly multi-segment plans with full JSON Schema generation, dry-run validation, and operation registry |
| **Multicam editing** | `MultiCamEdit`, `CutPoint` - switch between synchronized camera angles with transitions, replace audio with external track |
| **Transforms** | Cut (time/frame), resize, crop, FPS resampling, speed change, picture-in-picture, reverse, freeze frame, silence removal |
| **Transitions** | `FadeTransition`, `BlurTransition`, `InstantTransition` |
| **Effects** | Blur, zoom, color grading, vignette, Ken Burns, image overlay, fade, text overlay, volume adjust |
| **Audio** | Load/save, overlay, concat, normalize, time-stretch, silence detection, segment classification |
| **Text** | Transcription data classes, `TranscriptionOverlay` for subtitle rendering |
| **Scene detection** | Histogram-based scene boundaries (`detect`, `detect_streaming`, `detect_parallel`) |

API docs: [Core](https://videopython.com/api/index/) | [Video](https://videopython.com/api/core/video/) | [Audio](https://videopython.com/api/core/audio/) | [Editing Plans](https://videopython.com/api/editing/) | [Transforms](https://videopython.com/api/transforms/) | [Transitions](https://videopython.com/api/transitions/) | [Effects](https://videopython.com/api/effects/) | [Text](https://videopython.com/api/text/)

### `videopython.ai` - local AI features (install with `[ai]`)

| Area | Highlights |
|---|---|
| **Generation** | `TextToVideo`, `ImageToVideo`, `TextToImage`, `TextToSpeech`, `TextToMusic` |
| **Understanding** | `AudioToText` (transcription), `AudioClassifier`, `SceneVLM` (visual scene description), `ActionRecognizer` |
| **Scene detection** | `SemanticSceneDetector` (neural scene boundaries) |
| **Video analysis** | `VideoAnalyzer` - full-pipeline analysis combining multiple AI capabilities |
| **Transforms** | `FaceTracker`, `FaceTrackingCrop`, `SplitScreenComposite` |
| **Dubbing** | `VideoDubber` - voice cloning and revoicing with timing sync |
| **Object swapping** | `ObjectSwapper` - detect, segment, and inpaint objects in video |

API docs: [Generation](https://videopython.com/api/ai/generation/) | [Understanding](https://videopython.com/api/ai/understanding/) | [Transforms](https://videopython.com/api/ai/transforms/) | [Dubbing](https://videopython.com/api/ai/dubbing/) | [Object Swapping](https://videopython.com/api/ai/swapping/)

## Examples

- [Social Media Clip](https://videopython.com/examples/social-clip/)
- [AI-Generated Video](https://videopython.com/examples/ai-video/)
- [Auto-Subtitles](https://videopython.com/examples/auto-subtitles/)
- [Processing Large Videos](https://videopython.com/examples/large-videos/)

## Development

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for local setup, testing, and contribution workflow.

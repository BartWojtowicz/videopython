# videopython

Minimal Python library for video editing, processing, and AI video workflows.
Built primarily for practical editing workflows, with optional AI capabilities layered on top.

Full documentation lives at [videopython.com](https://videopython.com) (guides, examples, and complete API reference).  
Use this README for quick setup and a feature overview.

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
# Core video/audio features only
pip install videopython
# or
uv add videopython

# Include AI features
pip install "videopython[ai]"
# or
uv add videopython --extra ai
```

Python support: `>=3.10, <3.13`.

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

### JSON editing plans (`VideoEdit`)

```python
from videopython.base import VideoEdit

plan = {
    "segments": [
        {
            "source": "raw.mp4",
            "start": 10.0,
            "end": 20.0,
            "transforms": [{"op": "resize", "args": {"height": 1280}}, {"op": "speed_change", "args": {"speed": 1.25}}],
        }
    ],
    "post_effects": [
        {"op": "blur_effect", "args": {"mode": "constant", "iterations": 1}, "apply": {"start": 0.0, "stop": 1.0}}
    ],
}

edit = VideoEdit.from_dict(plan)
edit.validate()  # dry run via VideoMetadata (no frame loading)
final = edit.run()
final.save("output.mp4")
```

Use `post_transforms` for transforms and `post_effects` for effects. `VideoEdit.json_schema()` returns a parser-aligned JSON Schema for plan generation/validation.

### AI generation

```python
from videopython.ai import TextToImage, ImageToVideo, TextToSpeech

image = TextToImage(backend="openai").generate_image("A cinematic mountain sunrise")
video = ImageToVideo(backend="local").generate_video(image=image, fps=24).resize(1080, 1920)
audio = TextToSpeech(backend="openai").generate_audio("Welcome to videopython.")
video.add_audio(audio).save("ai_video.mp4")
```

## Functionality Overview

### `videopython.base` (no AI dependencies)

- Video I/O and metadata: `Video`, `VideoMetadata`, `FrameIterator`
- Editing plans: `VideoEdit`, `SegmentConfig` (JSON/LLM-friendly multi-segment plans with schema generation)
- Transformations: cut by time/frame, resize, crop, FPS resampling, speed change, picture-in-picture
- Clip composition: concatenate, split, transitions (`FadeTransition`, `BlurTransition`, `InstantTransition`)
- Visual effects: blur, zoom, color grading, vignette, Ken Burns, image overlays
- Audio pipeline: load/save audio, overlay/concat, normalize, time-stretch, silence detection, segment classification
- Text/subtitles: transcription data classes and `TranscriptionOverlay`
- Scene detection: histogram-based scene boundaries (`detect`, `detect_streaming`, `detect_parallel`)

Docs:
- [Core API](https://videopython.com/api/index/)
- [Video](https://videopython.com/api/core/video/)
- [Audio](https://videopython.com/api/core/audio/)
- [Editing Plans (`VideoEdit`)](https://videopython.com/api/editing/)
- [Transforms](https://videopython.com/api/transforms/)
- [Transitions](https://videopython.com/api/transitions/)
- [Effects](https://videopython.com/api/effects/)
- [Text & Transcription](https://videopython.com/api/text/)

### `videopython.ai` (install with `[ai]`)

- Generation: `TextToVideo`, `ImageToVideo`, `TextToImage`, `TextToSpeech`, `TextToMusic`
- Understanding:
  - Transcription and captioning: `AudioToText`, `ImageToText`
  - Detection/classification: `ObjectDetector`, `FaceDetector`, `TextDetector`, `ShotTypeClassifier`
  - Motion/action/scene understanding: `CameraMotionDetector`, `MotionAnalyzer`, `ActionRecognizer`, `SemanticSceneDetector`
  - Multi-signal frame analysis: `CombinedFrameAnalyzer`
- AI transforms: `FaceTracker`, `FaceTrackingCrop`, `SplitScreenComposite`
- Dubbing/revoicing: `videopython.ai.dubbing.VideoDubber`
- Object swapping/inpainting: `ObjectSwapper`

Docs:
- [AI Generation](https://videopython.com/api/ai/generation/)
- [AI Understanding](https://videopython.com/api/ai/understanding/)
- [AI Transforms](https://videopython.com/api/ai/transforms/)
- [AI Dubbing](https://videopython.com/api/ai/dubbing/)
- [AI Object Swapping](https://videopython.com/api/ai/swapping/)

## Backends and API Keys

Cloud-enabled features use these environment variables:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ELEVENLABS_API_KEY`
- `RUNWAYML_API_KEY`
- `LUMAAI_API_KEY`
- `REPLICATE_API_TOKEN`

Example:

```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

Notes:
- Local generation models can require substantial GPU resources.
- Backend/model details by class are documented at [videopython.com](https://videopython.com).

## Examples

- [Social Media Clip](https://videopython.com/examples/social-clip/)
- [AI-Generated Video](https://videopython.com/examples/ai-video/)
- [Auto-Subtitles](https://videopython.com/examples/auto-subtitles/)
- [Processing Large Videos](https://videopython.com/examples/large-videos/)

## Development

See [`DEVELOPMENT.md`](DEVELOPMENT.md) for local setup, testing, and contribution workflow.

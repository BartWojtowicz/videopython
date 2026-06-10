# API Reference

videopython is organized into four top-level subpackages. Everything outside `videopython.ai` is installable with the default `pip install videopython` — no ML dependencies required.

## `videopython.base`

Data containers and I/O primitives:

- [**Video**](core/video.md) - `Video`, `VideoMetadata`, `FrameIterator` - loading, saving, inspecting
- [**Text & Transcription**](text.md) - `Transcription` data classes and the `add_subtitles` overlay

## `videopython.audio`

- [**Audio**](core/audio.md) - `Audio` data container; load/save, overlay, concat, normalize, time-stretch, silence detection, segment classification

## `videopython.editing`

Editing primitives and the plan runner:

- [**Editing Plans (`VideoEdit`)**](editing.md) - Multi-segment editing plans with JSON parsing, validation, and schema generation
- [**Operations**](operations.md) - The `Operation` Pydantic base, auto-registry, and discriminated-union schema
- [**Transforms**](transforms.md) - Frame transformations (cut, resize, resample)
- [**Effects**](effects.md) - Visual effects (blur, zoom, overlays) — including `TranscriptionOverlay` for subtitles

## `videopython.ai`

AI-powered generation and understanding (requires `[ai]` extra):

- [**Generation**](ai/generation.md) - Generate videos, images, music, and speech from text
- [**Understanding**](ai/understanding.md) - Transcribe audio, describe images, detect scenes
- [**Video Analysis**](ai/video_analysis.md) - Aggregate serializable scene-first analysis across audio, visual, and temporal understanding
- [**Dubbing**](ai/dubbing.md) - Dub videos into different languages or revoice with custom text
- [**AI Transforms**](ai/transforms.md) - `FaceTrackingCrop` for face-aware reframing (headroom / thirds, bounded camera speed)

## Import Patterns

```python
# Top-level import for core class
from videopython import Video

# Import specific classes from base
from videopython.base import (
    Video,
    Transcription,
)
from videopython.audio import (
    Audio,
    AudioMetadata,
)
from videopython.editing import (
    CutSeconds,
    Resize,
    Blur,
)

# Import Operation foundation
from videopython.editing import (
    Operation,
    Effect,
    TimeRange,
    OpCategory,
)

# Import AI classes
from videopython.ai import (
    TextToVideo,
    TextToImage,
    AudioToText,
)

# Import dubbing classes
from videopython.ai.dubbing import (
    VideoDubber,
    DubbingResult,
    RevoiceResult,
)
```

# API Reference

videopython is organized into two main modules:

## `videopython.base`

Core video and audio processing functionality with no AI dependencies:

- [**Video**](core/video.md) - Core video class for loading, manipulating, and saving videos
- [**Audio**](core/audio.md) - Core audio class for loading, manipulating, analyzing, and saving audio
- [**Editing Plans (`VideoEdit`)**](editing.md) - Multi-segment editing plans with JSON parsing, validation, and schema generation
- [**Operations**](operations.md) - The `Operation` Pydantic base, auto-registry, and discriminated-union schema
- [**Transforms**](transforms.md) - Frame transformations (cut, resize, resample)
- [**Effects**](effects.md) - Visual effects (blur, zoom, overlays)
- [**Text & Transcription**](text.md) - Subtitle rendering and transcription data structures

## `videopython.ai`

AI-powered generation and understanding (requires `[ai]` extra):

- [**Generation**](ai/generation.md) - Generate videos, images, music, and speech from text
- [**Understanding**](ai/understanding.md) - Transcribe audio, describe images, detect scenes
- [**Video Analysis**](ai/video_analysis.md) - Aggregate serializable scene-first analysis across audio, visual, and temporal understanding
- [**Dubbing**](ai/dubbing.md) - Dub videos into different languages or revoice with custom text
- [**AI Transforms**](ai/transforms.md) - Face tracking crops, split-screen, and auto-framing

## Import Patterns

```python
# Top-level import for core class
from videopython import Video

# Import specific classes from base
from videopython.base import (
    Video,
    Audio,
    AudioMetadata,
    CutSeconds,
    Resize,
    Blur,
    Transcription,
)

# Import Operation foundation
from videopython.base import (
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

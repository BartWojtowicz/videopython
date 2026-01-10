# API Reference

videopython is organized into two main modules:

## `videopython.base`

Core video and audio processing functionality with no AI dependencies:

- [**Video**](core/video.md) - Core video class for loading, manipulating, and saving videos
- [**Audio**](core/audio.md) - Core audio class for loading, manipulating, and saving audio
- [**Transforms**](transforms.md) - Frame transformations (cut, resize, resample)
- [**Transitions**](transitions.md) - Video transitions (fade, blur)
- [**Effects**](effects.md) - Visual effects (blur, zoom, overlays)
- [**Text & Transcription**](text.md) - Subtitle rendering and transcription data structures

## `videopython.ai`

AI-powered generation and understanding (requires `[ai]` extra):

- [**Generation**](ai/generation.md) - Generate videos, images, music, and speech from text
- [**Understanding**](ai/understanding.md) - Transcribe audio, describe images, detect scenes

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
    FadeTransition,
    Transcription,
)

# Import AI classes
from videopython.ai import (
    TextToVideo,
    TextToImage,
    AudioToText,
    SceneDetector,
)
```

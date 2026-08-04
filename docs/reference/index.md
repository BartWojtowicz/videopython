# Reference

Factual description of the API. For learning, start with the
[tutorials](../tutorials/index.md); for a specific task, see the [how-to
guides](../how-to/index.md); for design rationale, see
[explanation](../explanation/index.md).

Everything outside `videopython.ai` (and the optional MCP server) works with a plain
`pip install videopython` — no ML dependencies.

## `videopython.base`

| Page | Contents |
|---|---|
| [Video](video.md) | `Video`, `VideoMetadata`, `FrameIterator` |
| [Transcription and subtitles](transcription.md) | `Transcription`, `TranscriptionSegment`, `TranscriptionWord`, `TranscriptionOverlay` |

Shared result types (`BoundingBox`, `DetectedObject`, `FaceTrack`, `SceneBoundary`,
`SceneDescription`, `AudioClassification`, …) are documented alongside the analyzers that
produce them, in [AI understanding](ai/understanding.md#result-types).

## `videopython.audio`

| Page | Contents |
|---|---|
| [Audio](audio.md) | `Audio` plus its metadata, level, silence and classification types |

## `videopython.editing`

| Page | Contents |
|---|---|
| [Edit plans](video-edit.md) | `VideoEdit`, `SegmentConfig`, the JSON wire format, validation and schema generation |
| [Operations](operations.md) | The `Operation` base, the registry, and the full op table |
| [Transforms](transforms.md) | `Resize`, `Crop`, `ResampleFPS`, `SpeedChange`, `FreezeFrame`, `SilenceRemoval` |
| [Effects](effects.md) | Blur, zoom, color, overlays, fades and the rest |

## `videopython.ai`

Requires the `[ai]` extra ([Install](../install.md)).

| Page | Contents |
|---|---|
| [Generation](ai/generation.md) | `TextToVideo`, `ImageToVideo`, `TextToImage`, `TextToSpeech`, `TextToMusic` |
| [Understanding](ai/understanding.md) | `AudioToText`, `AudioClassifier`, `SceneVLM`, `SemanticSceneDetector`, face trackers, `ObjectDetector`, result types |
| [Video analysis](ai/video-analysis.md) | `VideoAnalyzer`, `VideoAnalysisConfig`, `VideoAnalysis` |
| [Auto-editing](ai/auto-edit.md) | `AutoEditor`, planners, catalog and plan types |
| [Dubbing](ai/dubbing.md) | `VideoDubber`, `DubbingConfig`, result types, supported languages |
| [AI operations](ai/operations.md) | `FaceTrackingCrop`, `ObjectDetectionOverlay`, the detection renderer |

## `videopython.mcp`

| Page | Contents |
|---|---|
| [MCP server](mcp.md) | The `videopython-mcp` tools and resource |

## Import patterns

```python
from videopython import Video                      # the one top-level convenience

from videopython.base import Video, VideoMetadata, FrameIterator, Transcription
from videopython.audio import Audio, AudioMetadata

from videopython.editing import VideoEdit, SegmentConfig      # plans
from videopython.editing import Operation, Effect, TimeRange, OpCategory
from videopython.editing import Resize, Crop, Blur            # operations

from videopython.ai import TextToVideo, TextToImage, AudioToText
from videopython.ai.dubbing import VideoDubber, DubbingResult, RevoiceResult
```

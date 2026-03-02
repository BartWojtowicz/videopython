# Video Analysis

Create a single, serializable, scene-first analysis object.

## Overview

`VideoAnalyzer` runs global passes (transcription + scene detection), then analyzes each scene window.

`VideoAnalysis` can be serialized with:

- `to_dict()` / `from_dict()`
- `to_json()` / `from_json()`
- `save()` / `load()`

The output is centered on `analysis.scenes.samples` (one payload per scene).

## Basic Usage

```python
from videopython.ai import VideoAnalyzer

analyzer = VideoAnalyzer()
analysis = analyzer.analyze_path("video.mp4")

print(analysis.source.title)
print(len(analysis.scenes.samples) if analysis.scenes else 0)
print(analysis.scenes.samples[0].visual_segments if analysis.scenes else [])

# Persist results
analysis.save("video_analysis.json")

# Load later
loaded = analysis.load("video_analysis.json")
print(loaded.run_info.mode)
```

## Configure Analysis

```python
from videopython.ai import VideoAnalysisConfig, VideoAnalyzer

config = VideoAnalysisConfig(
    enabled_analyzers={"audio_to_text", "semantic_scene_detector", "scene_vlm"},
)

analyzer = VideoAnalyzer(config=config)
analysis = analyzer.analyze_path("video.mp4")
```

## Rich Understanding Preset

Use the built-in preset when you want broad understanding coverage across many video types:

```python
from videopython.ai import VideoAnalysisConfig, VideoAnalyzer

config = VideoAnalysisConfig.rich_understanding_preset()
analysis = VideoAnalyzer(config=config).analyze_path("video.mp4")
```

The preset is equivalent to `VideoAnalysisConfig()`.

## Notes on Output Fields

- Use `enabled_analyzers` to run a subset of steps.
- Full transcription is available at `analysis.audio.transcription`.
- Scene payload lives in `analysis.scenes.samples`.
- Each sample includes:
  - scene timing (`start_second`, `end_second`, optional frame bounds)
  - chunked visual segments from scene VLM (`visual_segments`)
  - each visual segment covers up to `_SCENE_VLM_MAX_SEGMENT_SECONDS` and uses `_SCENE_VLM_FRAMES_PER_SEGMENT` frames
  - scene-scoped actions and audio classification
- Scene VLM model size and max segment window are hard-coded module constants by design.

## Classes

::: videopython.ai.VideoAnalysisConfig

::: videopython.ai.VideoAnalyzer

::: videopython.ai.VideoAnalysis

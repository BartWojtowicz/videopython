# Video Analysis

Create a single, serializable analysis object that aggregates multiple AI understanding results.

## Overview

`VideoAnalyzer` orchestrates audio, temporal, motion, and frame analyzers and returns `VideoAnalysis`.

`VideoAnalysis` can be serialized with:

- `to_dict()` / `from_dict()`
- `to_json()` / `from_json()`
- `save()` / `load()`

The path-based flow (`analyze_path`) is designed for bounded frame memory usage by preferring streaming/chunked frame access.

## Basic Usage

```python
from videopython.ai import VideoAnalyzer

analyzer = VideoAnalyzer()
analysis = analyzer.analyze_path("video.mp4")

print(analysis.source.title)
print(analysis.summary)

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
    frame_sampling_mode="hybrid",
    frames_per_second=1.0,
    max_frames=240,
    max_memory_mb=512,  # Optional memory budget for sampled frames
    frame_chunk_size=24,
    action_scope="adaptive",  # "video" | "scene" | "adaptive"
    max_action_scenes=16,
    best_effort=True,
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

The preset enables all analyzers and keeps resource usage bounded with adaptive defaults.

## Notes on New Output Fields

- `FrameSamplingReport.effective_max_frames` shows the effective cap after applying `max_frames` and optional `max_memory_mb`.
- `FrameAnalysisSample.text_regions` contains structured OCR detections (`text`, `confidence`, `bounding_box`) in addition to the existing plain `text` list.
- `summary` now includes richer aggregate signals such as top actions/objects, OCR term frequencies, face presence ratio, and motion distributions when available.

## Classes

::: videopython.ai.VideoAnalysisConfig

::: videopython.ai.VideoAnalyzer

::: videopython.ai.VideoAnalysis

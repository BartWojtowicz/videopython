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
    frames_per_second=0.5,
    max_frames=120,
    frame_chunk_size=16,
    best_effort=True,
)

analyzer = VideoAnalyzer(config=config)
analysis = analyzer.analyze_path("video.mp4")
```

## Classes

::: videopython.ai.VideoAnalysisConfig

::: videopython.ai.VideoAnalyzer

::: videopython.ai.VideoAnalysis

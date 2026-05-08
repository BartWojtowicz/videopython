# Video Analysis

Create a single, serializable, scene-first analysis object.

## Overview

`VideoAnalyzer` runs global passes (transcription + scene detection), then for each detected scene runs the scene-VLM, audio classifier, and per-shot face tracker.

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
if analysis.scenes:
    sample = analysis.scenes.samples[0]
    if sample.scene_description:
        print(sample.scene_description.caption)
        print(sample.scene_description.subjects)
        print(sample.scene_description.shot_type)
    if sample.faces:
        for track in sample.faces:
            print(f"track #{track.track_id}: {track.length} frames")

# Persist results
analysis.save("video_analysis.json")

# Load later
loaded = analysis.load("video_analysis.json")
print(loaded.run_info.mode)
```

## Configure Analysis

Pick which analyzers run, and forward kwargs to their constructors via
`analyzer_params`:

```python
from videopython.ai import VideoAnalysisConfig, VideoAnalyzer

config = VideoAnalysisConfig(
    enabled_analyzers={
        "audio_to_text",
        "semantic_scene_detector",
        "scene_vlm",
        "face_tracker",
    },
    analyzer_params={
        "scene_vlm": {"model_size": "9b"},   # default is "4b"
        "audio_to_text": {
            "model_name": "large",
            "vocabulary": ["Klarna", "Allegro", "InPost"],  # brand-name biasing
        },
    },
)

analyzer = VideoAnalyzer(config=config, sampling="medium")
analysis = analyzer.analyze_path("video.mp4")
```

## Sampling Presets

`VideoAnalyzer(sampling=...)` controls the per-scene SceneVLM frame
budget. The preset tunes the per-scene frame cap, the log-curve
`scale`/`base` used to size short scenes, and the threshold below which
adjacent short scenes get merged into a single VLM call:

| `sampling` | per-scene frame cap | adjacent-merge threshold | typical use |
|---|---|---|---|
| `"low"` | 8 | 20s | quick previews, long videos |
| `"medium"` (default) | 30 | 10s | balanced default |
| `"high"` | 60 | 4s | rich analysis, talking-head depth |

`model_size` and `sampling` are orthogonal kwargs.
`model_size="4b" + sampling="high"` is the rich-analysis pairing;
`model_size="4b" + sampling="low"` is a fast preview.

## Rich Understanding Preset

Use the built-in preset when you want broad understanding coverage across many video types:

```python
from videopython.ai import VideoAnalysisConfig, VideoAnalyzer

config = VideoAnalysisConfig.rich_understanding_preset()
analysis = VideoAnalyzer(config=config).analyze_path("video.mp4")
```

The preset enables every analyzer (`audio_to_text`, `audio_classifier`,
`semantic_scene_detector`, `scene_vlm`, `face_tracker`) and is
equivalent to bare `VideoAnalysisConfig()`.

## Output Shape

- `analysis.audio.transcription` — full Whisper transcription.
- `analysis.scenes.samples` — list of `SceneAnalysisSample`, one per
  scene. Each sample carries:
  - scene timing (`start_second`, `end_second`, `start_frame`, `end_frame`)
  - `scene_description: SceneDescription | None` — the structured
    SceneVLM output (caption + subjects + shot_type). `None` when the
    VLM was disabled or its forward pass failed.
  - `audio_classification: AudioClassification | None` — events and
    clip-level predictions for the scene window.
  - `faces: list[FaceTrack] | None` — per-shot IoU-associated face
    tracks. One list per scene, each track carrying its own per-frame
    indices and bounding boxes.
- `analysis.run_info.stage_durations_seconds` — wall-clock time per
  stage (`whisper`, `scene_detection`, `scene_vlm`, `face_tracker`,
  `audio_classification`, plus `whisper_and_scene_detection_parallel`
  when those two run together).

## Classes

::: videopython.ai.VideoAnalysisConfig

::: videopython.ai.VideoAnalyzer

::: videopython.ai.VideoAnalysis

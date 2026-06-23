# Video Analysis

Create a single, serializable, scene-first analysis object.

## Overview

`VideoAnalyzer` runs global passes (transcription + scene detection), then for each detected scene runs the scene-VLM, audio classifier, and per-shot face tracker.

`VideoAnalysis` is a Pydantic model, so all `BaseModel` serialization
methods are available — `model_dump()`, `model_dump_json()`,
`model_validate()`, `model_validate_json()`. For the common file-I/O
case, the convenience wrappers `analysis.save(path)` and
`VideoAnalysis.load(path)` go through `model_dump_json` /
`model_validate_json` with UTF-8 + parent directory creation.

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
        "scene_vlm": {"model": "qwen3.6:27b"},  # any vision tag you've pulled
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

The SceneVLM `model` (an Ollama tag) and `sampling` are orthogonal: `sampling`
sizes the per-scene frame budget, `model` picks the captioning model. A capable
model with `sampling="high"` is the rich pairing; `sampling="low"` is a fast
preview.

## Rich Understanding Preset

Use the `"full"` profile (the default config) when you want broad understanding coverage across many video types:

```python
from videopython.ai import VideoAnalysisConfig, VideoAnalyzer

config = VideoAnalysisConfig.for_profile("full")
analysis = VideoAnalyzer(config=config).analyze_path("video.mp4")
```

The `"full"` profile enables every analyzer (`audio_to_text`, `audio_classifier`,
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

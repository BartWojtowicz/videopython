# Video analysis

`VideoAnalyzer` runs the global passes (transcription + scene detection), then per
detected scene runs the scene VLM, the audio classifier, and the per-shot face tracker.
The result is one serializable, scene-first `VideoAnalysis`.

```python
from videopython.ai import VideoAnalyzer

analysis = VideoAnalyzer().analyze_path("video.mp4")

print(analysis.source.title)
if analysis.scenes:
    sample = analysis.scenes.samples[0]
    if sample.scene_description:
        print(sample.scene_description.caption, sample.scene_description.shot_type)
    for track in (sample.faces or []):
        print(f"track #{track.track_id}: {track.length} frames")

analysis.save("video_analysis.json")
loaded = VideoAnalysis.load("video_analysis.json")
```

`VideoAnalysis` is a Pydantic model, so `model_dump()`, `model_dump_json()`,
`model_validate()` and `model_validate_json()` all work. `save()` / `load()` wrap the JSON
pair with UTF-8 and parent-directory creation.

## Configuration

```python
from videopython.ai import VideoAnalysisConfig, VideoAnalyzer

config = VideoAnalysisConfig(
    enabled_analyzers={"audio_to_text", "semantic_scene_detector", "scene_vlm", "face_tracker"},
    analyzer_params={
        "scene_vlm": {"model": "qwen3.6:27b"},
        "audio_to_text": {"model_name": "large", "vocabulary": ["Klarna", "Allegro"]},
    },
)
analysis = VideoAnalyzer(config=config, sampling="medium").analyze_path("video.mp4")
```

`VideoAnalysisConfig.for_profile("full")` enables every analyzer (`audio_to_text`,
`audio_classifier`, `semantic_scene_detector`, `scene_vlm`, `face_tracker`) and is
equivalent to a bare `VideoAnalysisConfig()`.

## Sampling presets

`sampling` sizes the per-scene SceneVLM frame budget: the frame cap, the log-curve
`scale`/`base` used for short scenes, and the threshold below which adjacent short scenes
are merged into one VLM call.

| `sampling` | Per-scene frame cap | Adjacent-merge threshold | Typical use |
|---|---|---|---|
| `"low"` | 8 | 20 s | Quick previews, long videos |
| `"medium"` (default) | 30 | 10 s | Balanced |
| `"high"` | 60 | 4 s | Rich analysis, talking-head depth |

`sampling` and the VLM `model` are orthogonal: one sizes the frame budget, the other picks
the captioning model.

## Output shape

- `analysis.audio.transcription` — the full Whisper transcription.
- `analysis.scenes.samples` — one `SceneAnalysisSample` per scene, each carrying:
    - scene timing (`start_second`, `end_second`, `start_frame`, `end_frame`);
    - `scene_description: SceneDescription | None` — caption, subjects, shot_type. `None`
      when the VLM was disabled or its forward pass failed;
    - `audio_classification: AudioClassification | None` — events and clip-level
      predictions for the scene window;
    - `faces: list[FaceTrack] | None` — per-shot IoU-associated tracks, each with its own
      frame indices and boxes.
- `analysis.run_info.stage_durations_seconds` — wall-clock per stage (`whisper`,
  `scene_detection`, `scene_vlm`, `face_tracker`, `audio_classification`, plus
  `whisper_and_scene_detection_parallel` when those two run together).

## Classes

::: videopython.ai.VideoAnalysisConfig

::: videopython.ai.VideoAnalyzer

::: videopython.ai.VideoAnalysis

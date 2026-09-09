# Video analysis

`VideoAnalyzer` runs the global passes (transcription + scene detection), then per
detected scene runs the scene VLM, the audio classifier, and the per-shot face tracker.
The result is one serializable, scene-first `VideoAnalysis`.

```python
from videopython.ai import VideoAnalysis, VideoAnalyzer

analysis = VideoAnalyzer().analyze_path("video.mp4")

print(analysis.source.title)
for outcome in analysis.run_info.analyzer_outcomes:
    print(outcome.analyzer, outcome.status, outcome.reason)
if analysis.scenes:
    sample = analysis.scenes.samples[0]
    if sample.scene_description:
        print(sample.scene_description.caption, sample.scene_description.shot_type)
    for track in (sample.faces or []):
        print(f"track #{track.track_id}: {track.length} frames")

analysis.save("video_analysis.json")
loaded = VideoAnalysis.load("video_analysis.json")
```

`VideoAnalysis` and its nested result types are Pydantic models, so `model_dump()`,
`model_dump_json()`, `model_validate()` and `model_validate_json()` work throughout the
result tree. `save()` / `load()` wrap the JSON pair with UTF-8 and parent-directory
creation. Use `loaded.verify_source()` to compare the recorded source digest with
the current file; `load()` alone does not read the media.

## Saved identity and migration

Every result requires an `AnalysisProvenance` object at `analysis.provenance`:

| Field | Contract |
|---|---|
| `format_version` | Required integer `1`; other versions are rejected. |
| `source_sha256` | SHA256 of the source file, or `null` for unbound in-memory input. |
| `sampling` | The `low`, `medium`, or `high` preset used for the run. |
| `models` | Analyzer ID to a model-ID/revision map, or `null` when provenance is unknown. A model revision can also be `null`. |

Model identities are recorded during analysis. Hugging Face models use repository
revisions; Ollama uses the server's resolved tag and digest. Bundled Silero and
TransNetV2 weights use a `package:<version>` revision, which identifies the package
release rather than a weight-file hash. A disabled stage, an early load failure, or
an unavailable identity can leave unknown provenance. Check stage outcomes separately:
known identity is not a successful-analysis flag.

`analyze_path()` records a resolved absolute source path and hashes the file with
bounded reads. `analyze(video, ...)` leaves `source_sha256=null`, even if a source-path
label was supplied, because the in-memory frames are not verified against that file.
Those unbound results can be serialized but cannot be imported into MCP.

`verify_source()` requires a recorded file identity, compares its digest, and returns
the resolved source path. It starts no models and does not replace saved settings
or unknown provenance with values from the current environment. MCP import/export
performs this check once per call. Source files must remain unchanged while a cached
analysis is in use.

**Migration:** regenerate older saved analyses with `VideoAnalyzer.analyze_path()`.
Files without provenance are rejected; there is no legacy loader. Do not add the
currently installed models as if they had produced an old result. See
[reuse across MCP sessions](../../how-to/mcp-server.md#reuse-analysis-across-sessions).

::: videopython.ai.video_analysis.AnalysisProvenance

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
- `analysis.run_info.analyzer_outcomes` — one record for every analyzer. `status` is
  `completed`, `skipped`, or `failed`. A skipped analyzer has reason `disabled`; a failed
  analyzer has reason `initialization_failed` or `execution_failed`.

## Classes

::: videopython.ai.VideoAnalysisConfig

::: videopython.ai.VideoAnalyzer

::: videopython.ai.VideoAnalysis

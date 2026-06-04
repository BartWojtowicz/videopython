# Editing Plans

`VideoEdit` is a multi-segment editing plan modeled as a Pydantic
`BaseModel`. Each segment selects a time range from a source video and
carries an ordered list of `Operation` instances to run against it.

## At a Glance

- One `operations` list per segment — transforms and effects are sequenced together.
- `post_operations` runs against the concatenated result.
- `validate()` is a dry-run via metadata; no frames are loaded.
- `run()` returns a `Video` in memory; `run_to_file()` streams directly to disk.

## Quick Start

```python
from videopython.editing import VideoEdit

plan = {
    "segments": [
        {
            "source": "input.mp4",
            "start": 5.0,
            "end": 12.0,
            "operations": [
                {"op": "crop", "width": 0.5, "height": 1.0, "mode": "center"},
                {"op": "resize", "width": 1080, "height": 1920},
                {
                    "op": "blur_effect",
                    "mode": "constant",
                    "iterations": 1,
                    "window": {"start": 0.0, "stop": 1.0},
                },
            ],
        },
        {"source": "input.mp4", "start": 20.0, "end": 28.0},
    ],
    "post_operations": [
        {"op": "color_adjust", "brightness": 0.05},
    ],
}

edit = VideoEdit.from_dict(plan)
predicted = edit.validate()        # dry-run via VideoMetadata
video = edit.run()                  # in-memory
video.save("output.mp4")

# Or stream directly to file (constant memory, any video length):
edit.run_to_file("output.mp4", crf=20, preset="medium")
```

## JSON Plan Format

```json
{
  "segments": [
    {
      "source": "path/to/video.mp4",
      "start": 5.0,
      "end": 15.0,
      "operations": [
        {"op": "resize", "width": 1080, "height": 1920},
        {"op": "blur_effect", "mode": "constant", "iterations": 2,
         "window": {"start": 0.0, "stop": 3.0}}
      ]
    }
  ],
  "post_operations": [
    {"op": "color_adjust", "brightness": 0.05}
  ],
  "match_to_lowest_fps": true,
  "match_to_lowest_resolution": true
}
```

Rules:

- `segments` is required and must be non-empty.
- Each op object has an `op` discriminator field; remaining fields belong
  to that op's Pydantic schema. Unknown fields are rejected.
- Effect time windows go in the op's `window` field
  (`{"start": s, "stop": e}`). Either endpoint may be omitted.
- Top-level and segment-level keys are strict (`extra="forbid"`).

## Pipeline Order

`VideoEdit` runs each segment's `operations` in order, concatenates the
results, then applies `post_operations` to the assembled output.

## Streaming Mode (`run_to_file`)

`run_to_file()` pipes ffmpeg decode → per-frame effect chain → ffmpeg
encode, keeping memory constant (~250 MB) regardless of video length.

Each operation contributes either a ffmpeg `-vf` filter
(`op.to_ffmpeg_filter(ctx)`) or a streaming `Effect`
(`op.streamable == True` plus `process_frame`). If any operation is not
streamable, `run_to_file` falls back to eager (`run()` + `save()`).

**Streamable transforms**: `resize`, `crop`, `resample_fps`.
**Streamable effects**: every `Effect` except `add_subtitles`.

## Context Data

Operations that need side-channel data (e.g. `silence_removal` and
`add_subtitles` need a transcription) declare it via
`requires: ClassVar[tuple[str, ...]]`. The runner picks matching keys out
of the `context` dict and threads them into `apply` / `predict_metadata`:

```python
edit = VideoEdit.from_dict(plan)
video = edit.run(context={"transcription": my_transcription})
```

## Validation

`VideoEdit.validate()` chains `Operation.predict_metadata` across the
plan and checks:

- segment `end` is within source duration
- each operation's metadata prediction succeeds
- effect `window` is within the predicted segment duration
- concatenation compatibility (exact fps + dimensions)

Returns the predicted final `VideoMetadata`. On failure it raises
`PlanValidationError` (a `ValueError` subclass, so `except ValueError`
keeps working) carrying structured `.errors` — each a `PlanError` with a
`code`, `location`, and the offending `field`/`value`/`limit`.

For dry-run validation without disk access, pass a pre-built
`VideoMetadata` to `validate_with_metadata(meta_or_dict, context=...)`.

### Repairing window overruns

When a duration-shrinking op (`cut`, `speed_change`, `silence_removal`)
precedes a windowed effect, the effect's `window.stop` can land past the
shortened clip. `run()` clamps it silently, but `validate()` rejects it by
default. To reconcile the two:

- `validate(clamp_windows=True)` (also on `validate_with_metadata`) clamps
  each overrunning `window.stop` to the predicted duration instead of
  raising. Only `window.stop` is clamped; an out-of-range `window.start`
  still raises.
- `repair(source_metadata, context=...)` returns `(repaired_edit, clamps)`
  — a corrected copy of the plan plus the list of `WindowClamp` records —
  leaving the original untouched. It clamps `window.stop` *only*; it is not
  a full validator, so `validate()` the returned plan before running it.
  Prefer `validate(clamp_windows=True)` unless you need the repaired plan
  object back.

## Matching Sources

When multiple segments draw from sources with different fps/resolution,
`VideoEdit` auto-matches:

- `match_to_lowest_fps` (default `true`) — resample all segments to the
  lowest source fps.
- `match_to_lowest_resolution` (default `true`) — resize all segments to
  the lowest source resolution.

Set either flag to `false` to require sources match natively; otherwise
`validate()` / `run()` raises.

## JSON Schema (`json_schema`)

`VideoEdit.json_schema()` returns a JSON Schema for the wire format,
including the discriminated union over every LLM-exposed `Operation`
(server-only ops like `image_overlay` are excluded — see
[Operations](operations.md#llm-exposed-vs-server-only-ops)). Pass it to
any LLM API as a tool/function schema or structured-output format. AI
operations appear in the union only after `import videopython.ai` has run.

```python
schema = VideoEdit.json_schema()
# tools=[{"input_schema": schema}]            # Anthropic
# tools=[{"type": "function", "function": {"parameters": schema}}]  # OpenAI
```

## API Reference

### VideoEdit

::: videopython.editing.VideoEdit

### SegmentConfig

`SegmentConfig` is exported, but most users should construct plans via
`VideoEdit.from_dict(...)` / `VideoEdit.from_json(...)`.

::: videopython.editing.SegmentConfig

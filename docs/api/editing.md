# Editing Plans

`VideoEdit` represents a complete multi-segment editing plan:

1. Extract one or more segments from source videos
2. Apply per-segment transforms, then effects
3. Concatenate processed segments
4. Apply post-assembly transforms, then effects

This is the recommended API for JSON/LLM-generated editing plans.

## At a Glance

- Use `segments[*].transforms` for transforms and `segments[*].effects` for effects
- Use `post_transforms` for transforms after concatenation
- Use `post_effects` for effects after concatenation (not `post_transforms`)
- Validate first with `edit.validate()` before `edit.run()` when plans are generated dynamically

## Quick Start

```python
from videopython.editing import VideoEdit

plan = {
    "segments": [
        {
            "source": "input.mp4",
            "start": 5.0,
            "end": 12.0,
            "transforms": [
                {"op": "crop", "args": {"width": 0.5, "height": 1.0, "mode": "center"}},
                {"op": "resize", "args": {"width": 1080, "height": 1920}},
            ],
            "effects": [
                {"op": "blur", "args": {"mode": "constant", "iterations": 1}, "apply": {"start": 0.0, "stop": 1.0}}
            ],
        },
        {
            "source": "input.mp4",
            "start": 20.0,
            "end": 28.0,
        },
    ],
    "post_effects": [
        {"op": "color_adjust", "args": {"brightness": 0.05}}
    ],
}

edit = VideoEdit.from_dict(plan)

# Dry-run validation using VideoMetadata (no frame loading)
predicted = edit.validate()
print(predicted)

video = edit.run()
video.save("output.mp4")
```

## JSON Plan Format

Top-level shape:

```json
{
  "segments": [
    {
      "source": "path/to/video.mp4",
      "start": 5.0,
      "end": 15.0,
      "transforms": [
        {"op": "crop", "args": {"width": 1080, "height": 1920}}
      ],
      "effects": [
        {"op": "blur_effect", "args": {"mode": "constant", "iterations": 2}, "apply": {"start": 0.0, "stop": 3.0}}
      ]
    }
  ],
  "post_transforms": [
    {"op": "resize", "args": {"width": 1080, "height": 1920}}
  ],
  "post_effects": [
    {"op": "color_adjust", "args": {"brightness": 0.05}}
  ]
}
```

Notes:

- `segments` is required and must be non-empty.
- `post_transforms` and `post_effects` are optional.
- `post_transforms` accepts only transform operations.
- `post_effects` accepts only effect operations.
- Segment keys are strict (`source`, `start`, `end`, `transforms`, `effects`).
- Step keys are strict:
  - transform step: `op`, optional `args`
  - effect step: `op`, optional `args`, optional `apply`
- Unknown top-level keys are ignored for forward compatibility.

## Context Data

Some operations need side-channel data that shouldn't be part of the JSON plan (e.g. transcription for `silence_removal`). Pass it via the `context` parameter:

```python
from videopython.editing import VideoEdit

edit = VideoEdit.from_dict(plan)
video = edit.run(context={"transcription": my_transcription})
```

Operations whose registry spec has the `requires_transcript` tag automatically receive `context["transcription"]` as a keyword argument. Other operations are unaffected.

## Pipeline Order (Enforced)

`VideoEdit` always runs operations in this order:

- Per segment:
  - transforms (in order)
  - effects (in order)
- After concatenation:
  - post transforms (in order)
  - post effects (in order)

Callers do not control transform/effect interleaving. The model enforces this discipline.

## Effect Time Semantics

- Segment effect `apply.start` / `apply.stop` are relative to the segment timeline (segment starts at `0`).
- Post effect `apply.start` / `apply.stop` are relative to the assembled output timeline.

## Validation and Compatibility Checks

`VideoEdit.validate()` performs a dry run using `VideoMetadata`:

- segment time bounds (`start`, `end`)
- transform metadata prediction (for transforms with registered `metadata_method`)
- effect time bounds
- concatenation compatibility (exact `fps`, exact dimensions)

Validation returns the predicted final `VideoMetadata` on success and raises `ValueError` on invalid plans.

Validation behavior notes:

- `cut` metadata prediction mirrors runtime rounded frame slicing semantics (fractional seconds are rounded to frames).
- `crop` metadata prediction mirrors runtime crop slicing behavior, including odd-size center crops and edge clipping.

## JSON Parsing Behavior

### Alias normalization

Input aliases are accepted (for example `blur`), but:

- `VideoEdit.to_dict()` emits canonical operation IDs (for example `blur_effect`)
- `VideoEdit.json_schema()` lists canonical operation IDs only

### Common parser constraints

- `resize` requires at least one non-null dimension (`width` or `height`)
  - valid: `{"op": "resize", "args": {"width": 320}}`
  - valid: `{"op": "resize", "args": {"height": 180}}`
  - invalid: `{"op": "resize"}`
  - invalid: `{"op": "resize", "args": {"width": null, "height": null}}`

### Unsupported operations in JSON plans

The parser rejects operations that are not supported in `VideoEdit` JSON plans, including:

- transitions (`fade_transition`, `blur_transition`, ...)
- multi-source operations (`picture_in_picture`, `split_screen`, ...)
- registered operations that are not JSON-instantiable because required constructor args are excluded from registry specs (for example `ken_burns`, `full_image_overlay`)

### AI operations and lazy registration

AI operation specs are registered only after importing `videopython.ai`.

If a plan references AI ops (for example `face_crop`, `split_screen`), import AI first:

```python
import videopython.ai  # registers AI ops
from videopython.editing import VideoEdit

edit = VideoEdit.from_dict(plan)
```

`videopython.base` does not auto-import AI modules.

## Schema Generation (`json_schema`)

Use `VideoEdit.json_schema()` to get a parser-aligned JSON Schema for the current registry state. The schema is designed to be passed directly to LLM APIs as a tool definition or structured-output format.

```python
from videopython.editing import VideoEdit

schema = VideoEdit.json_schema()
print(schema["properties"]["segments"]["minItems"])  # 1
```

### Using the schema with LLMs

The schema encodes all structural rules - valid operation IDs, required fields, parameter types, and value constraints - so the LLM does not need to learn them from examples:

```python
from videopython.editing import VideoEdit

schema = VideoEdit.json_schema()

# Pass as a tool/function schema to any LLM API:
# - OpenAI: tools=[{"type": "function", "function": {"parameters": schema}}]
# - Anthropic: tools=[{"input_schema": schema}]
# - Any structured-output API that accepts JSON Schema
```

For complete examples with OpenAI and Anthropic APIs, see the [LLM Integration Guide](../guides/llm-integration.md).

### Schema properties

- Built dynamically from the operation registry
- Canonical op IDs only (aliases omitted)
- Excludes unsupported categories/tags/non-JSON-instantiable ops
- Reflects current registration state (AI ops appear only if `videopython.ai` was imported)
- Encodes parser-aligned constraints (for example `resize` requires at least one non-null dimension)
- Includes rich value constraints (`minimum`, `maximum`, `exclusive_minimum`, `enum`) for all parameters

## Serialization (`to_dict`)

`VideoEdit.to_dict()` returns a canonical JSON-ready dict:

- canonical op IDs
- deep-copied step args / apply args
- stable output even if live operation instances are mutated after parsing

---

# Multicam Editing (`MultiCamEdit`)

`MultiCamEdit` is for podcast-style multicam recordings: switch between synchronized
camera angles at specified cut points with transitions, and replace audio with an
external track.

## Quick Start

```python
from videopython.editing import MultiCamEdit, CutPoint
from videopython.base import FadeTransition

edit = MultiCamEdit(
    sources={
        "wide": "cam1.mp4",
        "closeup1": "cam2.mp4",
        "closeup2": "cam3.mp4",
    },
    audio_source="podcast_audio.aac",
    cuts=[
        CutPoint(time=0.0, camera="wide"),
        CutPoint(time=15.0, camera="closeup1", transition=FadeTransition(0.5)),
        CutPoint(time=45.0, camera="wide", transition=FadeTransition(0.5)),
        CutPoint(time=60.0, camera="closeup2"),
    ],
)

video = edit.run()
video.save("podcast.mp4")
```

## Data Model

- **`sources`**: Named camera angles as `dict[str, Path]`.
- **`cuts`**: Ordered list of `CutPoint`s. First cut must start at `time=0.0`.
  Each segment runs from its `time` until the next cut's `time` (last segment
  runs to end of source).
- **`audio_source`**: Optional external audio file. If `None`, output is silent.
  Camera mic audio is always discarded.
- **`default_transition`**: Transition used between cuts when a `CutPoint` has no
  explicit `transition`. Defaults to `InstantTransition` (hard cut).

## Requirements

- All sources must have identical fps and resolution.
- All sources must be synchronized (same start time and duration).
- Cuts must be in strictly ascending order.

## Validation

Validate the plan and predict output metadata without loading video frames:

```python
predicted = edit.validate()
print(predicted)  # VideoMetadata(width=1280, height=720, fps=25, ...)
```

Validation accounts for duration consumed by fade/blur transitions.

## JSON Schema

Use `MultiCamEdit.json_schema()` to get a JSON Schema describing valid plans. Pass it to an LLM API as a tool definition or structured-output format:

```python
schema = MultiCamEdit.json_schema()
# schema includes sources, cuts, transitions, audio_source
```

## JSON Serialization

```python
# Serialize
data = edit.to_dict()

# Deserialize
edit = MultiCamEdit.from_dict(data)
edit = MultiCamEdit.from_json('{"sources": {...}, "cuts": [...]}')
```

## API Reference

### VideoEdit

::: videopython.editing.VideoEdit

### SegmentConfig

`SegmentConfig` is still exported, but most users should construct plans via `VideoEdit.from_dict(...)` or `VideoEdit.from_json(...)`.

::: videopython.editing.SegmentConfig

### MultiCamEdit

::: videopython.editing.MultiCamEdit

### CutPoint

::: videopython.editing.CutPoint

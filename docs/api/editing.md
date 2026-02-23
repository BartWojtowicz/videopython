# Editing Plans (`VideoEdit`)

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
from videopython.base import VideoEdit

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

If a plan references AI ops (for example `face_crop`, `auto_framing`), import AI first:

```python
import videopython.ai  # registers AI ops
from videopython.base import VideoEdit

edit = VideoEdit.from_dict(plan)
```

`videopython.base` does not auto-import AI modules.

## Schema Generation (`json_schema`)

Use `VideoEdit.json_schema()` to get a parser-aligned JSON Schema for the current registry state:

```python
from videopython.base import VideoEdit

schema = VideoEdit.json_schema()
print(schema["properties"]["segments"]["minItems"])  # 1
```

Schema properties:

- Built dynamically from the operation registry
- Canonical op IDs only (aliases omitted)
- Excludes unsupported categories/tags/non-JSON-instantiable ops
- Reflects current registration state (AI ops appear only if `videopython.ai` was imported)
- Encodes parser-aligned constraints (for example `resize` requires at least one non-null dimension)

## Serialization (`to_dict`)

`VideoEdit.to_dict()` returns a canonical JSON-ready dict:

- canonical op IDs
- deep-copied step args / apply args
- stable output even if live operation instances are mutated after parsing

## API Reference

### VideoEdit

::: videopython.base.VideoEdit

### SegmentConfig

`SegmentConfig` is still exported, but most users should construct plans via `VideoEdit.from_dict(...)` or `VideoEdit.from_json(...)`.

::: videopython.base.SegmentConfig

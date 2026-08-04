# Edit plans

`VideoEdit` is a multi-segment editing plan: a Pydantic model whose fields are the JSON
wire format. Each segment selects a time range from a source and carries an ordered list
of [operations](operations.md) to run against it.

- One `operations` list per segment; transforms and effects are sequenced together.
- `post_operations` runs against the concatenated result.
- `validate()` is a dry run over metadata — no frames are loaded.
- `run_to_file()` streams directly to disk and is the only execution engine.

## Usage

```python
from videopython.editing import VideoEdit

edit = VideoEdit.from_dict({
    "segments": [
        {
            "source": "input.mp4",
            "start": 5.0,
            "end": 12.0,
            "operations": [
                {"op": "crop", "width": 0.5, "height": 1.0, "mode": "center"},
                {"op": "resize", "width": 1080, "height": 1920},
                {"op": "blur_effect", "mode": "constant", "iterations": 1,
                 "window": {"start": 0.0, "stop": 1.0}},
            ],
        },
        {"source": "input.mp4", "start": 20.0, "end": 28.0},
    ],
    "post_operations": [{"op": "color_adjust", "brightness": 0.05}],
})

predicted = edit.validate()
edit.run_to_file("output.mp4", crf=20, preset="medium")
```

## JSON wire format

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
- Each op object carries an `op` discriminator; the remaining fields belong to that op's
  schema. Unknown fields are rejected.
- Effect time windows go in the op's `window` field (`{"start": s, "stop": e}`); either
  endpoint may be omitted.
- Top-level and segment-level keys are strict (`extra="forbid"`).
- The cut is the segment's `start`/`end`. There is no `cut` operation —
  `cut`/`cut_frames` are engine-internal.

## Execution order

Each segment's `operations` run in order, the segments are concatenated, then
`post_operations` are applied to the assembled program. What happens under the hood, and
which plan shapes are rejected as unstreamable, is described in
[the streaming engine](../explanation/streaming-engine.md).

### Streamability report

```python
report = edit.streamability()
report.streamable      # will the plan run?
report.unstreamable    # offending ops, with reason and reorder hint
report.errors()        # the same, as structured STREAMING_UNSUPPORTED PlanErrors
```

Purely structural — it touches no media, so it works as a job-admission gate.

::: videopython.editing.StreamabilityReport

::: videopython.editing.OpStreamability

::: videopython.editing.StreamingClass

## Context data

Operations declaring `requires: ClassVar[tuple[str, ...]]` (for example
`silence_removal` and `add_subtitles`, which need `"transcription"`) receive their input
from the runner:

```python
edit.run_to_file("out.mp4", context={"transcription": my_transcription})
```

Time-based values are re-based onto each segment's local timeline before delivery.

## Validation, repair, normalization

| Call | Returns | Raises |
|---|---|---|
| `validate()` | Predicted final `VideoMetadata` | `PlanValidationError` on the first failure |
| `validate_with_metadata(meta, context=...)` | Same, without disk access | Same |
| `check(meta, context=..., clamp_windows=...)` | `list[PlanError]`, `[]` means valid | never |
| `repair(meta, context=..., clamp_op_params=True, clamp_segment_end=False)` | `(repaired_edit, list[PlanRepair])` | only on a segment `end` past the source |
| `normalize_dimensions(meta, target, context=...)` | `(normalized_edit, list[PlanRepair])` | never |

All of them chain each operation's `predict_metadata` and check segment bounds, effect
windows, and concat compatibility (exact fps and dimensions). `normalize_dimensions`
accepts an explicit `(width, height)`, `"first"`, `"largest"`, or `"match"` (the lowest
common resolution).

What each stage owns — and why numeric bounds parse cleanly and fail at validation — is
[the plan lifecycle](../explanation/plan-lifecycle.md).

### Error types

`PlanValidationError` subclasses `ValueError` and carries structured `.errors`.

::: videopython.base.PlanError

::: videopython.base.PlanErrorCode

::: videopython.base.PlanRepair

::: videopython.base.PlanValidationError

## Matching sources

- `match_to_lowest_fps` (default `true`) — resample every segment to the lowest source
  fps.
- `match_to_lowest_resolution` (default `true`) — resize every segment to the lowest
  source resolution.

Set either to `false` to require native agreement; otherwise `validate()` /
`run_to_file()` raises.

## JSON Schema

```python
schema = VideoEdit.json_schema()               # LLM-exposed ops only
strict = VideoEdit.json_schema(strict=True)    # closed provider grammar
```

The default excludes server-only ops such as `image_overlay`
([why](operations.md#llm-exposed-vs-server-only)). AI operations appear only after
`import videopython.ai`. `strict=True` closes every object, makes every property
required, expresses the union as an `anyOf` without a `discriminator`, and hoists `$defs`
to the document root. Usage: [Author edit plans with your own
LLM](../how-to/llm-plans.md).

## Classes

::: videopython.editing.VideoEdit

::: videopython.editing.SegmentConfig

::: videopython.editing.TransitionSpec

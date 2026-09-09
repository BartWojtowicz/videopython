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
        {"source": "input.mp4", "start": 20.0, "end": 28.0,
         "operations": [{"op": "resize", "width": 1080, "height": 1920}]},
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

Time-based values are sliced and shifted onto each segment's local timeline. A bare
value is shared by all sources. For multiple sources, use a map keyed by the exact
`str(segment.source)` value:

```python
context = {"transcription": {"a.mp4": transcript_a, "b.mp4": transcript_b}}
edit.validate(context=context)
edit.run_to_file("out.mp4", context=context)
```

A missing source entry is a validation error for operations that require it.
Time-based context in `post_operations` is unsupported on multi-segment plans.

## Validation, repair, normalization

| Call | Result |
|---|---|
| `validate(context=..., clamp_windows=False)` | Predicted final `VideoMetadata`; raises on the first failure |
| `validate_with_metadata(meta, context=..., clamp_windows=False)` | Same, using supplied source metadata |
| `check(meta, context=..., clamp_windows=False)` | Collects independent plan errors; `[]` means no reported errors |
| `repair(meta, context=..., clamp_op_params=True, clamp_segment_end=False)` | `(repaired_edit, list[PlanRepair])`; a segment end past its source raises unless clamping is enabled |
| `normalize_dimensions(meta, target, context=...)` | `(normalized_edit, list[PlanRepair])` with appended resize operations |

`meta` is one `VideoMetadata` shared by all segments, or a map keyed by
`str(segment.source)`. An incomplete map raises `ValueError`, including in `check`,
`repair`, and `normalize_dimensions`. Supplying metadata avoids source-video probes;
referenced assets such as music and overlays can still be read or probed.

Validation predicts operations in order. A failure can prevent later checks on the
same chain. `check()` also reports structural streamability errors. Repair and
normalization are separate, best-effort steps; check their returned plans again.

`normalize_dimensions` accepts `(width, height)`, `"first"`, `"largest"` (greatest
predicted area), or `"match"` (minimum predicted width and height when resolution
matching is enabled, otherwise the first predictable size).

What each stage owns — and why numeric bounds parse cleanly and fail at validation — is
[the plan lifecycle](../explanation/plan-lifecycle.md).

### Error types

`PlanValidationError` subclasses `ValueError` and carries structured `.errors`.

::: videopython.editing.PlanError

::: videopython.editing.PlanErrorCode

::: videopython.editing.PlanRepair

::: videopython.editing.PlanValidationError

## Matching sources

For multiple segments, `match_to_lowest_fps=True` and
`match_to_lowest_resolution=True` normalize source metadata before each segment's
operations. Resolution matching uses the minimum width and minimum height across
sources. A single-segment plan needs no matching.

Operations can change those dimensions or fps again. The final segment outputs must
agree before concatenation. Set a flag to `False` to skip that source normalization;
your operations must then produce matching outputs. Use `normalize_dimensions()` to
append resizes for a common output canvas. Exact width-and-height resizes can distort
aspect ratio; crop to the target aspect first when that matters.

## Transitions

Set `transition_in` on the incoming segment. The first segment must leave it `None`.
The overlap must be shorter than both adjacent segments after operations.

```python
from videopython.editing import SegmentConfig, TransitionSpec, VideoEdit

edit = VideoEdit(segments=[
    SegmentConfig(source="input.mp4", start=0, end=5),
    SegmentConfig(source="input.mp4", start=5, end=10,
                  transition_in=TransitionSpec(type="dissolve", duration=0.5)),
])
edit.validate()
edit.run_to_file("dissolve.mp4")
```

This produces a 9.5-second program. Each transition subtracts its overlap from the
sum of segment durations. `audio=True` crossfades when both adjacent segments have
audio; otherwise audio joins at the boundary. The generated schema lists the
accepted transition types.

## MusicBed

`music_bed` mixes music across the assembled program, after transitions and
`post_operations`. It is a plan field, not an operation.

```python
from videopython.editing.audio_ops import MusicBed

edit.music_bed = MusicBed(source="music.mp3", gain=0.25, fade_in=0.5, fade_out=1.0)
```

The bed loops by default, is trimmed to the program duration, and does not extend
output length. With `loop=False`, a shorter bed is padded with silence. The source
must have a readable audio stream and is probed during validation.

`duck` reduces bed gain during transcription-derived speech windows: `0` leaves
it unchanged and `1` silences it. Ducking accepts only a single-segment plan. Pass
its source transcription in `context`; without one, the bed mixes at a flat gain.
Speech windows use source timing, so use ducking with operations that keep that timing.

::: videopython.editing.audio_ops.MusicBed

## JSON Schema

```python
schema = VideoEdit.json_schema()               # LLM-exposed ops only
strict = VideoEdit.json_schema(strict=True)    # closed provider grammar
```

The default excludes server-only ops such as `image_overlay`
([why](operations.md#llm-exposed-vs-server-only)). Import the AI operation classes
before schema generation to include them; see the [LLM guide](../how-to/llm-plans.md#include-ai-operations).
`strict=True` closes every object, makes every property
required, expresses the union as an `anyOf` without a `discriminator`, and hoists `$defs`
to the document root. Usage: [Author edit plans with your own
LLM](../how-to/llm-plans.md).

## Classes

::: videopython.editing.VideoEdit

::: videopython.editing.SegmentConfig

::: videopython.editing.TransitionSpec

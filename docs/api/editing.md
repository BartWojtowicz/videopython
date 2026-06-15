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

## The Streaming Engine

Streaming is the **only** execution engine. `run_to_file()` pipes ffmpeg
decode → per-frame effect chain → ffmpeg encode, keeping memory constant
(~250 MB) regardless of video length; `run()` is a view over the same
engine that streams into memory (lossless rawvideo) instead of encoding.

Each operation compiles to one of: an ffmpeg filter
(`op.to_ffmpeg_filter(ctx)`) -- the decode chain, or the encode chain when
ordered after frame effects -- or a per-frame streaming `Effect`
(`op.streamable == True` plus `process_frame`). Plans whose shape has no
streaming strategy (the remaining cases: a frame effect ordered after
encode-stage filters, time-based context after a duration-changing
transform, a few post-op shapes) are rejected with structured
`STREAMING_FALLBACK` errors before any decode.

**Filter transforms**: `resize`, `crop`, `resample_fps`, the
duration-changing `speed_change` and `freeze_frame` (predicted metadata is
folded through the chain so effect windows and audio follow the new
timeline), the transcription-consuming `silence_removal`, and `face_crop`
(ai extra; compile-time detection pass driving a per-frame crop track).
**Streaming effects**: every `Effect`, including the context-requiring
`add_subtitles` (pass `context=` to `run_to_file`).

`add_subtitles` renders via libass: the transcription is compiled to an
ASS document at plan-compile time and burned in by ffmpeg's `subtitles=`
filter — native speed, zero per-frame Python, classified as a `filter` in
the streamability report. It joins the filter chain at its plan position:
the decode chain normally (so transforms may follow it in plan order), or
the encode chain when frame effects precede it (so `[fade, add_subtitles]`
streams too). In memory, `run()` pipes the streamed-in frames through the same filter
via a lossless rawvideo roundtrip — one pixel path everywhere. Requires an ffmpeg built with
libass.

### Streamability report

`edit.streamability()` classifies every op by streaming class without
touching the disk — `filter` (ffmpeg filter chain), `frame_effect`
(`process_frame`), or `unstreamable` (the plan is rejected, with the
reason and a reorder hint on the entry):

```python
report = edit.streamability()
report.streamable        # will the plan run?
report.fallbacks         # the offending ops, with reasons
report.errors()          # the same as structured STREAMING_FALLBACK PlanErrors
```

`edit.check(meta)` reports the same `STREAMING_FALLBACK` errors after the
regular validity errors, and `run()`/`run_to_file()` raise them before any
decode. Streamability is purely structural (op classes, order, plan
shape), so a consumer can gate job admission on the report before
downloading sources.

## Context Data

Operations that need side-channel data (e.g. `silence_removal` and
`add_subtitles` need a transcription) declare it via
`requires: ClassVar[tuple[str, ...]]`. The runner picks matching keys out
of the `context` dict and threads them into `apply` / `predict_metadata`:

```python
edit = VideoEdit.from_dict(plan)
video = edit.run(context={"transcription": my_transcription})
edit.run_to_file("out.mp4", context={"transcription": my_transcription})
```

On both paths, time-based context values are re-based onto each cut
segment's local timeline; on the streaming path the resolved values are
delivered to the effect's `streaming_init`.

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

### Parse vs. validate

Parsing (`from_dict`) owns the plan **shape** (field types, required
fields, unknown ops). The numeric **bounds** of the skeleton — segment
`start`/`end` and effect `window` ranges — are owned by validation, not
parse: a negative `window.start` or a `start >= end` segment *parses* and
is reported by `validate`/`check`. This keeps every numeric violation a
structured, collectable, repairable `PlanError`.

### Collecting every error (`check`)

`check(source_metadata, context=..., clamp_windows=...)` is the
non-raising sibling of `validate_with_metadata`: it runs the same dry-run
but accumulates **every** `PlanError` and returns the list (`[]` == valid),
best-effort isolating failures per segment/op. Use it to re-prompt an LLM
once with all problems instead of one-at-a-time.

### Repairing the mechanical violations (`repair`)

`repair(source_metadata, context=..., clamp_op_params=True,
clamp_segment_end=False)` returns `(repaired_edit, repairs)` — a corrected
deep copy plus a `list[PlanRepair]` changelog (`location`, `field`, `old`,
`new`, `code`), leaving the original untouched. It clamps only the
unambiguous cases:

- effect `window.start`/`window.stop` into `[0, duration]` (segment ops
  and `post_operations`);
- op time fields past the clip end (e.g. `freeze_frame.timestamp`), generic
  via each op's declared `time_fields`;
- a negative segment `start` → `0`, and with `clamp_segment_end=True` a
  segment `end` past the source → the source end.

It never invents intent — a concat mismatch or `end <= start` is left for
`check()` / re-prompting — and never raises on an unrepairable op. A
clampable `window.stop` overrun (a duration-shrinking op before a windowed
effect) is the case `run()` already tolerates; `validate(clamp_windows=True)`
and `check(..., clamp_windows=True)` won't report it either.

### Normalizing concat geometry (`normalize_dimensions`)

`normalize_dimensions(source_metadata, target, context=...)` appends a
per-segment `resize` to a common canvas — `target` is an explicit
`(width, height)`, `"first"`, or `"largest"` — so the "all segments share
dimensions" concat invariant holds by construction. Best-effort and
non-raising like `repair()`/`check()`: a segment it can't yet predict is
left untouched for `check()` to report. Returns `(normalized_edit, repairs)`.

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

Pass `strict=True` (`VideoEdit.json_schema(strict=True)` /
`Operation.json_schema(strict=True)`) for a submittable provider strict-mode
grammar: every object closed (`additionalProperties: false`), every property
`required` (optionality follows the Pydantic type — genuinely optional fields
stay nullable, defaulted-but-required ones keep their concrete type, so a
grammar-valid response always parses back), the op union expressed as an
`anyOf` of closed variants without a `discriminator`, and the union's `$defs`
hoisted to the document root so every `$ref` resolves. Numeric constraints are
preserved, so grammar-constrained decoding makes simple bound violations
impossible at decode time.

## API Reference

### VideoEdit

::: videopython.editing.VideoEdit

### SegmentConfig

`SegmentConfig` is exported, but most users should construct plans via
`VideoEdit.from_dict(...)` / `VideoEdit.from_json(...)`.

::: videopython.editing.SegmentConfig

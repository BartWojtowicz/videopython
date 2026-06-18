# Operations

Every editing primitive in videopython is an `Operation` subclass — a
Pydantic `BaseModel` whose fields ARE the JSON wire format. Subclasses
auto-register via `__pydantic_init_subclass__`, so importing
`videopython.editing` (or `videopython.ai`) populates the registry. The
registry is what `VideoEdit.json_schema()` uses to build the
discriminated-union schema for LLM-driven plan generation.

## Subclass Contract

```python
from typing import ClassVar, Literal

import numpy as np
from pydantic import Field

from videopython.editing import Operation, OpCategory, FilterCtx
from videopython.base.video import VideoMetadata


class Resize(Operation):
    """Resize the video.

    Args:
        width: Target width in pixels.
        height: Target height in pixels.
    """

    op: Literal["resize"] = "resize"            # discriminator + registry key
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata: ...
    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None: ...   # filter-compiled transforms
```

There is **no** `apply()`. Operations execute only through `VideoEdit`'s
streaming engine (`run_to_file`); they never run against a
`Video` directly. A subclass implements:

- `predict_metadata(self, meta) -> VideoMetadata` — predict the output
  `VideoMetadata` and fail fast on plans that would crash at run time.
  Defaults to identity (override on the base `Operation`; on `Effect` it
  is identity, since effects preserve shape and frame count).
- **either** `to_ffmpeg_filter(self, ctx)` (and `to_ffmpeg_audio_filter`
  for a duration-changing transform's audio twin) — for ops compiled into
  the ffmpeg filter chain — **or** `streaming_init(self, total_frames,
  fps, width, height, **context)` + `process_frame(self, frame,
  frame_index)` — for per-frame Python effects.

Notes:

- `op` is a one-value `Literal` *field* (not a `ClassVar`). It flows
  into the JSON wire as the discriminator and is also the registry key.
- `category` is `OpCategory.TRANSFORM`, `OpCategory.EFFECT`, or
  `OpCategory.SPECIAL`.
- Every registered op is streamable, decided structurally by `op.streams()`
  (there is no `streamable` flag): a transform streams iff it implements
  `to_ffmpeg_filter`; an effect iff it implements `process_frame` +
  `streaming_init` (a frame effect) or `to_ffmpeg_filter` + `compiles_to_filter`
  (a filter effect).
- `internal_only: ClassVar[bool] = False`, when `True`, keeps an op OUT of the
  registry — constructed directly by the engine, never a chain op. `cut`/
  `cut_frames` use it, since trimming is the segment's own `start`/`end`.
- Context-dependent ops declare
  `requires: ClassVar[tuple[str, ...]] = ("transcription",)`. The runner
  picks the matching keys out of the `context` dict passed to
  `run_to_file(..., context=...)`, re-bases any time-based values onto the
  segment's local timeline, and threads them into the effect's
  `streaming_init` (and `predict_metadata`) as keyword arguments — or onto
  the `FilterCtx.context` for a filter-compiled op.

## Effects

`Effect(Operation)` adds a `window: TimeRange | None` field and preserves
shape and frame count (so its `predict_metadata` is identity). The
streaming engine resolves `window` against the segment timeline, leaving
frames outside the window untouched. A frame effect implements the
`streaming_init` / `process_frame` pair:

```python
class Glitch(Effect):  # a frame effect: no faithful ffmpeg form
    op: Literal["glitch"] = "glitch"
    # ... fields ...

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray: ...
```

The `window` field on the wire:

```json
{"op": "blur_effect", "mode": "constant", "iterations": 2, "window": {"start": 1.0, "stop": 3.0}}
```

The two text-rendering effects instead compile to a native filter (no per-frame
Python) by setting the `compiles_to_filter` property and implementing
`to_ffmpeg_filter`: `add_subtitles` (libass `subtitles=`) and `text_overlay`
(drawtext). Audio-coupled effects (`Fade`, `VolumeAdjust`) add
`to_ffmpeg_audio_filter` for their audio twin while their video runs per-frame.
Every other (pixel) effect runs vectorised numpy/cv2 in `process_frame`:
benchmarks showed compiling them to ffmpeg filters bought at best ~1.1–1.4x (from
skipping the rawvideo round-trip, not faster compute) and sometimes lost, so the
engine reserves filters for geometry/timing transforms and text rendering.

## Registry API

```python
from videopython.editing import Operation

# Snapshot of {op_id: subclass} for every registered operation:
Operation.registry()

# LLM-safe subset: only ops with llm_exposed=True (omits server-only ops):
Operation.llm_registry()

# Look up by op_id (raises KeyError if unknown):
cls = Operation.get("resize")

# Discriminated-union JSON Schema over the LLM-exposed ops:
schema = Operation.json_schema()
# ...or over every registered op (worker / from_dict path):
full = Operation.json_schema(include_server_only=True)
```

AI operations register lazily, so call `import videopython.ai` before
inspecting the registry if you need `face_crop` and friends.

### LLM-exposed vs server-only ops

Every `Operation` carries `llm_exposed: ClassVar[bool] = True`. Set it to
`False` for ops the model must never emit — typically ops that need a
server-resolved `source` path (`image_overlay`, `full_image_overlay`).
`Operation.llm_registry()` and the default `Operation.json_schema()` /
`VideoEdit.json_schema()` cover only `llm_exposed` ops, while
`Operation.registry()` and `from_dict` still see *all* ops so a stored
plan continues to execute.

The same idea applies at the **field** level: a field declared with
`Field(json_schema_extra={"llm_hidden": True})` is a valid wire field (it
still parses and runs) but is dropped from the LLM-facing schema. This hides
advanced overrides the model shouldn't fill in — e.g. the raw `font_filename`
path on `text_overlay`/`add_subtitles`, whose LLM-facing counterpart is the
`font` name enum. The default `Operation.json_schema()` and
`cls.llm_json_schema()` (below) strip these; `cls.model_json_schema()` keeps
them.

## Discovering Operations

```python
from videopython.editing import Operation, OpCategory

for op_id, cls in Operation.registry().items():
    print(f"{op_id}: {cls.__doc__.splitlines()[0]}")

transforms = {k: v for k, v in Operation.registry().items()
              if v.category is OpCategory.TRANSFORM}
```

## Per-Operation JSON Schema

Every subclass exposes `cls.model_json_schema()` (standard Pydantic),
returning the JSON Schema for that specific op's fields. For an LLM-facing
single-op schema, use `cls.llm_json_schema()` — identical but with
`llm_hidden` fields stripped:

```python
from videopython.editing import Operation

cls = Operation.get("blur_effect")
schema = cls.model_json_schema()         # full (all fields)
llm_schema = cls.llm_json_schema()       # LLM-facing (llm_hidden dropped)
# {
#   "properties": {
#     "op": {"const": "blur_effect", ...},
#     "mode": {"enum": ["constant", "ascending", "descending"], ...},
#     "iterations": {"type": "integer", "minimum": 1, ...},
#     "window": {"anyOf": [{"$ref": "..."}, {"type": "null"}], ...},
#     ...
#   },
#   ...
# }
```

`Operation.json_schema()` is the union over the LLM-exposed ops (pass
`include_server_only=True` for all of them), and that's the schema
`VideoEdit.json_schema()` embeds for the `operations` field.

## Registered Operations

### Base (no AI dependencies)

`cut`/`cut_frames` are internal-only: the engine trims each segment via its
`start`/`end`, so they are not chain ops and do not appear here. Every registered
op below is streamable (it compiles to an ffmpeg filter or is a per-frame effect).

| ID | Class | Category | Streamable |
|---|---|---|---|
| `resize` | `Resize` | transform | yes |
| `resample_fps` | `ResampleFPS` | transform | yes |
| `crop` | `Crop` | transform | yes |
| `speed_change` | `SpeedChange` | transform | yes — compiles to `setpts` + CFR resample; audio time-stretched in sync |
| `freeze_frame` | `FreezeFrame` | transform | yes — compiles to a `loop`-based chain; silence inserted in the audio |
| `silence_removal` | `SilenceRemoval` | transform | yes — `select` keep-window cut (requires `transcription` context) |
| `blur_effect` | `Blur` | effect | yes |
| `zoom_effect` | `Zoom` | effect | yes |
| `color_adjust` | `ColorGrading` | effect | yes |
| `vignette` | `Vignette` | effect | yes |
| `ken_burns` | `KenBurns` | effect | yes |
| `full_image_overlay` † | `FullImageOverlay` | effect | yes |
| `image_overlay` † | `ImageOverlay` | effect | yes |
| `fade` | `Fade` | effect | yes |
| `volume_adjust` | `VolumeAdjust` | effect | yes |
| `text_overlay` | `TextOverlay` | effect | yes |
| `add_subtitles` | `TranscriptionOverlay` | effect | yes — compiles to a libass `subtitles=` filter (requires `transcription` context) |
| `shake` | `Shake` | effect | yes |
| `punch_in` | `PunchIn` | effect | yes |
| `flash` | `Flash` | effect | yes |
| `chromatic_aberration` | `ChromaticAberration` | effect | yes |
| `glitch` | `Glitch` | effect | yes |
| `film_grain` | `FilmGrain` | effect | yes |
| `sharpen` | `Sharpen` | effect | yes |
| `pixelate` | `Pixelate` | effect | yes |
| `mirror_flip` | `MirrorFlip` | effect | yes |
| `kaleidoscope` | `Kaleidoscope` | effect | yes |

† Server-only (`llm_exposed=False`): excluded from `Operation.llm_registry()`
and the default LLM-facing schema because they need a server-resolved
`source` path. Still executable via `from_dict` / `Operation.registry()`.

### AI (require `import videopython.ai`)

| ID | Class | Category | Streamable |
|---|---|---|---|
| `face_crop` | `FaceTrackingCrop` | transform | yes — compile-time detection pass drives a per-frame crop track |
| `object_detection_overlay` | `ObjectDetectionOverlay` | effect | yes — per-frame box overlay; YOLOv8 detection on a `detection_interval` cadence; bounded memory, not bounded compute |

## API Reference

### Operation

::: videopython.editing.Operation

### Effect

::: videopython.editing.Effect

### TimeRange

::: videopython.editing.TimeRange

### OpCategory

::: videopython.editing.OpCategory

### FilterCtx

::: videopython.editing.FilterCtx

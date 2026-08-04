# Operations

Every editing primitive is an `Operation` subclass — a Pydantic model whose fields are the
JSON wire format. Subclasses auto-register on definition, so importing
`videopython.editing` (or `videopython.ai`) populates the registry that
[`VideoEdit.json_schema()`](video-edit.md#json-schema) builds its discriminated union
from.

Operations execute only through the [streaming
engine](../explanation/streaming-engine.md); there is no `apply()`.

## All registered operations

Every registered operation streams. `cut`/`cut_frames` are `internal_only` — the engine
builds them from each segment's `start`/`end` — so they are not chain ops and are not
listed.

### Transforms

| op | Class | Notes |
|---|---|---|
| `resize` | `Resize` | Aspect-preserving when only one dimension is given |
| `resample_fps` | `ResampleFPS` | Change frame rate |
| `crop` | `Crop` | Pixel ints or normalized 0–1 fractions |
| `speed_change` | `SpeedChange` | Constant or ramping; compiles to `setpts` + CFR resample, audio time-stretched in sync |
| `freeze_frame` | `FreezeFrame` | Compiles to a `loop` chain; silence inserted into the audio |
| `silence_removal` | `SilenceRemoval` | `select` keep-window cut; requires `transcription` context |

### Effects

| op | Class | Description |
|---|---|---|
| `blur_effect` | `Blur` | Gaussian blur, constant or ramping |
| `zoom_effect` | `Zoom` | Time-varying zoom in/out |
| `color_adjust` | `ColorGrading` | Brightness / contrast / saturation / temperature |
| `vignette` | `Vignette` | Radial darkening from the edges |
| `ken_burns` | `KenBurns` | Pan-and-zoom between two bounding boxes |
| `full_image_overlay` † | `FullImageOverlay` | Composite a full-frame image |
| `image_overlay` † | `ImageOverlay` | Scaled, positioned raster/SVG image (logo, watermark) |
| `fade` | `Fade` | Audio + video fade in / out / in_out |
| `volume_adjust` | `VolumeAdjust` | Audio-only |
| `text_overlay` | `TextOverlay` | Rendered text; compiles to drawtext |
| `add_subtitles` | `TranscriptionOverlay` | Word-level subtitles via libass; requires `transcription` context |
| `shake` | `Shake` | Per-frame jitter (random / rhythmic / decay) |
| `punch_in` | `PunchIn` | Snap-zoom emphasis with optional release |
| `flash` | `Flash` | Solid-color frame flash with attack/decay |
| `chromatic_aberration` | `ChromaticAberration` | R/B channel split (horizontal / vertical / radial) |
| `glitch` | `Glitch` | Random horizontal slice displacement + channel offsets |
| `film_grain` | `FilmGrain` | Additive seeded noise (mono or RGB) |
| `sharpen` | `Sharpen` | Unsharp-mask sharpening |
| `pixelate` | `Pixelate` | Mosaic blocks, full frame or region |
| `mirror_flip` | `MirrorFlip` | Flip or reflect one half onto the other |
| `kaleidoscope` | `Kaleidoscope` | N-way radial mirror around the center |

† Server-only (`llm_exposed=False`): excluded from `llm_registry()` and the default
LLM-facing schema because they need a server-resolved `source` path. Still executable via
`from_dict` and `registry()`.

### AI (require `import videopython.ai`)

| op | Class | Notes |
|---|---|---|
| `face_crop` | `FaceTrackingCrop` | Transform; a compile-time detection pass drives a per-frame crop track |
| `object_detection_overlay` | `ObjectDetectionOverlay` | Effect; per-frame boxes, D-FINE detection on a `detection_interval` cadence. Bounded memory, not bounded compute |

See [AI operations](ai/operations.md).

## Registry API

```python
from videopython.editing import Operation, OpCategory

Operation.registry()                    # {op_id: subclass} for every registered op
Operation.llm_registry()                # only llm_exposed=True ops
Operation.get("resize")                 # by op_id; KeyError if unknown

Operation.json_schema()                 # discriminated union over LLM-exposed ops
Operation.json_schema(include_server_only=True)   # over every registered op
Operation.json_schema(strict=True)      # closed provider grammar

cls = Operation.get("blur_effect")
cls.model_json_schema()                 # full Pydantic schema, all fields
cls.llm_json_schema()                   # LLM-facing: llm_hidden fields dropped
```

Filter by category:

```python
transforms = {k: v for k, v in Operation.registry().items()
              if v.category is OpCategory.TRANSFORM}
```

### LLM-exposed vs server-only

`llm_exposed: ClassVar[bool] = True` on every operation; set `False` for ops a model must
never emit (typically ones needing a server-resolved path). At field level,
`Field(json_schema_extra={"llm_hidden": True})` keeps a field on the wire but drops it
from the LLM-facing schema. Rationale:
[LLM-first design](../explanation/llm-first-design.md#two-visibility-switches).

## Writing an operation

```python
from typing import ClassVar, Literal

from pydantic import Field

from videopython.base.video import VideoMetadata
from videopython.editing import FilterCtx, OpCategory, Operation


class Resize(Operation):
    """Resize the video.

    Args:
        width: Target width in pixels.
        height: Target height in pixels.
    """

    op: Literal["resize"] = "resize"              # discriminator + registry key
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata: ...
    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None: ...
```

A subclass implements `predict_metadata(meta) -> VideoMetadata` (defaults to identity;
on `Effect` it is always identity) and **either**:

- `to_ffmpeg_filter(ctx)` — compiled into the FFmpeg filter chain. Duration-changing
  transforms add `to_ffmpeg_audio_filter` for their audio twin; **or**
- `streaming_init(total_frames, fps, width, height, **context)` + `process_frame(frame,
  frame_index)` — a per-frame Python effect.

Other class-level knobs:

| Attribute | Meaning |
|---|---|
| `category` | `OpCategory.TRANSFORM`, `EFFECT`, or `SPECIAL` |
| `internal_only` | `True` keeps the op out of the registry (engine-constructed only) |
| `requires` | Tuple of context keys the runner must supply |
| `llm_exposed` / field `llm_hidden` | Schema visibility, above |

Streamability is derived structurally by `op.streams()` — there is no flag. A transform
streams if it implements `to_ffmpeg_filter`; an effect if it implements
`process_frame` + `streaming_init`, or `to_ffmpeg_filter` + `compiles_to_filter`.

## Effects

`Effect(Operation)` adds `window: TimeRange | None` and preserves shape and frame count.
The engine resolves the window against the segment timeline and leaves frames outside it
untouched.

```json
{"op": "blur_effect", "mode": "constant", "iterations": 2,
 "window": {"start": 1.0, "stop": 3.0}}
```

## Classes

::: videopython.editing.Operation

::: videopython.editing.Effect

::: videopython.editing.TimeRange

::: videopython.editing.OpCategory

::: videopython.editing.FilterCtx

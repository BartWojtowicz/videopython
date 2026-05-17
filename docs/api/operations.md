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
from pydantic import Field

from videopython.editing import Operation, OpCategory
from videopython.base.video import Video, VideoMetadata


class Resize(Operation):
    """Resize the video.

    Args:
        width: Target width in pixels.
        height: Target height in pixels.
    """

    op: Literal["resize"] = "resize"            # discriminator + registry key
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    streamable: ClassVar[bool] = True

    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)

    def apply(self, video: Video) -> Video: ...
    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata: ...
    def to_ffmpeg_filter(self, ctx) -> str | None: ...   # streamable transforms only
```

Notes:

- `op` is a one-value `Literal` *field* (not a `ClassVar`). It flows
  into the JSON wire as the discriminator and is also the registry key.
- `category` is `OpCategory.TRANSFORM`, `OpCategory.EFFECT`, or
  `OpCategory.SPECIAL`.
- `streamable: ClassVar[bool] = True` lets `VideoEdit.run_to_file()`
  treat this op as streaming-compatible. For transforms that means
  implementing `to_ffmpeg_filter`; for effects that means implementing
  `process_frame` and `streaming_init`.
- Context-dependent ops declare
  `requires: ClassVar[tuple[str, ...]] = ("transcription",)` and use a
  wider `apply` signature (`def apply(self, video, transcription=None)`)
  with `# type: ignore[override]`.

## Effects

`Effect(Operation)` adds a `window: TimeRange | None` field and a
shape-and-frame-count-preserving invariant. Subclasses override
`_apply(self, video)`; the base `Effect.apply` resolves the window,
slices the video, runs `_apply`, splices the result back, and asserts
the invariant.

```python
class ColorGrading(Effect):
    op: Literal["color_adjust"] = "color_adjust"
    streamable: ClassVar[bool] = True

    brightness: float = Field(0.0, ge=-1, le=1)
    # ... more fields ...

    def _apply(self, video: Video) -> Video: ...
```

The `window` field on the wire:

```json
{"op": "color_adjust", "brightness": 0.1, "window": {"start": 1.0, "stop": 3.0}}
```

Audio-mutating effects (`Fade`, `VolumeAdjust`) and ops that don't fit
the frame-preserving shape (`TranscriptionOverlay`) override `apply`
directly.

## Registry API

```python
from videopython.editing import Operation

# Snapshot of {op_id: subclass} for every registered operation:
Operation.registry()

# Look up by op_id (raises KeyError if unknown):
cls = Operation.get("resize")

# Discriminated-union JSON Schema covering every registered op:
schema = Operation.json_schema()
```

AI operations register lazily, so call `import videopython.ai` before
inspecting the registry if you need `face_crop` and friends.

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
returning the JSON Schema for that specific op's fields:

```python
from videopython.editing import Operation

cls = Operation.get("blur_effect")
schema = cls.model_json_schema()
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

`Operation.json_schema()` is the union over all registered ops, and
that's the schema `VideoEdit.json_schema()` embeds for the `operations`
field.

## Registered Operations

### Base (no AI dependencies)

| ID | Class | Category | Streamable |
|---|---|---|---|
| `cut_frames` | `CutFrames` | transform | no |
| `cut` | `CutSeconds` | transform | no |
| `resize` | `Resize` | transform | yes |
| `resample_fps` | `ResampleFPS` | transform | yes |
| `crop` | `Crop` | transform | yes |
| `speed_change` | `SpeedChange` | transform | no |
| `reverse` | `Reverse` | transform | no |
| `freeze_frame` | `FreezeFrame` | transform | no |
| `silence_removal` | `SilenceRemoval` | transform | no (requires `transcription`) |
| `blur_effect` | `Blur` | effect | yes |
| `zoom_effect` | `Zoom` | effect | yes |
| `color_adjust` | `ColorGrading` | effect | yes |
| `vignette` | `Vignette` | effect | yes |
| `ken_burns` | `KenBurns` | effect | yes |
| `full_image_overlay` | `FullImageOverlay` | effect | yes |
| `image_overlay` | `ImageOverlay` | effect | yes |
| `fade` | `Fade` | effect | yes |
| `volume_adjust` | `VolumeAdjust` | effect | yes |
| `text_overlay` | `TextOverlay` | effect | yes |
| `add_subtitles` | `TranscriptionOverlay` | effect | no (requires `transcription`) |
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

### AI (require `import videopython.ai`)

| ID | Class | Category | Streamable |
|---|---|---|---|
| `face_crop` | `FaceTrackingCrop` | transform | no |

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

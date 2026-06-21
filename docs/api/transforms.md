# Transforms

Transforms are `Operation` subclasses that produce a new `Video` from a
single input video. They may change dimensions, fps, duration, or frame
count. See [Operations](operations.md) for the base contract.

## Usage

Transforms are not applied to a `Video` directly. They run only through
the streaming engine: add the operation(s) to a `VideoEdit` and render
with `run_to_file`.

The time cut is the segment's own `start`/`end`; resizing, cropping, and fps
changes go in `operations`:

```python
from videopython.editing import VideoEdit, SegmentConfig, Resize, Crop

edit = VideoEdit(segments=[SegmentConfig(source="input.mp4", start=0, end=10, operations=[
    Crop(width=0.5, height=0.5),        # 50% center crop
    Resize(width=1280, height=720),
])])
edit.run_to_file("output.mp4")
```

A `SegmentConfig`'s `operations` list also accepts the inline dict form:

```python
plan = {
    "segments": [{
        "source": "input.mp4",
        "start": 0,
        "end": 10,
        "operations": [
            {"op": "crop", "width": 0.5, "height": 0.5},
            {"op": "resize", "width": 1280, "height": 720},
        ],
    }]
}
```

## Available Transforms

Cutting is the segment's own `start`/`end`; `cut`/`cut_frames` are internal-only
(constructed by the engine, not usable as chain ops), so they are omitted here.

| op | Class | Streamable | Notes |
|---|---|---|---|
| `resize` | `Resize` | yes | Resize, optional aspect-preserving |
| `resample_fps` | `ResampleFPS` | yes | Change frame rate |
| `crop` | `Crop` | yes | Pixel or normalized 0–1 fractions |
| `speed_change` | `SpeedChange` | yes | Constant or ramping speed |
| `freeze_frame` | `FreezeFrame` | yes | Hold a frame for a duration |
| `silence_removal` | `SilenceRemoval` | yes | Cuts silent gaps; requires transcription context |

## Crop Coordinates

`Crop` accepts pixel ints or normalized 0–1 floats. Floats in `(0, 1]`
are treated as fractions of source dimensions; everything else is
interpreted as a pixel count.

Add any of these to a `VideoEdit` and render with `run_to_file`:

```python
from videopython.editing import Crop, CropMode

video_op = Crop(width=640, height=480)                              # pixels
video_op = Crop(width=0.5, height=0.5)                              # 50% center crop
video_op = Crop(width=0.5, height=1.0, x=0.5, y=0.0, mode=CropMode.CUSTOM)
```

## Context-Dependent Transforms

`SilenceRemoval` declares `requires = ("transcription",)`. Add it to a
segment's `operations` and pass the transcription to the runner via
`context`:

```python
edit = VideoEdit(segments=[SegmentConfig(source="input.mp4", start=0, end=10, operations=[
    SilenceRemoval(),
])])
edit.run_to_file("out.mp4", context={"transcription": my_transcription})
```

## API Reference

!!! note "`CutSeconds` / `CutFrames` are engine-internal"
    These are documented because the engine constructs them from each segment's
    `start`/`end`, but they are `internal_only` — not in the op registry or the LLM
    schema, and rejected if placed in a plan's `operations` list. Cut via the
    segment range instead.

### CutSeconds

::: videopython.editing.CutSeconds

### CutFrames

::: videopython.editing.CutFrames

### Resize

::: videopython.editing.Resize

### ResampleFPS

::: videopython.editing.ResampleFPS

### Crop

::: videopython.editing.Crop

### CropMode

::: videopython.editing.CropMode

### SpeedChange

::: videopython.editing.SpeedChange

### FreezeFrame

::: videopython.editing.FreezeFrame

### SilenceRemoval

::: videopython.editing.SilenceRemoval

---

For AI-powered transforms (face tracking, auto-framing), see
[AI Transforms](ai/transforms.md).

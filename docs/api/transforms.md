# Transforms

Transforms are `Operation` subclasses that produce a new `Video` from a
single input video. They may change dimensions, fps, duration, or frame
count. See [Operations](operations.md) for the base contract.

## Usage

```python
from videopython.base import Video, Resize, Crop, CutSeconds

video = Video.from_path("input.mp4")
video = CutSeconds(start=0.0, end=10.0).apply(video)
video = Crop(width=0.5, height=0.5).apply(video)        # 50% center crop
video = Resize(width=1280, height=720).apply(video)
video.save("output.mp4")
```

Inside a `VideoEdit` plan, transforms go in the `operations` list with
their fields inline:

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

| op | Class | Streamable | Notes |
|---|---|---|---|
| `cut_frames` | `CutFrames` | no | Cut by frame range |
| `cut` | `CutSeconds` | no | Cut by time range |
| `resize` | `Resize` | yes | Resize, optional aspect-preserving |
| `resample_fps` | `ResampleFPS` | yes | Change frame rate |
| `crop` | `Crop` | yes | Pixel or normalized 0–1 fractions |
| `speed_change` | `SpeedChange` | no | Constant or ramping speed |
| `reverse` | `Reverse` | no | Reverse playback |
| `freeze_frame` | `FreezeFrame` | no | Hold a frame for a duration |
| `silence_removal` | `SilenceRemoval` | no | Requires transcription context |

## Crop Coordinates

`Crop` accepts pixel ints or normalized 0–1 floats. Floats in `(0, 1]`
are treated as fractions of source dimensions; everything else is
interpreted as a pixel count.

```python
from videopython.base import Crop, CropMode

Crop(width=640, height=480).apply(video)                              # pixels
Crop(width=0.5, height=0.5).apply(video)                              # 50% center crop
Crop(width=0.5, height=1.0, x=0.5, y=0.0, mode=CropMode.CUSTOM).apply(video)
```

## Context-Dependent Transforms

`SilenceRemoval` declares `requires = ("transcription",)`. Inside a
`VideoEdit`, pass it via `context`; standalone, pass it directly to
`apply`:

```python
edit.run(context={"transcription": my_transcription})
# or
SilenceRemoval().apply(video, transcription=my_transcription)
```

## API Reference

### CutSeconds

::: videopython.base.CutSeconds

### CutFrames

::: videopython.base.CutFrames

### Resize

::: videopython.base.Resize

### ResampleFPS

::: videopython.base.ResampleFPS

### Crop

::: videopython.base.Crop

### CropMode

::: videopython.base.CropMode

### SpeedChange

::: videopython.base.SpeedChange

### Reverse

::: videopython.base.Reverse

### FreezeFrame

::: videopython.base.FreezeFrame

### SilenceRemoval

::: videopython.base.SilenceRemoval

---

For AI-powered transforms (face tracking, auto-framing), see
[AI Transforms](ai/transforms.md).

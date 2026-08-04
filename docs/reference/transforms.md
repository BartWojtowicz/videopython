# Transforms

`Operation` subclasses that may change dimensions, fps, duration, or frame count. The base
contract is in [Operations](operations.md); AI-powered transforms are in
[AI operations](ai/operations.md).

## Usage

Transforms run only through the streaming engine — put them in a segment's `operations`
and render with `run_to_file`. The time cut is the segment's own `start`/`end`.

```python
from videopython.editing import VideoEdit, SegmentConfig, Crop, Resize

edit = VideoEdit(segments=[SegmentConfig(source="input.mp4", start=0, end=10, operations=[
    Crop(width=0.5, height=0.5),         # 50% center crop
    Resize(width=1280, height=720),
])])
edit.run_to_file("output.mp4")
```

The dict form is equivalent:

```json
{"segments": [{"source": "input.mp4", "start": 0, "end": 10, "operations": [
  {"op": "crop", "width": 0.5, "height": 0.5},
  {"op": "resize", "width": 1280, "height": 720}
]}]}
```

## Crop coordinates

`Crop` accepts pixel ints or normalized floats. A float in `(0, 1]` is a fraction of the
source dimension; anything else is a pixel count.

```python
from videopython.editing import Crop, CropMode

Crop(width=640, height=480)                                     # pixels
Crop(width=0.5, height=0.5)                                     # 50% center crop
Crop(width=0.5, height=1.0, x=0.5, y=0.0, mode=CropMode.CUSTOM)
```

## Context-dependent transforms

`SilenceRemoval` declares `requires = ("transcription",)`:

```python
edit = VideoEdit(segments=[SegmentConfig(source="input.mp4", start=0, end=10,
                                         operations=[SilenceRemoval()])])
edit.run_to_file("out.mp4", context={"transcription": my_transcription})
```

## Classes

::: videopython.editing.Resize

::: videopython.editing.ResampleFPS

::: videopython.editing.Crop

::: videopython.editing.CropMode

::: videopython.editing.SpeedChange

::: videopython.editing.FreezeFrame

::: videopython.editing.SilenceRemoval

### Engine-internal cuts

!!! note
    `CutSeconds` / `CutFrames` are `internal_only`: the engine constructs them from each
    segment's `start`/`end`. They are not in the registry or the LLM schema and are
    rejected if placed in a plan's `operations` list. Cut via the segment range instead.

::: videopython.editing.CutSeconds

::: videopython.editing.CutFrames

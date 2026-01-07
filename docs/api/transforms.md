# Transforms

Transformations modify video frames (cutting, resizing, resampling).

## TransformationPipeline

Chain multiple transformations together:

```python
from videopython.base import TransformationPipeline, CutSeconds, Resize

pipeline = TransformationPipeline([
    CutSeconds(start=0, end=10),
    Resize(width=1280, height=720),
])
video = pipeline.run(video)
```

::: videopython.base.TransformationPipeline

## Transformation (Base Class)

::: videopython.base.Transformation

## CutSeconds

::: videopython.base.CutSeconds

## CutFrames

::: videopython.base.CutFrames

## Resize

::: videopython.base.Resize

## ResampleFPS

::: videopython.base.ResampleFPS

## Crop

::: videopython.base.Crop

## CropMode

::: videopython.base.CropMode

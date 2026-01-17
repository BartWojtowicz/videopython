# Transforms

Transformations modify video frames (cutting, resizing, resampling).

## Fluent API

Chain transformations directly on Video objects:

```python
from videopython.base import Video

# Chain multiple transformations
video = Video.from_path("input.mp4").cut(0, 10).resize(1280, 720).resample_fps(30)

# Validate operations before executing (using metadata)
output_meta = video.metadata.cut(0, 10).resize(1280, 720).resample_fps(30)
print(f"Output will be: {output_meta}")
```

## Available Methods

| Video Method | VideoMetadata Method | Description |
|--------------|---------------------|-------------|
| `video.cut(start, end)` | `meta.cut(start, end)` | Cut by time range (seconds) |
| `video.cut_frames(start, end)` | `meta.cut_frames(start, end)` | Cut by frame range |
| `video.resize(width, height)` | `meta.resize(width, height)` | Resize dimensions |
| `video.crop(width, height)` | `meta.crop(width, height)` | Center crop |
| `video.resample_fps(fps)` | `meta.resample_fps(fps)` | Change frame rate |
| `video.transition_to(other, t)` | `meta.transition_to(other, time)` | Combine videos |
| `video.ken_burns(start, end, easing)` | - | Pan-and-zoom effect |
| `video.picture_in_picture(overlay, ...)` | - | Overlay video as PiP |

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

## SpeedChange

::: videopython.base.SpeedChange

## PictureInPicture

::: videopython.base.PictureInPicture

---

For AI-powered transforms (face tracking, auto-framing), see [AI Transforms](ai/transforms.md).

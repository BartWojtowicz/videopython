# Effects

Effects modify video frames without changing their count or dimensions.

## Usage

```python
from videopython.base import Video, Blur, Zoom

video = Video.from_path("input.mp4")

# Apply blur effect
blur = Blur(mode="constant", iterations=50)
video = blur.apply(video, start=0, stop=2.0)

# Apply zoom effect
zoom = Zoom(zoom_factor=1.5, mode="in")
video = zoom.apply(video)
```

## Effect (Base Class)

::: videopython.base.Effect

## Blur

::: videopython.base.Blur

## Zoom

::: videopython.base.Zoom

## FullImageOverlay

::: videopython.base.FullImageOverlay

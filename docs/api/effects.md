# Effects

Effects modify video frames without changing their count or dimensions.

## Usage

```python
from videopython.base import Video, Blur, Zoom, ColorGrading, Vignette, KenBurns, BoundingBox
from videopython.base import Fade, VolumeAdjust, TextOverlay

video = Video.from_path("input.mp4")

# Apply blur effect
blur = Blur(mode="constant", iterations=50)
video = blur.apply(video, start=0, stop=2.0)

# Apply zoom effect
zoom = Zoom(zoom_factor=1.5, mode="in")
video = zoom.apply(video)

# Color grading
grading = ColorGrading(brightness=1.1, contrast=1.2, saturation=1.1, temperature=0.1)
video = grading.apply(video)

# Vignette effect
vignette = Vignette(strength=0.5, radius=0.8)
video = vignette.apply(video)

# Ken Burns pan-and-zoom (fluent API)
start_region = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)  # Top-left quarter
end_region = BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5)    # Bottom-right quarter
video = video.ken_burns(start_region, end_region, easing="ease_in_out")

# Fade in from black over 1 second
video = Fade(mode="in", duration=1.0).apply(video)

# Fade out to black at the end
video = Fade(mode="out", duration=0.5).apply(video)

# Adjust volume (mute first 2 seconds)
video = VolumeAdjust(volume=0.0).apply(video, start=0, stop=2.0)

# Add text overlay
video = TextOverlay(text="Hello World", position=(0.5, 0.9), font_size=48).apply(video)
```

## Effect (Base Class)

::: videopython.base.Effect

## AudioEffect (Base Class)

Audio-only effects that inherit from `Effect` for execution engine compatibility but skip frame processing entirely.

::: videopython.base.AudioEffect

## Blur

::: videopython.base.Blur

## Zoom

::: videopython.base.Zoom

## FullImageOverlay

::: videopython.base.FullImageOverlay

## ColorGrading

::: videopython.base.ColorGrading

## Vignette

::: videopython.base.Vignette

## KenBurns

::: videopython.base.KenBurns

## Fade

::: videopython.base.Fade

## VolumeAdjust

::: videopython.base.VolumeAdjust

## TextOverlay

::: videopython.base.TextOverlay

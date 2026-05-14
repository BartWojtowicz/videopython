# Effects

Effects are `Operation` subclasses that preserve video shape and frame
count. Each carries an optional `window: TimeRange | None` field that
limits the effect to a sub-range of the video. See
[Operations](operations.md) for the base contract.

## Usage

```python
from videopython.base import Video, BoundingBox
from videopython.editing import (
    Blur, Zoom, ColorGrading, Vignette, KenBurns,
    Fade, VolumeAdjust, TextOverlay, TimeRange,
)

video = Video.from_path("input.mp4")

# Effect across the full duration:
video = Blur(mode="constant", iterations=50).apply(video)

# Effect on a sub-range via the `window` field:
video = Blur(
    mode="constant",
    iterations=50,
    window=TimeRange(start=0.0, stop=2.0),
).apply(video)

video = Zoom(zoom_factor=1.5, mode="in").apply(video)
video = ColorGrading(brightness=0.1, contrast=1.2, saturation=1.1).apply(video)
video = Vignette(strength=0.5, radius=0.8).apply(video)

start_region = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)
end_region = BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5)
video = KenBurns(start_region=start_region, end_region=end_region,
                 easing="ease_in_out").apply(video)

video = Fade(mode="in", duration=1.0).apply(video)
video = Fade(mode="out", duration=0.5).apply(video)
video = VolumeAdjust(volume=0.0, window=TimeRange(stop=2.0)).apply(video)  # mute first 2s
video = TextOverlay(text="Hello World", position=(0.5, 0.9),
                    font_size=48).apply(video)
```

Inside a `VideoEdit` plan, effects go in the segment's `operations`
list. The `window` field travels inline as a nested object:

```python
{"op": "blur_effect", "mode": "constant", "iterations": 50,
 "window": {"start": 0.0, "stop": 2.0}}
```

## Available Effects

Every effect except `add_subtitles` is streamable (compatible with
`VideoEdit.run_to_file()` for constant-memory processing).

| op | Class | Description |
|---|---|---|
| `blur_effect` | `Blur` | Gaussian blur, constant or ramping |
| `zoom_effect` | `Zoom` | Time-varying zoom in/out |
| `color_adjust` | `ColorGrading` | Brightness / contrast / saturation / temperature |
| `vignette` | `Vignette` | Radial darkening from the edges |
| `ken_burns` | `KenBurns` | Pan-and-zoom between two bounding boxes |
| `full_image_overlay` | `FullImageOverlay` | Composite a full-frame image |
| `fade` | `Fade` | Audio + video fade in/out/in_out |
| `volume_adjust` | `VolumeAdjust` | Audio-only effect |
| `text_overlay` | `TextOverlay` | Rendered text on top of frames |
| `add_subtitles` | `TranscriptionOverlay` | Requires `transcription` context |

## API Reference

### Effect

::: videopython.editing.Effect

### Blur

::: videopython.editing.Blur

### Zoom

::: videopython.editing.Zoom

### FullImageOverlay

::: videopython.editing.FullImageOverlay

### ColorGrading

::: videopython.editing.ColorGrading

### Vignette

::: videopython.editing.Vignette

### KenBurns

::: videopython.editing.KenBurns

### Fade

::: videopython.editing.Fade

### VolumeAdjust

::: videopython.editing.VolumeAdjust

### TextOverlay

::: videopython.editing.TextOverlay

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

# YouTube / experimental effects:
from videopython.editing import (
    Shake, PunchIn, Flash, ChromaticAberration, Glitch,
    FilmGrain, Sharpen, Pixelate, MirrorFlip, Kaleidoscope,
)

video = Shake(intensity_px=6, mode="rhythmic", frequency_hz=4).apply(video)
video = PunchIn(zoom_factor=1.5, attack_frames=3, release_frames=0).apply(video)
video = Flash(color=(255, 255, 255), peak_alpha=1.0,
              attack_frames=2, decay_frames=4,
              window=TimeRange(start=1.0, stop=1.3)).apply(video)
video = ChromaticAberration(shift_px=4, mode="radial").apply(video)
video = Glitch(intensity=0.4, slice_count=12, seed=42).apply(video)
video = FilmGrain(intensity=0.08, monochrome=True).apply(video)
video = Sharpen(amount=1.0, kernel_size=5).apply(video)
video = Pixelate(block_size=24,
                 region=BoundingBox(x=0.4, y=0.2, width=0.2, height=0.2)).apply(video)
video = MirrorFlip(mode="mirror_left").apply(video)
video = Kaleidoscope(segments=6).apply(video)
```

Inside a `VideoEdit` plan, effects go in the segment's `operations`
list. The `window` field travels inline as a nested object:

```python
{"op": "blur_effect", "mode": "constant", "iterations": 50,
 "window": {"start": 0.0, "stop": 2.0}}
```

## Available Effects

Every effect is streamable (compatible with `VideoEdit.run_to_file()` for
constant-memory processing). Context-requiring effects (`add_subtitles`)
stream too: pass `context=` to `run_to_file` and the runner re-bases it onto
each segment's local timeline.

| op | Class | Description |
|---|---|---|
| `blur_effect` | `Blur` | Gaussian blur, constant or ramping |
| `zoom_effect` | `Zoom` | Time-varying zoom in/out |
| `color_adjust` | `ColorGrading` | Brightness / contrast / saturation / temperature |
| `vignette` | `Vignette` | Radial darkening from the edges |
| `ken_burns` | `KenBurns` | Pan-and-zoom between two bounding boxes |
| `full_image_overlay` | `FullImageOverlay` | Composite a full-frame image |
| `image_overlay` | `ImageOverlay` | Scaled, positioned raster/SVG image (logo / watermark) |
| `fade` | `Fade` | Audio + video fade in/out/in_out |
| `volume_adjust` | `VolumeAdjust` | Audio-only effect |
| `text_overlay` | `TextOverlay` | Rendered text on top of frames |
| `add_subtitles` | `TranscriptionOverlay` | Word-level subtitles; requires `transcription` context |
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

## API Reference

### Effect

::: videopython.editing.Effect

### Blur

::: videopython.editing.Blur

### Zoom

::: videopython.editing.Zoom

### FullImageOverlay

::: videopython.editing.FullImageOverlay

### ImageOverlay

::: videopython.editing.ImageOverlay

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

### Shake

::: videopython.editing.Shake

### PunchIn

::: videopython.editing.PunchIn

### Flash

::: videopython.editing.Flash

### ChromaticAberration

::: videopython.editing.ChromaticAberration

### Glitch

::: videopython.editing.Glitch

### FilmGrain

::: videopython.editing.FilmGrain

### Sharpen

::: videopython.editing.Sharpen

### Pixelate

::: videopython.editing.Pixelate

### MirrorFlip

::: videopython.editing.MirrorFlip

### Kaleidoscope

::: videopython.editing.Kaleidoscope

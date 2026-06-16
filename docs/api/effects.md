# Effects

Effects are `Operation` subclasses that preserve video shape and frame
count. Each carries an optional `window: TimeRange | None` field that
limits the effect to a sub-range of the video. See
[Operations](operations.md) for the base contract.

## Usage

Effects are not applied to a `Video` directly. They run only through the
streaming engine: add the operation(s) to a `VideoEdit` and render with
`run_to_file`. Each effect carries an optional `window` that limits it to
a sub-range of the segment.

```python
from videopython.base import BoundingBox
from videopython.editing import (
    VideoEdit, SegmentConfig,
    Blur, Zoom, ColorGrading, Vignette, KenBurns,
    Fade, VolumeAdjust, TextOverlay, TimeRange,
)

edit = VideoEdit(segments=[SegmentConfig(source="input.mp4", start=0, end=5, operations=[
    # Effect across the full segment:
    Blur(mode="constant", iterations=50),
    # Effect on a sub-range via the `window` field:
    Blur(mode="constant", iterations=50, window=TimeRange(start=0.0, stop=2.0)),
])])
edit.run_to_file("output.mp4")
```

The constructors below produce the operation objects to drop into a
segment's `operations` list (as above) and render with `run_to_file`:

```python
video_op = Blur(mode="constant", iterations=50)
video_op = Blur(mode="constant", iterations=50, window=TimeRange(start=0.0, stop=2.0))
video_op = Zoom(zoom_factor=1.5, mode="in")
video_op = ColorGrading(brightness=0.1, contrast=1.2, saturation=1.1)
video_op = Vignette(strength=0.5, radius=0.8)

start_region = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)
end_region = BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5)
video_op = KenBurns(start_region=start_region, end_region=end_region,
                    easing="ease_in_out")

video_op = Fade(mode="in", duration=1.0)
video_op = Fade(mode="out", duration=0.5)
video_op = VolumeAdjust(volume=0.0, window=TimeRange(stop=2.0))  # mute first 2s
video_op = TextOverlay(text="Hello World", position=(0.5, 0.9), font_size=48)

# YouTube / experimental effects:
from videopython.editing import (
    Shake, PunchIn, Flash, ChromaticAberration, Glitch,
    FilmGrain, Sharpen, Pixelate, MirrorFlip, Kaleidoscope,
)

video_op = Shake(intensity_px=6, mode="rhythmic", frequency_hz=4)
video_op = PunchIn(zoom_factor=1.5, attack_frames=3, release_frames=0)
video_op = Flash(color=(255, 255, 255), peak_alpha=1.0,
                 attack_frames=2, decay_frames=4,
                 window=TimeRange(start=1.0, stop=1.3))
video_op = ChromaticAberration(shift_px=4, mode="radial")
video_op = Glitch(intensity=0.4, slice_count=12, seed=42)
video_op = FilmGrain(intensity=0.08, monochrome=True)
video_op = Sharpen(amount=1.0, kernel_size=5)
video_op = Pixelate(block_size=24,
                    region=BoundingBox(x=0.4, y=0.2, width=0.2, height=0.2))
video_op = MirrorFlip(mode="mirror_left")
video_op = Kaleidoscope(segments=6)
```

A `SegmentConfig`'s `operations` list also accepts the inline dict form;
the `window` field travels as a nested object:

```python
{"op": "blur_effect", "mode": "constant", "iterations": 50,
 "window": {"start": 0.0, "stop": 2.0}}
```

The subtitles effect (`add_subtitles`) requires a `transcription`
context, passed to the runner: `run_to_file(..., context={"transcription": ...})`.

## Available Effects

Every effect is streamable (compatible with `VideoEdit.run_to_file()` for
constant-memory processing). Context-requiring ops (`add_subtitles`) stream
too: pass `context=` to `run_to_file` and the runner re-bases it onto each
segment's local timeline. `add_subtitles` is special: it compiles to a
libass `subtitles=` ffmpeg filter at plan-compile time instead of running
per-frame Python.

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
| `add_subtitles` | `TranscriptionOverlay` | Word-level subtitles via libass; requires `transcription` context |
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

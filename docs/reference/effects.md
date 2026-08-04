# Effects

`Operation` subclasses that preserve shape and frame count, each carrying an optional
`window: TimeRange | None` limiting it to a sub-range of the segment. The base contract is
in [Operations](operations.md); the full effect table is
[there](operations.md#effects). AI-powered effects are in
[AI operations](ai/operations.md).

## Usage

Effects run only through the streaming engine — put them in a segment's `operations` and
render with `run_to_file`.

```python
from videopython.editing import VideoEdit, SegmentConfig, Blur, TimeRange

edit = VideoEdit(segments=[SegmentConfig(source="input.mp4", start=0, end=5, operations=[
    Blur(mode="constant", iterations=50),                                   # whole segment
    Blur(mode="constant", iterations=50, window=TimeRange(start=0.0, stop=2.0)),  # sub-range
])])
edit.run_to_file("output.mp4")
```

On the wire the window is a nested object:

```json
{"op": "blur_effect", "mode": "constant", "iterations": 50,
 "window": {"start": 0.0, "stop": 2.0}}
```

`add_subtitles` additionally needs a `transcription` in `run_to_file(context=...)` — see
[Transcription and subtitles](transcription.md).

## Constructor examples

```python
from videopython.base import BoundingBox
from videopython.editing import (
    Blur, ChromaticAberration, ColorGrading, Fade, FilmGrain, Flash, Glitch,
    Kaleidoscope, KenBurns, MirrorFlip, Pixelate, PunchIn, Shake, Sharpen,
    TextOverlay, TimeRange, Vignette, VolumeAdjust, Zoom,
)

Zoom(zoom_factor=1.5, mode="in")
ColorGrading(brightness=0.1, contrast=1.2, saturation=1.1)
Vignette(strength=0.5, radius=0.8)
KenBurns(start_region=BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5),
         end_region=BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5),
         easing="ease_in_out")

Fade(mode="in", duration=1.0)
VolumeAdjust(volume=0.0, window=TimeRange(stop=2.0))          # mute the first 2s
TextOverlay(text="Hello World", position=(0.5, 0.9), font_size=48)

Shake(intensity_px=6, mode="rhythmic", frequency_hz=4)
PunchIn(zoom_factor=1.5, attack_frames=3, release_frames=0)
Flash(color=(255, 255, 255), peak_alpha=1.0, attack_frames=2, decay_frames=4,
      window=TimeRange(start=1.0, stop=1.3))
ChromaticAberration(shift_px=4, mode="radial")
Glitch(intensity=0.4, slice_count=12, seed=42)
FilmGrain(intensity=0.08, monochrome=True)
Sharpen(amount=1.0, kernel_size=5)
Pixelate(block_size=24, region=BoundingBox(x=0.4, y=0.2, width=0.2, height=0.2))
MirrorFlip(mode="mirror_left")
Kaleidoscope(segments=6)
```

## How effects execute

Only `text_overlay` (drawtext) and `add_subtitles` (libass) compile to native FFmpeg
filters. Every other effect runs vectorised numpy/cv2 per frame; `Fade` and `VolumeAdjust`
additionally contribute an audio filter. The measurements behind that split are in
[the streaming engine](../explanation/streaming-engine.md#why-pixel-effects-are-not-ffmpeg-filters).

## Classes

::: videopython.editing.Effect

::: videopython.editing.Blur

::: videopython.editing.Zoom

::: videopython.editing.FullImageOverlay

::: videopython.editing.ImageOverlay

::: videopython.editing.ColorGrading

::: videopython.editing.Vignette

::: videopython.editing.KenBurns

::: videopython.editing.Fade

::: videopython.editing.VolumeAdjust

::: videopython.editing.TextOverlay

::: videopython.editing.Shake

::: videopython.editing.PunchIn

::: videopython.editing.Flash

::: videopython.editing.ChromaticAberration

::: videopython.editing.Glitch

::: videopython.editing.FilmGrain

::: videopython.editing.Sharpen

::: videopython.editing.Pixelate

::: videopython.editing.MirrorFlip

::: videopython.editing.Kaleidoscope

# Text & Transcription

Classes for handling transcriptions and rendering text overlays on video.

## Transcription Classes

### Transcription

::: videopython.base.Transcription

### TranscriptionSegment

::: videopython.base.TranscriptionSegment

### TranscriptionWord

::: videopython.base.TranscriptionWord

## Overlay Classes

### TranscriptionOverlay

Render transcriptions as subtitles with word-level highlighting:

```python
from videopython.base import Video
from videopython.editing import TranscriptionOverlay

video = Video.from_path("input.mp4")
# transcription = ... (from AudioToText or manually created)

overlay = TranscriptionOverlay(
    style="boxed",       # boxed | outline | clean | karaoke
    region="bottom",     # top | center | bottom
    font_scale=0.055,    # font height as a fraction of frame height
    # font_filename is optional; omit it (or pass None) to use the
    # bundled default font. Pass a path to use your own .ttf/.otf.
    font_filename=None,
)
video = overlay.apply(video, transcription)
```

Geometry is **resolution-independent** by default: `font_scale`/`region` are
fractions of the frame, so the same overlay renders correctly at any output
size and an upstream `face_crop`/`resize` cannot make it overflow. The
absolute fields (`font_size`, `position`, `box_width`, explicit colors, ...)
remain optional advanced overrides for back-compat -- leave them unset to
derive from the `style`/`region`/`font_scale` presets. Because the layout is
frame-relative and shared between the dry-run and the render,
`VideoEdit.validate()` rejects an un-fittable subtitle plan up front instead
of crashing mid-render.

::: videopython.editing.TranscriptionOverlay

### ImageText

Low-level text rendering on images:

::: videopython.base.ImageText

### TextAlign

::: videopython.base.TextAlign

### AnchorPoint

::: videopython.base.AnchorPoint

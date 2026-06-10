# Text & Transcription

Classes for handling transcriptions and burning subtitles onto video.

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
size. The absolute fields (`font_size`, `position`, `box_width`, explicit
colors, ...) remain optional advanced overrides -- leave them unset to derive
from the `style`/`region`/`font_scale` presets. Rendering is done by libass
(ffmpeg's `subtitles=` filter) from a compile-time ASS document: native
speed, and long cues wrap within the box instead of failing to fit.

::: videopython.editing.TranscriptionOverlay

### AnchorPoint

::: videopython.editing.AnchorPoint

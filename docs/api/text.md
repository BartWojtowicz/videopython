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
from videopython.base import Video, TranscriptionOverlay

video = Video.from_path("input.mp4")
# transcription = ... (from AudioToText or manually created)

overlay = TranscriptionOverlay(
    font_filename="path/to/font.ttf",
    font_size=40,
    highlight_color=(76, 175, 80),
)
video = overlay.apply(video, transcription)
```

::: videopython.base.TranscriptionOverlay

### ImageText

Low-level text rendering on images:

::: videopython.base.ImageText

### TextAlign

::: videopython.base.TextAlign

### AnchorPoint

::: videopython.base.AnchorPoint

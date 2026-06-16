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

Render transcriptions as subtitles with word-level highlighting. The
`add_subtitles` op (class `TranscriptionOverlay`) runs through the
streaming engine, so it executes inside a `VideoEdit` rather than against
a `Video` directly. It declares `requires=("transcription",)`; pass the
transcription via the `context` argument to `run_to_file`:

```python
from videopython.editing import VideoEdit

# transcription = ... (from AudioToText or manually created)

edit = VideoEdit.from_dict(
    {
        "segments": [
            {
                "source": "input.mp4",
                "start": 0.0,
                "end": 5.0,
                "operations": [
                    {
                        "op": "add_subtitles",
                        "style": "boxed",   # boxed | outline | clean | karaoke
                        "region": "bottom", # top | center | bottom
                        "font_scale": 0.055,  # font height as a fraction of frame height
                        # "font": "poppins-bold",  # optional bundled font; omit for default
                    }
                ],
            }
        ]
    }
)
edit.run_to_file("output.mp4", context={"transcription": transcription})
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

# Transcription and subtitles

Transcription data classes (`videopython.base`) and the subtitle-burning operation
(`videopython.editing`). Producing a transcription is
[`AudioToText`](ai/understanding.md#audiototext); a worked example is
[Tutorial 2](../tutorials/subtitles.md).

## Data classes

::: videopython.base.Transcription

::: videopython.base.TranscriptionSegment

::: videopython.base.TranscriptionWord

## TranscriptionOverlay

The `add_subtitles` operation. It declares `requires=("transcription",)`, so the
transcription is supplied through `run_to_file(context=...)` rather than a constructor
argument, and the runner re-bases it onto each segment's local timeline. Rendering goes
through libass (FFmpeg's `subtitles=` filter) from an ASS document compiled at plan time.

```python
from videopython.editing import VideoEdit

edit = VideoEdit.from_dict({
    "segments": [{
        "source": "input.mp4",
        "start": 0.0,
        "end": 5.0,
        "operations": [{
            "op": "add_subtitles",
            "style": "boxed",       # boxed | outline | clean | karaoke
            "region": "bottom",     # top | center | bottom
            "font_scale": 0.055,    # fraction of frame height
            # "font": "poppins-bold",
        }],
    }]
})
edit.run_to_file("output.mp4", context={"transcription": transcription})
```

### The recommended surface

| Field | Meaning |
|---|---|
| `style` | A named look bundling text/highlight colors, border and background. `boxed` reproduces the historical defaults. |
| `region` | Which vertical safe-area band the box sits in: `top`, `center`, `bottom`. |
| `font_scale` | Base font height as a fraction of frame height, so one plan renders correctly at 480p and 4K. Long cues wrap inside the box. |
| `font` | A bundled font by name: `anton`, `bebas-neue`, `lato-bold`, `poppins-bold` (full list: `videopython.base.fonts.FONT_NAMES`). `None` uses the bundled default. |

These are all relative or enumerated, which is why a stored plan round-trips and an LLM
can fill them in — see [LLM-first design](../explanation/llm-first-design.md#two-visibility-switches).

!!! note "Advanced overrides"
    The absolute fields (`font_size`, `text_color`, `highlight_color`,
    `background_color`, `background_padding`, `position`, `anchor`, `box_width`,
    `font_border_size`, `highlight_size_multiplier`, `max_words_per_cue`, `capitalize`)
    remain available as optional overrides. Leave them unset to derive everything from
    `style`/`region`/`font_scale`, or set one to pin it.

    Prefer *not* setting `font_size`: an absolute size chosen without knowing the final
    post-transform frame is exactly what overflows at render time. With the relative
    surface, `VideoEdit.validate()` rejects an un-fittable plan up front instead of
    crashing mid-render.

    `font_filename` is a raw TrueType path; it takes precedence over `font` and is
    `llm_hidden`, so it never appears in an LLM-facing schema.

::: videopython.editing.TranscriptionOverlay

::: videopython.editing.SubtitleStyle

::: videopython.editing.SubtitleRegion

::: videopython.editing.AnchorPoint

# Auto-Subtitles

Automatically transcribe speech and add word-level subtitles to any video.

## Goal

Take a video with speech, transcribe the audio using AI, and overlay synchronized subtitles with word-by-word highlighting.

## Full Example

```python
from videopython import Video
from videopython.ai import AudioToText
from videopython.editing import VideoEdit, SegmentConfig
from videopython.editing.transcription_overlay import TranscriptionOverlay

def add_subtitles(input_path: str, output_path: str):
    # Load the video and transcribe its audio.
    video = Video.from_path(input_path)
    transcriber = AudioToText()
    transcription = transcriber.transcribe(video)

    # Build a streaming plan with the add_subtitles op (TranscriptionOverlay).
    # Geometry is resolution-independent by default, so the same overlay works
    # at any output resolution. A single segment covers the whole source.
    edit = VideoEdit(segments=[SegmentConfig(
        source=input_path,
        start=0,
        end=video.total_seconds,
        operations=[
            TranscriptionOverlay(
                style="boxed",      # color/border/background/highlight preset
                region="bottom",    # top | center | bottom
                font_scale=0.055,   # fraction of frame height (auto-scales)
            ),
        ],
    )])

    # The op consumes the transcription via run_to_file's context; the runner
    # re-bases it onto the segment's local timeline.
    edit.run_to_file(output_path, context={"transcription": transcription})

add_subtitles("interview.mp4", "interview_subtitled.mp4")
```

## Step-by-Step Breakdown

### 1. Transcribe Audio

```python
transcriber = AudioToText()  # Local Whisper model
transcription = transcriber.transcribe(video)
```

The result is a `Transcription` carrying segments with word-level timestamps:

```python
for segment in transcription.segments:
    print(f"{segment.start:.2f}-{segment.end:.2f}: {segment.text}")
    for word in segment.words:
        print(f"  {word.word} ({word.start:.2f}-{word.end:.2f})")
```

See [Text & Transcription](../api/text.md) for the full data classes.

Model options:

- `tiny`, `base`, `small`, `medium`, `large`, `turbo`
- Enable diarization with `AudioToText(enable_diarization=True)` when needed.
- VAD-gated language detection runs by default (`enable_vad=True`); pass `enable_vad=False` to skip it.
- Bias Whisper toward brand or proper-noun spellings with `AudioToText(vocabulary=["Klarna", "Allegro"])`. See [Brand-name vocabulary biasing](../api/ai/understanding.md#brand-name-vocabulary-biasing).

### 2. Configure Subtitle Style

```python
overlay = TranscriptionOverlay(
    style="boxed",       # boxed | outline | clean | karaoke
    region="bottom",     # top | center | bottom
    font_scale=0.055,    # font height as a fraction of frame height
    font="poppins-bold", # bundled font by name, or None for the default
)
```

Key parameters (the recommended, resolution-independent surface):

- `style` -- A named look bundling text/highlight colors, border, and background so you express intent instead of a dozen numbers. `boxed` reproduces the historical defaults.
- `region` -- Which vertical safe-area band the box sits in: `top`, `center`, or `bottom`.
- `font_scale` -- Base font height as a fraction of frame height. Because it is relative, the same plan renders correctly whether the output is 480p or 4K. Long cues wrap within the subtitle box.
- `font` -- A bundled font by name: `anton`, `bebas-neue`, `lato-bold`, or `poppins-bold` (the full list is `videopython.base.fonts.FONT_NAMES`). LLM-friendly because the names are a fixed enum the model can pick from and stored plans round-trip on them. `None` uses the bundled default. For a custom TrueType file, use the advanced `font_filename` override instead (it takes precedence over `font`).

!!! note "Advanced overrides"
    The absolute fields (`font_size`, `text_color`, `highlight_color`,
    `background_color`, `position`, `anchor`, `box_width`, `margin`, ...)
    still exist as optional overrides for back-compat: leave them unset to
    derive from `style`/`region`/`font_scale`, or set one to pin it. Prefer
    *not* setting `font_size` -- an absolute size chosen without knowing the
    final (post-transform) frame is exactly what used to overflow at render
    time. With the relative surface, `VideoEdit.validate()` also rejects an
    un-fittable plan up front instead of crashing mid-render.

### 3. Run the Plan

The `TranscriptionOverlay` op (`add_subtitles`) declares `requires=("transcription",)`,
so the transcription is passed through `run_to_file`'s `context` rather than to
a constructor. The runner re-bases it onto each segment's local timeline before
the op compiles it to a libass `subtitles=` filter:

```python
edit = VideoEdit(segments=[SegmentConfig(
    source="interview.mp4",
    start=0,
    end=video.total_seconds,
    operations=[overlay],
)])
edit.run_to_file("interview_subtitled.mp4", context={"transcription": transcription})
```

The overlay renders each word at its exact timestamp, highlighting the current word being spoken.

## Customization

### Styling Options

```python
# Minimal outlined subtitles near the bottom (no background box)
overlay = TranscriptionOverlay(style="clean", region="bottom")

# Bigger karaoke-style subtitles centered for short-form vertical video
overlay = TranscriptionOverlay(style="karaoke", region="center", font_scale=0.07)

# Need a specific look? Override individual fields on top of a preset:
overlay = TranscriptionOverlay(style="outline", text_color=(255, 255, 0))
```

### Processing Long Videos

For long videos, transcription can take time. Consider processing in segments:

```python
from videopython import Video

# Process first 5 minutes
video = Video.from_path("long_video.mp4", start_second=0, end_second=300)
```

## Tips

- **Font size**: Prefer `font_scale` over absolute `font_size`. ~0.05-0.06 reads well across resolutions; bump toward 0.07-0.08 for mobile/short-form. It auto-scales, so you never re-tune it per output size.
- **Contrast**: The `boxed`/`karaoke` presets put text on a semi-transparent box that works on most backgrounds; `outline`/`clean` rely on a border instead.
- **Position**: `region="bottom"` is standard; `region="center"` suits short-form vertical video.
- **Languages**: Whisper supports 90+ languages. The API auto-detects language by default.

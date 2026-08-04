# 2. Subtitle a video with AI

In this tutorial you will transcribe a video's speech with a local Whisper model and burn
word-level subtitles onto it. Everything runs on your machine — no API keys — and it
works on CPU, though the first run downloads model weights.

You need `pip install "videopython[ai]"` ([Install](../install.md)) and a video with
speech in it, saved as `input.mp4`. [Tutorial 1](first-edit.md) is assumed.

## Step 1 — Transcribe

```python
from videopython.base import Video
from videopython.ai import AudioToText

video = Video.from_path("input.mp4")
transcription = AudioToText().transcribe(video)
```

The first call downloads the Whisper weights; later runs reuse them. `AudioToText` uses
the `turbo` model by default — large-v3 quality at roughly 8× the speed.

Look at what came back:

```python
for segment in transcription.segments[:3]:
    print(f"{segment.start:.2f}-{segment.end:.2f}: {segment.text}")
    for word in segment.words:
        print(f"    {word.word} ({word.start:.2f}-{word.end:.2f})")
```

A `Transcription` holds segments, and each segment holds words with their own
timestamps. That word-level timing is what makes the highlighted-word subtitle style
possible.

## Step 2 — Build a plan with a subtitle operation

Subtitles are an operation like any other — `add_subtitles`, the `TranscriptionOverlay`
class. One segment covering the whole video is enough:

```python
from videopython.editing import VideoEdit, SegmentConfig, TranscriptionOverlay

edit = VideoEdit(segments=[SegmentConfig(
    source="input.mp4",
    start=0,
    end=video.total_seconds,
    operations=[
        TranscriptionOverlay(
            style="boxed",     # boxed | outline | clean | karaoke
            region="bottom",   # top | center | bottom
            font_scale=0.055,  # font height as a fraction of the frame height
        ),
    ],
)])
```

Note what you did *not* specify: no font size in pixels, no x/y position. `font_scale`
and `region` are fractions of the frame, so the same plan renders correctly whether the
output is 480p or 4K.

## Step 3 — Hand the transcription to the runner

The operation does not take the transcription in its constructor. It declares
`requires = ("transcription",)`, and the runner supplies it at render time from a
`context` dict:

```python
edit.run_to_file("subtitled.mp4", context={"transcription": transcription})
```

Play `subtitled.mp4` — each word lights up as it is spoken.

Why the detour? Because a plan is data. A `Transcription` is a big object that would not
survive a round trip through JSON in any useful way, so operations that need bulky
side-channel input declare *what* they need, and the caller passes it in separately. The
runner also re-bases the timestamps: if your segment started at 30 s, the transcription's
source-absolute times are shifted onto the segment's local timeline for you.

## Step 4 — Restyle

Change the look by changing the preset, not by fiddling with a dozen numbers:

```python
# Minimal outlined text, no background box
TranscriptionOverlay(style="clean", region="bottom")

# Big karaoke text in the middle, for vertical short-form
TranscriptionOverlay(style="karaoke", region="center", font_scale=0.07)

# A preset with one field pinned
TranscriptionOverlay(style="outline", text_color=(255, 255, 0))

# A bundled font by name
TranscriptionOverlay(style="boxed", font="poppins-bold")
```

Bundled fonts are `anton`, `bebas-neue`, `lato-bold`, and `poppins-bold`
(`videopython.base.fonts.FONT_NAMES`). They are a fixed enum, which means an LLM can
pick one and a stored plan round-trips on it.

## Step 5 — Combine with everything else

Subtitles are an ordinary operation, so they compose. Here is the whole tutorial as one
vertical, subtitled, faded-in clip:

```python
from videopython.editing import VideoEdit, SegmentConfig, Resize, Fade, TranscriptionOverlay

edit = VideoEdit(segments=[SegmentConfig(
    source="input.mp4",
    start=0,
    end=min(30.0, video.total_seconds),
    operations=[
        Resize(height=1920),
        Fade(mode="in", duration=0.5),
        TranscriptionOverlay(style="karaoke", region="center", font_scale=0.07),
    ],
)])
edit.validate()
edit.run_to_file("clip.mp4", context={"transcription": transcription})
```

## What you learned

- `AudioToText` returns a `Transcription` with **word-level** timestamps, locally.
- Subtitles are the `add_subtitles` operation, styled with **relative** geometry
  (`style`, `region`, `font_scale`) so one plan works at any resolution.
- Operations that need bulky input **declare** it and receive it through
  `run_to_file(context=...)`, which also re-bases time-based values onto the segment.

## Next

- [How-to: dub a video into another language](../how-to/dubbing.md) — the same
  transcription, plus translation and voice cloning.
- [How-to: let a local LLM edit for you](../how-to/auto-editing.md) — hand over the
  editorial decisions.
- [Reference: transcription and subtitles](../reference/transcription.md) — every field
  on `TranscriptionOverlay`.
- [Reference: AI understanding](../reference/ai/understanding.md) — diarization,
  vocabulary biasing, and confidence fields on `AudioToText`.

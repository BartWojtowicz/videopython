# 1. Your first edit

In this tutorial you will turn a raw video file into a short, polished clip: a ten-second
cut, resized to vertical, fading in from black. Then you will add a second piece of
footage and let videopython stitch the two together.

You need videopython and FFmpeg installed ([Install](../install.md)) and one `.mp4` file.
Save it next to your script as `input.mp4`. Nothing here needs a GPU or the `[ai]` extra.

## Step 1 — Look at the source

Before editing anything, ask the file what it is. `VideoMetadata` reads the container
header only, so this is instant even on a huge file and no frames are loaded.

```python
from videopython.base import VideoMetadata

meta = VideoMetadata.from_path("input.mp4")
print(meta)
print(meta.width, meta.height, meta.fps, meta.total_seconds)
```

Note the duration — the next step cuts inside it.

## Step 2 — Describe the edit

videopython does not have `video.cut()` or `video.resize()` methods you call one after
another. Instead you *describe* the whole edit as a `VideoEdit` plan, and the library
executes it in one pass.

A plan is a list of **segments**. Each segment names a source, the time range to take
from it, and an ordered list of **operations** to apply to that range.

```python
from videopython.editing import VideoEdit

edit = VideoEdit.from_dict({
    "segments": [
        {
            "source": "input.mp4",
            "start": 0,
            "end": 10,
            "operations": [
                {"op": "resize", "width": 1080, "height": 1920},
                {"op": "fade", "mode": "in", "duration": 0.5},
            ],
        }
    ]
})
```

Two things worth noticing:

- **The cut is the segment's `start`/`end`**, not an operation. Trimming is what a
  segment *is*, so there is no `cut` op to add.
- **Nothing has happened yet.** `edit` is a data structure. It is a Pydantic model, so
  the dict above was type-checked as it was parsed, but no file has been opened.

## Step 3 — Validate before rendering

Ask the plan whether it would work:

```python
predicted = edit.validate()
print(predicted)          # the VideoMetadata your output will have
```

`validate()` reads the source's metadata, chains every operation's shape prediction
through the plan, and returns the metadata of the result — the resolution, fps and
duration you are going to get. It still loads no frames, so it is fast enough to run on
every user action in an application.

Try breaking it on purpose. Set `"end": 999999` and call `validate()` again: you get a
`PlanValidationError` naming the segment, the field, the value and the limit — before a
single frame was decoded.

## Step 4 — Render

```python
edit.run_to_file("output.mp4")
```

`run_to_file()` is the only execution engine. It streams FFmpeg decode → your operations
→ FFmpeg encode, one frame at a time, so peak memory is roughly constant no matter how
long the source is. Play `output.mp4`: a 10-second vertical clip that fades up from
black.

You can control the encode:

```python
edit.run_to_file("output.mp4", crf=20, preset="slow")   # higher quality, slower
```

`crf` is quality (0–51, lower is better, 23 is the default, 18 is visually lossless) and
`preset` is the speed/compression trade-off (`ultrafast` … `veryslow`).

## Step 5 — Add a second segment

Segments concatenate in order. Add a second one — say a later stretch of the same file —
and give the whole program a color adjustment with `post_operations`, which run once
against the concatenated result:

```python
edit = VideoEdit.from_dict({
    "segments": [
        {
            "source": "input.mp4",
            "start": 0,
            "end": 10,
            "operations": [
                {"op": "resize", "width": 1080, "height": 1920},
                {"op": "fade", "mode": "in", "duration": 0.5},
            ],
        },
        {
            "source": "input.mp4",
            "start": 20,
            "end": 28,
            "operations": [
                {"op": "resize", "width": 1080, "height": 1920},
            ],
        },
    ],
    "post_operations": [
        {"op": "color_adjust", "saturation": 1.15, "contrast": 1.05},
    ],
})

edit.validate()
edit.run_to_file("output.mp4")
```

Concatenation requires the segments to agree on fps and dimensions. Here you resized both
by hand, but the plan also does it for you: `match_to_lowest_fps` and
`match_to_lowest_resolution` are `true` by default, so mixing sources of different sizes
still renders.

## Step 6 — The same plan, as objects

The dict form is the JSON wire format — it is what you store in a database and what an
LLM emits. When you are writing Python by hand, the operation classes are often nicer,
and they are exactly the same models:

```python
from videopython.editing import VideoEdit, SegmentConfig, Resize, Fade

edit = VideoEdit(segments=[SegmentConfig(
    source="input.mp4",
    start=0,
    end=10,
    operations=[
        Resize(width=1080, height=1920),
        Fade(mode="in", duration=0.5),
    ],
)])
edit.run_to_file("output.mp4")
```

`Resize` keeps the aspect ratio if you give it only one dimension — `Resize(width=1280)`.

## What you learned

- An edit is a **plan**: segments, each with a time range and an ordered operation list,
  plus optional `post_operations` over the concatenated result.
- `validate()` is a **dry run over metadata** — it predicts the output and reports
  structured errors without decoding anything.
- `run_to_file()` **streams**, so memory does not grow with the length of the video.
- The dict form and the class form are the same models; the dict form *is* the JSON.

## Next

- [Tutorial 2: subtitle a video with AI](subtitles.md) — put a local Whisper model and a
  burned-in subtitle track on top of what you just learned.
- [How-to: build a vertical social clip](../how-to/social-clip.md) — the same ideas,
  aimed at a real deliverable.
- [Explanation: the streaming engine](../explanation/streaming-engine.md) — what
  `run_to_file()` actually does with your operations.
- [Reference: operations](../reference/operations.md) — every op you can put in that
  list.

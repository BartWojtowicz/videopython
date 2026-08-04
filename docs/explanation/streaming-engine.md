# The streaming engine

`VideoEdit.run_to_file()` is the only way to execute an operation. There is no
`operation.apply(video)`, and operations never run against a `Video` object. This page
explains what that buys and what it costs.

## One execution path, constant memory

The engine streams: FFmpeg decode → the operation chain → FFmpeg encode, one frame at a
time. Peak memory is roughly a frame plus the encoder's buffers, so an hour-long source
costs the same as a ten-second one.

The alternative — the design most editing libraries take — is to load frames into an array
and let each operation return a new array. That is friendlier for one-off scripting and
catastrophic for the actual use cases here: hour-long sources, LLM-authored plans running
on a server, batch jobs. Rather than maintain both paths and have every operation
implement two semantics that can silently diverge, videopython keeps one.

The cost is real and worth stating: you cannot apply an effect to a `Video` you built in
memory. Save it and put the file in a plan. Generated media (`TextToVideo`,
`ImageToVideo`) is the usual place this bites — see [Assemble a video from AI-generated
media](../how-to/ai-generated-video.md).

## Two kinds of operation

Every operation compiles into one of two things.

**Filters** compile to a native FFmpeg filter via `to_ffmpeg_filter(ctx)`. This is
reserved for what FFmpeg does that numpy cannot do, or cannot do as well:

- all transforms — `resize`, `crop`, `resample_fps`, the duration-changing `speed_change`
  and `freeze_frame`, the transcription-consuming `silence_removal`, and `face_crop`;
- the two text-rendering effects — `text_overlay` (drawtext) and `add_subtitles` (libass).

**Per-frame effects** are shape-preserving Python over each decoded frame, via
`streaming_init` + `process_frame`. **Every pixel effect** lives here: `blur`, `sharpen`,
`zoom`, `film_grain`, `chromatic_aberration`, `mirror_flip`, `vignette`, `color_adjust`,
`kaleidoscope`, `shake`, `flash`, `glitch`, `pixelate`, `ken_burns`, `punch_in`, and the
image overlays.

### Why pixel effects are not FFmpeg filters

This looks like the obvious optimization, so it was measured. Compiling pixel effects to
native filters bought at best ~1.1–1.4×, and in some cases (`gblur`) *lost*. The gain
that did exist came from skipping the rawvideo round-trip, not from faster math — the
effects are vectorised numpy/cv2 and are not the bottleneck.

So the engine reserves FFmpeg for geometry, timing and text rendering, and keeps the
per-frame path for its simplicity and its exact, testable output.

## What that means for a segment

- A segment whose operations are **all filters** renders in a **single FFmpeg
  invocation** — no rawvideo round-trip, no Python loop.
- A **single per-frame effect** switches that segment to decode → Python → encode.
  Filters ordered *before* the effect join the decode chain; filters *after* it join the
  encode chain. So `[fade, add_subtitles]` streams fine.
- **Duration-changing transforms fold their predicted metadata through the chain**, so
  later effect windows and the audio track follow the new timeline.
- `post_operations` run as a second pass over the assembled program, so any operation —
  filter, effect or transform — can apply to the whole concatenated timeline.

## Shapes that cannot stream

Streamability is decided structurally, from operation classes, their order, and the plan
shape — never from the media. Four shapes have no streaming strategy and are rejected with
`STREAMING_UNSUPPORTED` errors *before any decode*:

| Shape | Fix |
|---|---|
| A per-frame effect ordered after encode-stage filters | Move the effect earlier |
| A context-requiring op after a duration-changing transform | Move the op before the transform |
| `face_crop` behind per-frame effects | Put `face_crop` first |
| A time-based-context post-op on a multi-segment plan | Move the op into a segment — source-absolute context cannot re-base onto a concat |

Because the decision needs no media, `edit.streamability()` is safe to call as an
admission gate before a worker downloads anything:

```python
report = edit.streamability()
report.streamable        # bool
report.unstreamable      # offending ops, with reasons and reorder hints
report.errors()          # the same as structured PlanErrors
```

`edit.check(meta)` reports the same errors after the ordinary validity errors, and
`run_to_file()` raises them before it opens the source.

## Context data on a streaming path

Some operations need input a JSON plan cannot carry — a whole `Transcription`, for
instance. Those declare `requires: ClassVar[tuple[str, ...]]`, and the runner pulls the
matching keys out of `run_to_file(context=...)`.

Time-based values are **re-based onto the segment's local timeline** before delivery: a
transcription with source-absolute timestamps, used in a segment starting at 30 s, arrives
shifted so that the segment starts at zero. Without that, every context-consuming
operation would have to know its own offset, and every caller would have to pre-shift by
hand.

The resolved values reach the operation through `streaming_init` (per-frame effects) or
`FilterCtx.context` (filter-compiled ops), and through `predict_metadata` during
validation.

## Subtitles specifically

`add_subtitles` does not draw text per frame. At plan-compile time the transcription is
compiled to an ASS document, and FFmpeg's `subtitles=` filter burns it in with libass —
native speed, and long cues wrap inside the box rather than overflowing the frame. This
requires an FFmpeg built with libass, which every mainstream package provides.

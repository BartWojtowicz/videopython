# Glossary

Terms that mean something specific in videopython.

**catalog**
: A flat list of candidate scenes built from one or more `VideoAnalysis` results, each
with a stable id, exact bounds, caption, transcript and keyframe. What the auto-editing
planner and the MCP agent choose from. See
[AI auto-editing](reference/ai/auto-edit.md).

**check**
: The non-raising sibling of `validate`. Runs the same dry run but accumulates **every**
`PlanError` instead of stopping at the first, so a refine loop can fix everything in one
round. `[]` means valid.

**context**
: The dict passed to `run_to_file(context=...)`, carrying input a JSON plan cannot hold —
a `Transcription`, for instance. Operations declare what they need through `requires`, and
the runner re-bases time-based values onto each segment's local timeline.

**edit plan** / **`VideoEdit`**
: The complete description of an edit: segments, their operations, `post_operations`, and
the matching flags. A Pydantic model whose fields are the JSON wire format. See
[Edit plans](reference/video-edit.md).

**`EditPlan`**
: A *different*, smaller model used only by auto-editing and MCP: segments that reference
catalog scenes **by id** rather than by source and timestamps. `resolve_plan` turns one
into a `VideoEdit`.

**effect**
: An operation that preserves shape and frame count, and carries an optional `window`.
Blur, color grading, fades, overlays, subtitles.

**filter (op)**
: An operation that compiles into the FFmpeg filter chain via `to_ffmpeg_filter`. All
transforms plus the two text-rendering effects. Contrast with **frame effect**.

**frame effect**
: An operation implemented as Python over each decoded frame (`streaming_init` +
`process_frame`). Every pixel effect. See
[the streaming engine](explanation/streaming-engine.md).

**`llm_exposed` / `llm_hidden`**
: Schema visibility switches. An operation with `llm_exposed=False` is executable but
absent from LLM-facing schemas (typically because it needs a server-resolved path). A
field marked `llm_hidden` stays on the wire but is stripped from those schemas.

**normalize (dimensions)**
: Appending a per-segment `resize` so every segment shares a canvas and the concat
invariant holds by construction. `normalize_dimensions(meta, target)`.

**operation**
: Any editing primitive — a Pydantic model with an `op` discriminator, auto-registered on
definition. Transforms and effects are both operations. See
[Operations](reference/operations.md).

**`PlanError` / `PlanRepair`**
: Structured records. An error carries `code`, `location`, `field`, `value`, `limit`; a
repair carries `code`, `location`, `field`, `old`, `new`. Branch on `code`, never on
message text.

**post-operation**
: An operation in `post_operations`, applied to the concatenated program rather than to
one segment.

**profile**
: A named `VideoAnalysisConfig` preset selecting which analyzers run. `"full"` runs
everything; the MCP server's `"editing"` profile skips audio classification.

**repair**
: Clamping the mechanical, unambiguous violations of a plan — window overruns, negative
starts, out-of-range time parameters — and returning a changelog. Never invents intent.

**sampling**
: The per-scene frame budget for the scene VLM: `"low"`, `"medium"`, `"high"`. Orthogonal
to which model does the captioning.

**segment**
: One entry in a plan's `segments`: a source, a `start`/`end` range of it, and an ordered
operation list. **Trimming is what a segment is**, which is why there is no `cut`
operation.

**streamable / streamability**
: Whether a plan can run through the streaming engine. Decided structurally from operation
classes and their order — never from the media — so `edit.streamability()` can gate a job
before anything is downloaded.

**transform**
: An operation that may change dimensions, fps, duration, or frame count. Resize, crop,
fps resample, speed change, freeze frame, silence removal.

**window**
: An effect's `{"start": s, "stop": e}` sub-range within the segment. Frames outside it are
left untouched; either endpoint may be omitted.

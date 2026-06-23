# Automatic Editing

`AutoEditor` turns source videos plus a one-line brief into a finished cut, with
a **local** vision-language model (via [Ollama](https://ollama.com)) doing the
editorial selection. No cloud API keys, no timestamps to hand-author.

```python
from videopython.ai import AutoEditor, OllamaVisionLLM

# The planner is a local Ollama vision model. Pull it first: `ollama pull qwen3.6:27b`.
editor = AutoEditor(planner=OllamaVisionLLM(model="qwen3.6:27b"))

edit = editor.edit(
    ["clip_a.mp4", "clip_b.mp4", "clip_c.mp4"],
    brief="A punchy 15-second teaser; lead with the most dynamic shot.",
)
edit.run_to_file("teaser.mp4")
```

`editor.edit(...)` returns a validated [`VideoEdit`](../api/editing.md) — the same
plan object you can author by hand — so you can inspect, tweak, or re-render it.

## How it works

The model is bad at one thing (precise timestamps) and good at another (judging
shots). The design plays to that split:

1. **Analyze** — each source is run through [`VideoAnalyzer`](../api/ai/video_analysis.md):
   scene boundaries, a per-scene caption (SceneVLM), and a transcript.
2. **Catalog** — the scenes are projected into a flat catalog of candidate clips,
   each with a stable `scene_id`, its exact source/start/end, caption, transcript,
   and a **keyframe image**.
3. **Plan** — the planner sees the catalog (text + keyframes) and the brief, then
   authors an `EditPlan` that references scenes **by id** and adds operations
   (resize, crossfades, ...). It never invents timestamps.
4. **Resolve + run** — each `scene_id` maps back to its exact bounds, producing a
   real `VideoEdit`. The plan is repaired, dimension-normalized, and validated
   before rendering.

Because the model selects by id, its temporal imprecision never reaches the
render — the precise bounds come from scene detection, the editorial judgment
from the model.

## Refine loop

`AutoEditor` runs an internal repair/validate loop (`max_rounds`, default 3). If a
plan references an unknown id or violates a bound, the structured error is fed
back to the planner for another attempt. If no valid edit emerges within the
budget it raises `AutoEditError`.

```python
editor = AutoEditor(
    planner=OllamaVisionLLM(model="qwen3.6:27b"),
    max_rounds=3,
    normalize_target="largest",   # unify segment dimensions to the largest source
)
```

## Reusing analysis

Analysis is the expensive step. If you already have `VideoAnalysis` objects (or
want to analyze once and try several briefs), skip straight to planning:

```python
from videopython.ai import VideoAnalyzer

analyses = [VideoAnalyzer().analyze_path(p) for p in ["a.mp4", "b.mp4"]]
edit = editor.edit_from_analyses(analyses, brief="A calm, scenic 20s intro.")
```

## Choosing a planner model

The planner must be **vision-capable** (it is sent keyframes) **and** support
Ollama's structured-output `format` (schema-conditioned decoding). `qwen3.6:27b`
is the default (Apache-2.0). Some builds — e.g. certain MLX vision models — accept
images but ignore `format`, and planning fails; if a model returns prose instead
of JSON, pick another tag. `OllamaVisionLLM` is model-agnostic via `model=`, and
`StructuredVisionLLM` is the seam if you want to back the planner with something
other than Ollama.

## When to use which LLM mode

| You want… | Use |
|---|---|
| videopython to edit for you, fully local | **`AutoEditor`** (this page) |
| your own frontier model / harness to drive the tools | [MCP server](mcp.md) |
| to author/validate plans from your own LLM integration | [LLM Integration](llm-integration.md) |

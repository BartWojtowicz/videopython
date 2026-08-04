# Let a local LLM edit for you

`AutoEditor` turns source videos plus a one-line brief into a finished cut. A local
vision-language model served by [Ollama](https://ollama.com) makes the editorial
selection — no cloud keys, no timestamps to hand-author.

Needs `pip install "videopython[ai]"` and `ollama pull qwen3.6:27b`.

## Cut a teaser from three clips

```python
from videopython.ai import AutoEditor, OllamaVisionLLM

editor = AutoEditor(planner=OllamaVisionLLM(model="qwen3.6:27b"))

edit = editor.edit(
    ["clip_a.mp4", "clip_b.mp4", "clip_c.mp4"],
    brief="A punchy 15-second teaser; lead with the most dynamic shot.",
)
edit.run_to_file("teaser.mp4")
```

`edit(...)` returns a validated [`VideoEdit`](../reference/video-edit.md) — the same plan
object you would author by hand — so you can inspect it, tweak a segment, or re-render it
without re-running the model.

## What it does under the hood

1. **Analyze** — each source goes through
   [`VideoAnalyzer`](../reference/ai/video-analysis.md): scene boundaries, a per-scene
   caption, a transcript.
2. **Catalog** — scenes are flattened into candidate clips, each with a stable
   `scene_id`, exact bounds, caption, transcript, and a keyframe image.
3. **Plan** — the planner sees the catalog (text + keyframes) and your brief, and authors
   a plan that references scenes **by id** plus operations.
4. **Resolve and run** — ids map back to exact bounds, and the plan is repaired,
   dimension-normalized and validated before rendering.

The model never authors timestamps, so its temporal imprecision cannot reach the render.
Precise bounds come from scene detection; editorial judgment comes from the model. See
[LLM-first design](../explanation/llm-first-design.md#selection-by-id).

## Tune the refine loop

If a plan references an unknown id or violates a bound, the structured error goes back to
the planner. `AutoEditError` is raised if no valid edit emerges within the budget.

```python
editor = AutoEditor(
    planner=OllamaVisionLLM(model="qwen3.6:27b"),
    max_rounds=3,                  # default
    normalize_target="largest",    # unify segment dimensions to the largest source
)
```

## Analyze once, try several briefs

Analysis is the expensive step — reuse it:

```python
from videopython.ai import VideoAnalyzer

analyses = [VideoAnalyzer().analyze_path(p) for p in ["a.mp4", "b.mp4"]]

edit = editor.edit_from_analyses(analyses, brief="A calm, scenic 20s intro.")
alt  = editor.edit_from_analyses(analyses, brief="Fast, punchy, 8 seconds.")
```

## Choose a planner model

The planner must be **vision-capable** (it receives keyframes) **and** support Ollama's
structured-output `format`. `qwen3.6:27b` is the default and is Apache-2.0.

Some builds — certain MLX vision models, for instance — accept images but ignore
`format`; planning then fails because the model returns prose instead of JSON. If that
happens, pick another tag. `OllamaVisionLLM` is model-agnostic via `model=`, and
`StructuredVisionLLM` is the seam for backing the planner with something other than
Ollama.

## Drive the catalog yourself

For custom planning logic, the pieces are public:

```python
from videopython.ai import VideoAnalyzer, build_catalog
from videopython.ai.auto_edit import EditPlan, resolve_plan

analyses = [VideoAnalyzer().analyze_path("a.mp4")]
bundle = build_catalog(analyses)                 # bundle.catalog + bundle.keyframes

plan = EditPlan.model_validate({"segments": [{"scene_id": bundle.catalog.scenes[0].id}]})
edit = resolve_plan(plan, bundle.catalog)        # -> VideoEdit
edit.run_to_file("out.mp4")
```

Full signatures: [AI auto-editing reference](../reference/ai/auto-edit.md).

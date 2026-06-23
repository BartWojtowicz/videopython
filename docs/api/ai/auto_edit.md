# AI Auto-Editing

LLM-authored editing: build a scene catalog from one or more sources and let a
local vision-language model plan a `VideoEdit` from it. See the
[Automatic Editing guide](../../guides/auto-editing.md) for the end-to-end flow,
and the [MCP server](../../guides/mcp.md) for the agent-driven variant.

## Basic Usage

```python
from videopython.ai import AutoEditor, OllamaVisionLLM

editor = AutoEditor(planner=OllamaVisionLLM(model="qwen3.6:27b"))
edit = editor.edit(["a.mp4", "b.mp4"], brief="A 15s teaser, most dynamic shot first.")
edit.run_to_file("teaser.mp4")
```

## The catalog / by-id plan

`build_catalog` projects `VideoAnalysis` results into an `EditCatalog` of
candidate `CatalogScene`s (each with a stable `id`, exact bounds, caption,
transcript) plus one keyframe per scene. The planner authors an `EditPlan` whose
segments reference scenes by `scene_id`; `resolve_plan` maps those ids back to a
runnable `VideoEdit`. The model never authors timestamps.

```python
from videopython.ai import VideoAnalyzer, build_catalog
from videopython.ai.auto_edit import EditPlan, resolve_plan

analyses = [VideoAnalyzer().analyze_path("a.mp4")]
bundle = build_catalog(analyses)            # bundle.catalog + bundle.keyframes
plan = EditPlan.model_validate({"segments": [{"scene_id": bundle.catalog.scenes[0].id}]})
edit = resolve_plan(plan, bundle.catalog)   # -> VideoEdit
```

## Planner

::: videopython.ai.auto_edit.AutoEditor

::: videopython.ai.auto_edit.OllamaVisionLLM

::: videopython.ai.auto_edit.StructuredVisionLLM

## Catalog & plan

::: videopython.ai.auto_edit.build_catalog

::: videopython.ai.auto_edit.resolve_plan

::: videopython.ai.auto_edit.EditPlan

::: videopython.ai.auto_edit.EditCatalog

::: videopython.ai.auto_edit.CatalogScene

## Errors

::: videopython.ai.auto_edit.AutoEditError

::: videopython.ai.auto_edit.PlannerError

::: videopython.ai.auto_edit.UnknownSceneIdsError

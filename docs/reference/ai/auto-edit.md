# AI auto-editing

LLM-authored editing: build a scene catalog from one or more sources and let a local
vision-language model plan a `VideoEdit` from it. End-to-end usage is in
[Let a local LLM edit for you](../../how-to/auto-editing.md); the agent-driven variant is
the [MCP server](../mcp.md).

```python
from videopython.ai import AutoEditor, OllamaVisionLLM

editor = AutoEditor(planner=OllamaVisionLLM(model="qwen3.6:27b"))
edit = editor.edit(["a.mp4", "b.mp4"], brief="A 15s teaser, most dynamic shot first.")
edit.run_to_file("teaser.mp4")
```

## The catalog and the by-id plan

`build_catalog` projects `VideoAnalysis` results into an `EditCatalog` of candidate
`CatalogScene`s — each with a stable `id`, exact bounds, caption and transcript — plus one
keyframe per scene. The planner authors an `EditPlan` whose segments reference scenes by
`scene_id`, and `resolve_plan` maps those ids back to a runnable `VideoEdit`. The model
never authors timestamps
([why](../../explanation/llm-first-design.md#selection-by-id)).

```python
from videopython.ai import VideoAnalyzer, build_catalog
from videopython.ai.auto_edit import EditPlan, resolve_plan

analyses = [VideoAnalyzer().analyze_path("a.mp4")]
bundle = build_catalog(analyses)              # bundle.catalog + bundle.keyframes

plan = EditPlan.model_validate({"segments": [{"scene_id": bundle.catalog.scenes[0].id}]})
edit = resolve_plan(plan, bundle.catalog)     # -> VideoEdit
```

## Speech candidates

`build_catalog(..., mode="speech", speech=SpeechCandidateConfig(...))` selects timed
speech passages. `mode="visual"` is the default and requires `speech=None`.
`SpeechCandidateConfig` requires positive `min_duration` and `max_duration` in
seconds, with `max_duration >= min_duration`. `pause_duration` is positive and
defaults to 0.8 seconds.

Speech mode uses these deterministic rules:

- Each nonempty transcript segment must contain timed words. Otherwise that source
  contributes no candidates. Empty or absent transcriptions also contribute none.
- Words are sorted by start, then end. Times must be finite with
  `0 <= start <= end`; invalid times raise `ValueError`. Zero-duration words retain
  their supplied timestamp and text ownership. Their duration is not inferred.
- A span ends after `.`, `?`, `!`, `…`, `。`, `？`, or `！` (with closing quotes or
  brackets allowed), or before a gap of at least `pause_duration`. A boundary cannot
  pass through another overlapping word. An unfinished trailing span is discarded.
- Consecutive spans are combined until the minimum duration is met. If adding a
  span would exceed the maximum or cross a gap of at least `pause_duration`, the
  shorter pending group is discarded. A single
  span longer than the maximum is also discarded. Durations include shorter internal pauses.
- Candidates are chronological within each source and do not overlap. Source order
  follows the supplied analyses. Start/end are the first word start and the latest
  word end, without added padding. Source duration is required. Passages extending
  beyond it are omitted, not clamped through a timed word; supplied word records stay
  unchanged. An empty `catalog.scenes` is a valid outcome.

Speech mode does not assign a shot caption, shot type, or face flag. It can cross
visual cuts. Sentence punctuation and pauses are selection cues, not a topic or
meaning model: abbreviations, recognition errors, and an excerpt that starts
mid-sentence can produce poor editorial boundaries. Review the retrieved text and
rendered cuts. Supplied timing precision limits audio alignment; frame timing still
follows the renderer's source-frame contract.

`CatalogBundle.transcripts` maps every candidate ID to its full normalized transcript
text, in either mode. `CatalogScene.transcript` remains a short excerpt (280 characters
by default). Whitespace is normalized; source words and times are not changed.

### IDs and migration

Visual mode keeps the existing IDs. Speech IDs include a digest of the resolved
source path, transcript, and speech settings, followed by the candidate index.
Repeated builds with the same inputs produce the same IDs. Sources with identical
file stems in different directories have distinct speech IDs.

Rebuild the catalog and regenerate saved by-ID plans when changing mode, transcript,
source path, or speech settings. Old speech IDs will fail resolution rather than
select new ranges. The plan field remains `scene_id`. This is catalog identity,
not a media-content integrity check.

::: videopython.ai.auto_edit.SpeechCandidateConfig

## Planner

::: videopython.ai.auto_edit.AutoEditor

::: videopython.ai.auto_edit.OllamaVisionLLM

::: videopython.ai.auto_edit.StructuredVisionLLM

## Catalog and plan

::: videopython.ai.auto_edit.build_catalog

::: videopython.ai.auto_edit.resolve_plan

::: videopython.ai.auto_edit.EditPlan

::: videopython.ai.auto_edit.EditCatalog

::: videopython.ai.auto_edit.CatalogScene

## Errors

::: videopython.ai.auto_edit.AutoEditError

::: videopython.ai.auto_edit.PlannerError

::: videopython.ai.auto_edit.UnknownSceneIdsError

## Keyframe memory

Python catalogs with `keyframes=True` retain every selected full-resolution RGB frame.
Selected-frame extraction reads directly into one output array, preserving request
order and duplicate timestamps. An unreachable frame raises `VideoLoadError` rather
than returning a shorter array. The array needs approximately `N × width × height × 3`
bytes for `N` frames, plus decoder and process overhead. At 300 frames of 1920×1080,
that array alone is about 1.87 GB. Use `keyframes=False` for text-only selection, then
request a shortlist. This analysis allocation is separate from the renderer's bounded
frame buffers. MCP has its own [request and cache limits](../mcp.md#image-budget).

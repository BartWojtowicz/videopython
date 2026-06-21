# Auto-Editing: LLM-Authored Edits from Video Understanding

**Status: proposal — not implemented.** This is a design doc for discussion,
not documentation of existing behaviour. It builds on the execution and refine
machinery already documented in [`../guides/llm-integration.md`](../guides/llm-integration.md).

## Goal

Give a model enough understanding of one or more source videos that videopython
can author a `VideoEdit` plan automatically from a creative brief — e.g. "make a
20s highlight reel from these three clips", "cut this interview to the strongest
60 seconds", "assemble a montage of the outdoor shots". The output is a normal
`VideoEdit`, so everything downstream (`repair()` / `check()` /
`normalize_dimensions()` / `run_to_file()`) is unchanged. This capability lives
**inside videopython**, in the `videopython.ai` layer alongside the other AI
features.

## What already exists

- **`VideoEdit`** — Pydantic discriminated-union plan; `json_schema(strict=True)`
  for grammar-constrained decode; `llm_registry()` to expose only safe ops; the
  full refine loop (`check()` / `repair()` / `normalize_dimensions()`); streaming
  `run_to_file()`.
- **`VideoAnalysis`** — per-video structured understanding: TransNetV2 scene
  boundaries (`scene_index`, frame-exact `start_second`/`end_second`), Whisper
  word-level transcript, per-scene VLM `caption`/`subjects`/`shot_type`, audio
  events, face tracks. All JSON-serializable.
- The whole `videopython.ai` layer (understanding, generation, dubbing,
  translation) is **local-first**: it runs Whisper, Qwen VLM, Qwen3, diffusers,
  etc. locally behind granular extras, with **no external LLM API SDK vendored**.
  Auto-editing should match that character.

What is missing is the layer that turns understanding + a brief into a plan.

## The core design decision

A `VideoEdit` segment is defined by **precise continuous timestamps** against
each source (`start`/`end` in seconds). The temptation is to hand a vision model
the raw footage and ask for a plan directly (e.g. Gemini's native video
ingestion). The problem: vision LLMs are **bad at emitting precise timestamps**.
They sample video coarsely (Gemini ~1 fps) and return eyeballed cut points off by
a second or more. A cut landing mid-word or mid-action looks broken.

So separate the two jobs the model would otherwise conflate:

- **Where to cut** — precise, and a solved CV/ASR problem. Scene detection gives
  frame-exact boundaries; Whisper gives word-level spans for clean speech cuts.
  This is exactly what `VideoAnalysis` already computes. The model must **not**
  invent these numbers.
- **What to use and why** — editorial judgement: which moments, in what order,
  pacing, transitions, text, music. This is what the model is for.

`VideoAnalysis` is half the answer, not the wrong answer. Its weak link is the
**caption** layer (local Qwen, generic: "a woman walks down a street" drops
composition, motion, mood). Its **structural** layer (scenes + word timestamps)
is precisely what we keep.

## Recommended architecture

Turn continuous-time authoring into **discrete selection over a precisely-bounded
menu**, and let the planner see real pixels.

```
1. UNDERSTAND   (existing)          per source -> scenes + transcript + keyframe
2. CATALOG      (new)               flatten all sources -> one candidate list + keyframes
3. PLAN         (new, pluggable)    brief + catalog + keyframes + strict schema -> plan-by-id
4. RESOLVE      (new)               ids -> exact start/end -> VideoEdit
5. REFINE+RUN   (existing)          repair() / check() / run_to_file()
```

1. **Understand** — `VideoAnalysis` per source. Scenes carry stable IDs and
   frame-exact bounds; transcript carries word timestamps. Cache it; it is
   reusable and independent of any brief.
2. **Catalog** — flatten every source's scenes into one compact "edit catalog":
   candidate segments, each with a stable `id`, plus one representative keyframe
   per scene. The `id` is the contract between the model and the resolver.
3. **Plan** — a single multimodal call: brief + catalog JSON + keyframes
   (interleaved, labelled by `id`) + `VideoEdit.json_schema(strict=True)`. The
   model is constrained to **reference scenes by `id`** (or transcript spans),
   never to emit raw timestamps.
4. **Resolve** — map each referenced `id` back to its exact `source` + `start` +
   `end`, producing a real `VideoEdit`. The model's temporal imprecision never
   reaches the plan.
5. **Refine + run** — the existing loop: `repair()`, `normalize_dimensions()`,
   `check()`, re-prompt with structured errors if needed, then `run_to_file()`.

### Why this beats the two obvious alternatives

- **vs. `VideoAnalysis`-only (text captions into a text LLM):** the planner sees
  keyframes, so editorial quality is not capped by the local caption layer.
- **vs. vision-model-on-raw-video (the Gemini-direct idea):** timestamps are
  exact (from CV/ASR, not the model's guess); it is far cheaper (a few KB of JSON
  + a handful of keyframes vs. full-video frames across N clips); the
  understanding artifact is cacheable; it is robust to temporal imprecision; and
  it stays model-agnostic.

A third payoff matters specifically for a local-first library: framing the task
as **discrete selection over a catalog under a strict grammar** lowers the
reasoning bar enough that a *local* model can produce a reasonable plan. A
frontier API model raises quality, but the design does not require one.

## Capturing it inside videopython

The principle to preserve: videopython runs its AI locally and **vendors no
external LLM API SDK**. So the library owns the orchestration and the model is a
plug.

**1. videopython owns the loop.** An orchestrator (e.g. `AutoEditor` /
`auto_edit(...)`) inside `videopython.ai` runs understand -> catalog -> plan ->
resolve -> refine -> run. All the editorial logic, prompt construction, catalog
projection, id resolution, and the `check()`/`repair()` retry loop live in the
library. This is the "agent logic" — and it belongs here, because every piece it
coordinates (`VideoAnalysis`, `VideoEdit`) already does.

**2. The model is injected via a Protocol, not an SDK.** A small structured-vision
interface the library depends on, with zero third-party deps:

```python
class StructuredVisionLLM(Protocol):
    def generate_json(
        self,
        *,
        system: str,
        parts: list[Part],            # interleaved TextPart / ImagePart (keyframes)
        schema: dict[str, Any],       # EditPlan.json_schema(strict=True)
    ) -> dict[str, Any]: ...          # parsed JSON object (raise PlannerError if unusable)
```

The orchestrator builds `parts` (catalog text + labelled keyframes) and `schema`;
the backend just runs its model. A backend raises `PlannerError` for unusable
*output* (the editor retries); infrastructure failures propagate.

**3. Ship a local-first default: `OllamaVisionLLM`.** The default planner talks to
a local **Ollama** server, so the user runs any vision-capable model they have
pulled (`llama3.2-vision`, `qwen2.5vl`, `llava`, ...) — not a hard-coded model.
Ollama is local (no external API) and a *light* dependency (an HTTP client, no
torch), behind the `ollama` extra. Crucially, Ollama's structured-output
`format` takes the `EditPlan` schema, so the local path gets **grammar-constrained
JSON decode** — the constraint the in-process `transformers` path could not give.

**4. Local-only (committed).** Local inference only — Ollama is a local daemon, so
this keeps the "no external API" property. The `StructuredVisionLLM` Protocol is
the extension point (swap models via the `model=` arg, or a different backend, or
a stub for tests); the only shipped backend is `OllamaVisionLLM`. Wiring a
frontier API client through the same Protocol stays *possible* downstream, but
videopython ships nothing for it.

**Package placement.** Lives in a new subpackage `videopython.ai.auto_edit`,
mirroring `video_analysis` (orchestrator + models + focused helpers). The entry
point is a class, `AutoEditor`, exactly paralleling `VideoAnalyzer` — see
[Code structure](#code-structure) for the module breakdown.

```python
from videopython.ai import AutoEditor, OllamaVisionLLM

editor = AutoEditor(planner=OllamaVisionLLM(model="qwen2.5vl"))   # any pulled Ollama model
edit = editor.edit(
    ["clipA.mp4", "clipB.mp4", "clipC.mp4"],
    brief="20s upbeat highlight reel, punchy cuts, captions on speech",
)                                                                # returns a validated VideoEdit
edit.run_to_file("out.mp4")
```

## Delivery modes: programmatic API vs. MCP

Two ways to expose this capability. They are complementary, not competing, and
both ride on the **same core primitives** (catalog builder, id resolver, and the
existing schema / `validate` / `repair`).

**Programmatic (`auto_edit`)** — the local-only path above. One Python call runs
the whole loop with a local planner. Right for batch, embedded, and automated
use, and the path that honours local-only end to end.

**MCP server** — expose videopython's verbs as MCP tools (`analyze_video`,
`build_catalog`, `validate_edit`, `repair_edit`, `run_edit`) and the schema as a
resource, so an MCP-capable agent harness drives the edit. Keyframes ride back as
image content in tool results, so the agent literally sees the footage. This maps
1:1 onto the catalog design and is a *thin wrapper* over the same functions
`auto_edit` calls — not a second implementation.

The distinction that matters: **MCP is a transport, not a planner.** In the MCP
model the editorial reasoning lives in the *client's* model, not in videopython.
Two consequences:

- It is an excellent fit for *interactive, human-in-the-loop* editing ("analyze
  these, make a 20s reel, now punchier") — the agent calls tools iteratively and
  refines. Better than a one-shot `auto_edit` for exploration.
- It does **not** satisfy local-only by itself. If the driving agent is a
  frontier model (the natural case — "an agent like you"), the intelligence is an
  external model reached over MCP — the same external dependency we are deferring,
  just behind a different transport. It stays local only if the MCP client is
  itself a local model, which is not the common setup.

Recommendation: build the core primitives once (they serve both), ship the
local-only `auto_edit` now, and treat an MCP server as an **additive** delivery
mode to add when interactive-agent editing becomes a goal — with eyes open that
it relocates the planner outside the local boundary.

## Proposed data contract: the edit catalog

A compact, LLM-optimized projection of one or more `VideoAnalysis` results.
Sketch (final shape TBD):

```json
{
  "scenes": [
    {
      "id": "clipA#3",
      "source": "clipA.mp4",
      "start": 12.40,
      "end": 15.80,
      "duration": 3.40,
      "shot_type": "close-up",
      "caption": "Subject laughs and turns toward the camera",
      "transcript_excerpt": "...and that's when it clicked.",
      "has_speech": true,
      "has_faces": true
    }
  ]
}
```

Keyframes are passed alongside as images keyed by `id` (one per scene, downscaled
to ~512-768px). Word-level transcript spans are available for sub-scene cuts: the
model picks a sentence span and the resolver maps it to exact word `start`/`end`.

## Code structure

A new subpackage `videopython/ai/auto_edit/`, shaped like `video_analysis/`
(orchestrator + models + focused helpers). Everything is import-light (core deps
only); `local.py`'s one extra dependency (`ollama`, a light HTTP client) loads
lazily via `_optional.require(..., "ollama")`, per the AI layer's convention.

```
src/videopython/ai/auto_edit/
  __init__.py     # subpackage re-exports
  models.py       # CatalogScene, EditCatalog, CatalogBundle; PlanSegment, EditPlan   (pydantic, import-light)
  catalog.py      # build_catalog(analyses) -> CatalogBundle   (projection + keyframe extraction; base/ffmpeg only)
  resolve.py      # resolve_plan(plan, catalog) -> VideoEdit   (editing only)
  backend.py      # StructuredVisionLLM protocol + Part/TextPart/ImagePart + PlannerError   (typing/numpy only)
  editor.py       # AutoEditor: analyze -> catalog -> prompt -> plan -> resolve -> repair/check loop
  local.py        # OllamaVisionLLM backend over a local Ollama server   (light; require("ollama"))
```

One responsibility per module:

- **models.py** — the wire types. `EditCatalog`/`CatalogScene` are the LLM-facing
  candidate menu; `EditPlan`/`PlanSegment` are what the model returns — referencing
  scenes by stable `id` and reusing `OperationInput`/`TransitionSpec` from `editing`
  for op/transition richness. Keyframes stay *off* the pydantic models (numpy out
  of JSON); `CatalogBundle` is a small dataclass pairing the `EditCatalog` with
  `keyframes: dict[str, np.ndarray]`.
- **catalog.py** — pure projection from `VideoAnalysis[]`: flatten scenes, assign
  stable ids (`f"{source_stem}#{scene_index}"`), copy exact bounds + caption +
  shot_type + transcript excerpt, extract one midpoint keyframe per scene via
  `Video.extract_frames_at_times`. No torch.
- **resolve.py** — `resolve_plan` maps each `PlanSegment.scene_id` back to its
  `CatalogScene` (source + exact start/end) and builds a real `VideoEdit`; unknown
  ids are reported, not silently dropped. Also exposes the `EditPlan` strict schema
  by *reusing* `editing`'s strict-rewrite, not copying it.
- **backend.py** — the SDK-free seam: `StructuredVisionLLM.generate_json(system,
  parts, schema) -> dict`, where `parts` interleaves `TextPart`/`ImagePart`. The
  only thing a planner must implement; mirrors the existing `SpeechBackend`
  protocol pattern. `PlannerError` marks unusable model output (retriable) so the
  editor can distinguish it from infrastructure failures (which propagate).
- **editor.py** — `AutoEditor` owns all editorial logic: build the prompt (brief +
  per-scene labelled text + keyframes), pass `EditPlan`'s schema, call the backend,
  parse, resolve to a `VideoEdit`, then the `repair()` / `normalize_dimensions()` /
  `check()` refine loop with a bounded re-prompt on `PlannerError`, schema, or
  scene-id errors. Returns a validated `VideoEdit`.
- **local.py** — `OllamaVisionLLM` implements the backend against a local Ollama
  server: base64-encode keyframes, hand `EditPlan`'s schema to Ollama's
  structured-output `format` (grammar-constrained decode), parse the JSON. The
  model is the caller's choice via `model=`; nothing Qwen-specific.

Public symbols added to `videopython/ai/__init__.py`'s lazy `_exports` (all ->
`.auto_edit`): `AutoEditor`, `AutoEditError`, `OllamaVisionLLM`,
`StructuredVisionLLM`, `PlannerError`, `TextPart`, `ImagePart`, `EditCatalog`,
`EditPlan`, `build_catalog`. (`TextPart`/`ImagePart` are exported because
implementing a custom backend needs them.) The remaining subpackage symbols
(`CatalogScene`, `CatalogBundle`, `PlanSegment`, `resolve_plan`,
`UnknownSceneIdsError`) are reachable via `videopython.ai.auto_edit` but kept off
the top-level surface.

**Reuse, not duplication:**

- The op/transition schema helpers and the strict-rewrite come from `editing`
  (extracted into `editing/_schema.py`, shared with `VideoEdit.json_schema`);
  `EditPlan` only swaps `source`/`start`/`end` for `scene_id`.
- Validity comes from the existing `check()` / `repair()` /
  `normalize_dimensions()` loop — `auto_edit` adds no new validation.
- The Ollama backend needs no in-process model glue, so there is **no coupling to
  `SceneVLM`** and no `transformers`/`torch` dependency on the planner path.

A win of the Ollama backend: the strict `EditPlan` schema is handed to Ollama's
structured-output `format`, so the local path gets **grammar-constrained JSON
decode** — the very constraint the in-process `transformers` VLM path could not
provide. The `check()`/`repair()` loop still backstops cross-field rules a grammar
can't express, and the by-id catalog design keeps the reasoning bar low enough for
modest local models.

## Build order

Phase 1 — core primitives (this milestone), bottom-up so each layer is testable
without the one above it:

1. `models.py` + `catalog.py` — build a catalog from real `VideoAnalysis` output;
   unit-test ids, bounds, transcript slicing, keyframe extraction.
2. `resolve.py` — `EditPlan` -> `VideoEdit`, tested against `check()` /
   `run_to_file` with a hand-written plan (no model in the loop).
3. `backend.py` + a trivial **stub backend** — makes the whole `AutoEditor` loop
   CI-testable with a canned plan, no GPU, no model download.
4. `editor.py` — prompt building + refine loop, tested against the stub.
5. `local.py` — the `OllamaVisionLLM` backend (behind the `ollama` extra). Its
   logic (message building, `format`, JSON parse, `PlannerError`) is unit-tested
   with a fake client; the live path needs a running Ollama server + a pulled
   model, so it is validated by hand rather than in CI.

**Quality gate:** run `OllamaVisionLLM` on real footage with a capable pulled
model and judge edit quality / schema adherence (open question #1). The stub
backend means steps 1-4 land and are useful regardless of which model wins.

Phase 2 — MCP layer on top:

- New top-level `videopython/mcp/` (the server is cross-cutting — it exposes
  analysis + catalog + editing, not just `auto_edit`), behind a `[mcp]` extra and
  a `videopython-mcp` console script.
- Tools wrap Phase-1 functions 1:1: `analyze_video`, `build_catalog` (keyframes
  returned as image content), `validate_edit`, `repair_edit`, `run_edit`; the
  `VideoEdit` JSON schema as a resource.
- No new editing logic — the planner is the MCP client's model.

## Multi-video specifics

The "multiple videos" case is a montage / B-roll assembly problem: select and
order the best moments across N sources into one coherent edit. A flat catalog of
candidate scenes across all sources is the right shape — `VideoEdit` already
supports heterogeneous `segments` with per-segment ops, transitions, and a
program-level `music_bed`.

## Cost / context strategy

- One keyframe per scene + downscaling keeps most jobs within budget.
- For large libraries, two-stage: a cheap **text-only shortlist** over
  captions/transcripts, then a **multimodal detail** pass with keyframes only on
  the shortlist.
- The understanding artifact is cached and regenerated independently of the brief.

## Gap analysis (what's new)

- **Keyframes not persisted today.** `VideoAnalysis` extracts frames for the VLM
  then stores only descriptions. Re-extract scene-midpoint keyframes (cheap, via
  `extract_frames_at_times`) or extend the pipeline to keep representative refs.
- **Catalog builder** — `VideoAnalysis[]` -> catalog models + keyframe set.
- **Scene-id -> segment resolver** — plan-by-id (and transcript spans) ->
  concrete `VideoEdit`. Import-light.
- **`StructuredVisionLLM` Protocol + `PlannerError`** — SDK-free model interface.
- **`OllamaVisionLLM` backend** — local Ollama server, model-agnostic, behind the
  new `ollama` extra (`director` aggregates `vision`+`asr`+`ollama`).
- **`AutoEditor` orchestrator** — owns understand -> ... -> run and the refine loop.

## Decided

- **Local-only via Ollama**, one shipped backend (`OllamaVisionLLM`, model-agnostic
  through `model=`); `StructuredVisionLLM` is the seam for swapping it.
- **By-id plan** (`EditPlan` referencing `scene_id`), not full-`VideoEdit`
  emission — the model never authors timestamps.
- **Class entry `AutoEditor`**, mirroring `VideoAnalyzer`.
- **Extras**: a granular `ollama` extra for the planner; a `director` aggregate
  (`vision`+`asr`+`ollama`) for the full pipeline, mirroring how `dub` aggregates.
- **MCP is Phase 2**, additive, with the planner outside the local boundary.

## Open questions

1. **Local planner quality (gating risk).** Which pulled Ollama vision model is
   good enough for editorial selection + schema adherence? Resolve by running on
   real footage (the quality gate). The catalog/by-id design lowers the bar, but
   model choice still dominates.
2. **Sub-scene granularity.** How far to lean on transcript spans vs. whole
   scenes as the atomic unit.
3. **Keyframe budget.** One per scene vs. multiple for long/active scenes; the
   downscale target before base64-encoding for Ollama.

## References

- [`../guides/llm-integration.md`](../guides/llm-integration.md) — schema,
  strict-mode grammar, and the `check()` / `repair()` / `normalize_dimensions()`
  refine loop this design reuses verbatim.

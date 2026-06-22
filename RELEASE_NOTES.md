# Release Notes

## 0.52.0

`videopython.ai` technical-debt cleanup (PR-2 of 2): the taxonomy moves, file/class
splits, and remaining cleanups from the `TODO.md` audit. Reorganization, not feature
work — no AI capability changes. Follows 0.51.0 (PR-1).

### Breaking

- **Module moves** (dubbing-only concerns relocated into `dubbing/`, enforcing a
  leaf-capabilities-vs-orchestrators layering). Import paths change:
  - `videopython.ai.generation.translation` -> `videopython.ai.dubbing.translation`
  - `videopython.ai.generation._tts_backend` -> `videopython.ai.dubbing._tts_backend`
  - `videopython.ai.understanding.separation` -> `videopython.ai.dubbing.separation`
    (fixes a leaf->orchestrator import inversion; `_merge_regions` is now public
    `merge_regions`). `SeparatedAudio` already lived in `dubbing.models`.
- **`FaceTracker` split into two focused classes** (the name is removed). It carried
  two unrelated algorithms behind one ~10-arg constructor:
  - `FaceSmoothingTracker` — single-subject EMA path (`detect_and_track` /
    `track_video`), used by `FaceTrackingCrop`.
  - `FaceShotTracker` — per-shot IoU multi-track (`track_shot`), used by analysis.
  Both share a small internal `_FaceDetector` and are context-managed.
- **`AudioClassifier` moved** from `understanding/audio.py` to
  `understanding/classification.py` (the `videopython.ai` / `videopython.ai.understanding`
  re-exports are unchanged, so `from videopython.ai import AudioClassifier` still works;
  only the deep `understanding.audio` import path changed).
- **`video_analysis.stages` renamed to `video_analysis.detectors`** (it is detector
  adapters + timing helpers, not a plugin framework).

### Added / Internal

- `videopython.ai.AiError` — common base for the AI error types (`OllamaError`,
  `PlannerError`, `AutoEditError`, `GarbageTranscriptError`, `RemuxError`,
  `UnknownSceneIdsError`), so callers can `except AiError`. Each keeps its semantic
  builtin base (`RuntimeError` / `ValueError`).
- `videopython.ai.keyframe` — public `downscale_keyframe` / `encode_png_b64` /
  `keyframe_to_png_b64`; the MCP server no longer imports `ai._ollama` privates.
- `dubbing/audio_ops.py` merges the former `expressiveness` + `loudness` leaf modules;
  `TimingAdjustment` moved from `timing.py` to `models.py`.
- `DubbingConfig.from_args()` dedups the config-or-kwargs guard; `video_analysis`
  source-metadata parsing (ffmpeg tags / ISO-6709 geo / creation-time) extracted into
  `video_analysis/source_metadata.py` with a `try_init` best-effort helper.

### Deferred (tracked in `TODO.md`)

Two pure-internal Tier-5 items are left for a focused follow-up (no API/behavior
change): deduplicating the Demucs block shared by the dubbing pipeline's `process()`
and `revoice()`, and dropping the speculative single-entry `SUPPORTED_MODELS` /
`STEM_NAMES_6S` config.

## 0.51.0

`videopython.ai` technical-debt cleanup (PR-1 of 2): a correctness fix plus the
removal of abstraction residue left behind when the granular extras (0.43.0) were
collapsed into one `[ai]` (0.48.0) and the alternative translator/TTS backends
were deleted (0.49.0). Net ~430 fewer lines; no capability removed. See `TODO.md`
for the full audit and the PR-2 (taxonomy) follow-up.

### Fixed

- **AI plan-ops are now always in the planner / MCP op schema.** `FaceTrackingCrop`
  (`face_crop`) and `ObjectDetectionOverlay` (`object_detection_overlay`) register
  only as an import side-effect, and `ai/__init__` re-exports them lazily — so in a
  fresh process that built `EditPlan.json_schema()` / `Operation.llm_registry()`
  (the auto-edit planner, the MCP `edit_plan_schema` resource) before touching them,
  both ops were silently absent and `Operation.get("face_crop")` raised. A new
  `videopython.ai.ops` self-registration module, imported by `ai.auto_edit`, fixes
  this; regression-tested from a clean interpreter.

### Breaking

- **Planner seam mirrors the Ollama client.** `StructuredVisionLLM.generate_json`
  now takes `(system, text, images, schema)` instead of a `parts` list; the
  `TextPart` / `ImagePart` / `Part` types are removed (the planner flattened them
  straight back to text+images, so they added only indirection). `OllamaVisionLLM`
  and the `StructuredVisionLLM` protocol stay. Removed from `videopython.ai` and
  `videopython.ai.auto_edit` exports: `TextPart`, `ImagePart`, `Part`.
- **Dead API removed**: `UnsupportedLanguageError` (never raised),
  `TranslationBackend` (single implementor; translation is hardwired to
  `OllamaTranslator`), `OllamaTranslator.supports()`, `DubbingConfig.translator`
  (the `"auto"`/`"ollama"` values were behaviorally identical — use
  `translator_model` / `translator_host`), `TimingSynchronizer.check_overlaps` +
  its `gap_threshold` arg, `VideoAnalysisConfig.rich_understanding_preset()` (use
  `for_profile("full")`), and the path-based `remux.replace_audio_stream` (the
  in-memory `replace_audio_stream_from_audio` is the only caller path).
- **`ai._optional.require()` signature**: dropped the always-`"ai"` `extra`
  positional — now `require(module, *, feature=...)` with a hardcoded `[ai]` install
  hint. Internal helper; call sites updated.

### Internal

- New `ai.understanding._yolo.YoloDetector` base unifies the duplicated YOLO wrapper
  behind `ObjectDetector` and the face detector (lazy load, device resolution,
  `detect`/`detect_batch`).
- `ManagedPredictor` now provides a default `unload()` driven by a `_model_attrs`
  class attribute, replacing ~10 near-identical hand-written `unload()` bodies.
  `ObjectDetector` and `FaceTracker` are now context-managed and expose `unload()`,
  closing a VRAM-release gap.
- Docstrings/comments that still described the removed granular extras
  (`[asr]`/`[vision]`/`[tts]`/`[dub]`/...) were rewritten for the single `[ai]` extra.

## 0.50.1

MCP ergonomics at scale. MCP keyframes are downscaled (longest side <= 768px)
before base64-encoding, shrinking the build_catalog / scene_keyframes payload by
roughly 10x (SceneVLM captioning and the local planner keep full-resolution
frames). `build_catalog` now always returns the full catalog text
but caps inlined keyframes; the rest are fetched on demand with a new
`scene_keyframes(scene_ids)` tool, and an explicit note lists any omitted scene
ids so the agent never assumes it saw every frame. `analyze_video` gains a
`profile="editing"` option that skips audio classification for a faster,
catalog-only analysis on long sources.

## 0.50.0

MCP server. A new `videopython/mcp/` exposes the auto-edit pipeline as Model
Context Protocol tools, so an MCP-capable agent (its own model is the planner)
drives editing: `analyze_video`, `build_catalog` (returns the candidate scenes
as JSON plus one keyframe image each, so the model sees the footage),
`validate_edit`, `repair_edit`, `run_edit`, and the edit-plan JSON schema as a
resource. The agent authors an `EditPlan` by scene id; the server caches analyses
+ the catalog so payloads stay small and reuses the existing
resolve/check/repair/run machinery (no new editing logic).

Install with `pip install 'videopython[ai,mcp]'` and run `videopython-mcp`
(stdio); register it with any MCP client. New `[mcp]` extra (`mcp>=1.27,<2`).

## 0.49.0

Local AI models consolidated onto Ollama. Scene captioning (`SceneVLM`) and
dubbing translation now run through a local **Ollama** server instead of
in-process `transformers` / `llama-cpp-python`, behind one shared
structured-generation client (`ai/_ollama.py`). This removes the heaviest,
build-painful dependencies and gives both paths grammar-constrained JSON decode.

### Breaking

- **`SceneVLM`** talks to Ollama now. Its constructor takes `model` (an Ollama
  tag you have pulled, e.g. `gemma3:27b`), `host`, and `options` instead of
  `model_size` / `device` / transformers knobs. A running Ollama server with a
  vision model that supports structured output is required.
- **Translation** is a single `OllamaTranslator` (`generation/translation.py`).
  `MarianTranslator` and the Qwen3 (`llama-cpp`) backend are gone, along with the
  Marian-vs-Qwen3 auto-resolver. `DubbingConfig.translator` is now
  `"auto"`/`"ollama"` (the old `"marian"`/`"qwen3"` values are removed); choose
  the model via `translator_model` / `translator_host`.
- **Dependencies dropped from `[ai]`**: `qwen-vl-utils`, `llama-cpp-python`,
  `sentencepiece`. `transformers` stays (AudioClassifier + MusicGen still use it).

### Why

`llama-cpp-python` was the worst dependency to build/resolve, and the in-process
`transformers` VLM path could not grammar-constrain its JSON. Routing both through
Ollama deletes those deps, unifies the LLM/VLM calls behind one client, and makes
the JSON reliable. Models that don't support Ollama's structured-output `format`
(e.g. some MLX builds) won't work; `gemma3:27b` is verified.

## 0.48.0

LLM-authored editing: a new `videopython.ai.auto_edit` layer turns one or more
analyzed videos plus a creative brief into a runnable `VideoEdit`. It builds a
scene **catalog** from `VideoAnalysis` (stable per-scene ids, exact CV-derived
bounds, a keyframe, caption, transcript), asks a vision model to select and order
scenes **by id** — never authoring timestamps — and resolves that plan back to
exact source/start/end, so the model's temporal imprecision can't reach the cut
points. Validity stays the existing `repair()` / `normalize_dimensions()` /
`check()` loop; `auto_edit` adds no new validation.

### Local-first, model-agnostic planner via Ollama

The default planner, `OllamaVisionLLM`, talks to a local Ollama server, so you run
a pulled vision model with no external API. The `EditPlan` schema is handed to
Ollama's structured-output `format`, giving the local path grammar-constrained JSON
decode. The planner is injected through the `StructuredVisionLLM` protocol, so
swapping models or backends needs no SDK.

The chosen model must support **both** vision and Ollama's structured-output
`format` — not all do. `gemma3:27b` is verified working; some builds (e.g. certain
MLX ones) ignore schema conditioning and fail.

```python
from videopython.ai import AutoEditor, OllamaVisionLLM

editor = AutoEditor(planner=OllamaVisionLLM(model="qwen2.5vl"))
edit = editor.edit(["a.mp4", "b.mp4"], brief="20s highlight reel, captions on speech")
edit.run_to_file("out.mp4")
```

Install with `pip install 'videopython[ai]'`. The schema-building helpers behind
`VideoEdit.json_schema` moved to a shared `editing/_schema.py`, reused by
`EditPlan.json_schema` (no behavior change).

### Single `[ai]` extra (breaking)

The per-capability extras (`[asr]`, `[vision]`, `[separation]`, `[translation]`,
`[tts]`, `[generation]`, `[dub]`) are collapsed into one `[ai]` extra that installs
every AI capability. Heavy ML deps still load lazily at first use, so a plain
`import videopython` stays light. Consumers pinning a granular extra (e.g.
`videopython[vision]`) must switch to `videopython[ai]`.

## 0.47.0

Pixel effects go back to numpy. After 0.46.0 migrated nine effects to native
ffmpeg filters, benchmarks showed the migrations were not worth it: ffmpeg only
beat the vectorised numpy/cv2 `process_frame` by ~1.1–1.4x (and that gain was
from skipping the rawvideo round-trip, not faster compute), some were *slower*
(`gblur`), and the `geq`-based ones were catastrophic. So the engine now reserves
ffmpeg filters for what numpy can't do well — geometry/timing transforms and text
rendering — and runs every pixel effect per-frame. No wire-format change; every
op remains streamable.

### Every pixel effect runs per-frame numpy again

The benchmark (1080×1920, ~390 frames) was decisive: decode + encode dominate
every render (~6s, paid by both paths), pure numpy compute is tiny (<1s for most
effects), and the rawvideo round-trip a per-frame effect adds is only ~0.8s. So
the per-effect filter win was marginal at best:

| effect | filter (ffmpeg) | frame (numpy) | filter speedup |
|---|---|---|---|
| `blur` (gblur) | 12–15s | 7.3s | **0.6x (slower)** |
| `sharpen` (unsharp) | 7.6s | 8.1s | 1.07x |
| `zoom` (zoompan) | 6.5s | 7.6s | 1.17x |
| `film_grain` (noise) | 27.8s | 31.9s | 1.15x |
| `chromatic` h/v (rgbashift) | 6.4s | 8.7s | 1.36x |
| `mirror` h/v (hflip) | 6.1s | 8.0s | 1.31x |

Every pixel effect is now a per-frame `process_frame` numpy/cv2 implementation:
`blur`, `sharpen`, `zoom`, `film_grain`, `chromatic_aberration`, `mirror_flip`,
`vignette`, `color_adjust`, `kaleidoscope`, `shake`, `flash`, `glitch`,
`pixelate`, `ken_burns`, `punch_in`, and the image overlays. Output is bit-identical
to the pre-0.46.0 numpy path (no re-baseline). The only effects that still compile
to ffmpeg filters are the two with no good numpy form: `text_overlay` (drawtext)
and `add_subtitles` (libass). Transforms (`resize`, `crop`, `resample_fps`,
`speed_change`, `freeze_frame`, `silence_removal`, `face_crop`) are unchanged —
they stay native filters and an all-transform segment still renders in a single
ffmpeg invocation.

Relative to 0.46.0, the streamability *class* of every migrated pixel effect moves
from `filter` back to `frame_effect` (all remain streamable); consumers gating on
`edit.streamability()` should expect that reclassification.

### Removed the `filter_needs_rgb24` mechanism (internal)

Only the `geq`/`rgbashift`/`gblur` filters read RGB and needed the decode chain
converted to `rgb24` first; with every pixel effect back on numpy, no effect sets
`filter_needs_rgb24`. The `Effect.filter_needs_rgb24` ClassVar and the plan
builder's `format=rgb24` prepend are removed. (The single-invocation fast path's
own rgb24 round-trip — which mirrors the framewise encoder for byte-parity — is
unrelated and stays.)

### Removed the `streamable` ClassVar (internal)

Every registered op is streamable, so the `streamable` flag was redundant. It is
replaced by `Operation.streams()` — a transform streams iff it overrides
`to_ffmpeg_filter`; an effect iff it overrides `process_frame` or sets
`compiles_to_filter`. The classifier and plan builder now decide structurally,
so the report and the builder cannot disagree (the residual "filter compiles to
None at this position" case is still caught by the runtime drift guard). A custom
`Operation`/`Effect` subclass no longer declares `streamable`; it streams by
implementing one of those methods.

### Removed dead post-op-fold machinery (internal)

`EffectScheduleEntry.index_offset` / `total_effect_frames` were only set by the
per-segment post-op fold removed in 0.46.0 (post-ops now run as a pass over the
assembled program), so they always took their defaults. Dropped, with the
framewise loop and a stale docstring simplified accordingly.

## 0.46.0

A faster, single-mechanism editing engine. Filter-only segments render in one
ffmpeg invocation, nine effects became native ffmpeg filters, and
`post_operations` run as a real pass over the assembled program. The net effect
is fewer rejected plans and less per-frame Python.

Breaking: the streamability *classification* of several plan shapes changed (more
shapes stream; nine effects move from `frame_effect` to `filter`), and
`cut`/`cut_frames` are no longer chain ops. Consumers that gate on
`edit.streamability()` / `edit.check()` should expect those reclassifications. No
public method signature changed.

### Single-invocation fast path for filter-only segments

A segment with no scheduled per-frame effect now renders in ONE ffmpeg process
(`stream_segment_filtergraph`) instead of decode -> rawvideo pipe -> Python loop
-> rawvideo pipe -> encode. Segments carrying frame effects still use the
per-frame pipeline; both emit identical encode params so their outputs
concat-copy together. Also fixed a latent `Crop` center-mode bug: the compiled
`crop=` filter now floors to even dimensions and re-centers, matching
`predict_metadata` (libx264/yuv420p reject odd dimensions).

### Nine effects are now native ffmpeg filters

`vignette`, `mirror_flip`, `chromatic_aberration`, `kaleidoscope`, `color_adjust`
(geq/hflip/rgbashift), `sharpen` (unsharp), `text_overlay` (drawtext), `zoom`
(zoompan), and monochrome `film_grain` (noise) compile to native filters in their
common forms and take the fast path. Output is a faithful visual twin
re-baselined to ffmpeg (verified in-pipeline against the numpy `process_frame`
twins), not pixel-identical -- `film_grain` in particular is a different but
statistically-matched grain. Windowed/large-kernel/per-channel/no-font variants,
and the remaining effects (blur, glitch, pixelate, ken_burns, punch_in, shake,
flash, the overlays), stay on the per-frame path. `process_frame` is retained on
every migrated effect.

A new `Effect.filter_needs_rgb24: ClassVar[bool]` declares that a decode-stage
`to_ffmpeg_filter` reads RGB channels (geq r/g/b, rgbashift); the builder
prepends `format=rgb24` to the decode chain so it does not read the native yuv
stream. Default `False` (geometry/overlay filters like `hflip`/`drawtext`/
`subtitles=` are format-agnostic).

### `post_operations` run as a pass over the assembled program

`post_operations` no longer fold into per-segment frame schedules. The assembled
program is treated as one synthetic segment and run back through the same engine,
so previously-rejected shapes now stream: transforms as post-ops; filter-class
effects as post-ops; `post_operations` combined with segment transitions; and
audio-coupled post-ops (`fade`/`volume_adjust`) over a multi-segment program (the
envelope now spans the whole timeline instead of restarting at each boundary).
The only post-op shape still rejected is a context-requiring (time-based) post-op
on a multi-segment plan -- `POST_OP_REQUIRES_CONTEXT` from `check()` (source-
absolute context cannot re-base onto a concat). Single-segment context post-ops
stream fine.

### Only streamable ops in the registry

`cut`/`cut_frames` were the only non-streamable ops; trimming is the segment's own
`start`/`end` mechanism, so they are now `internal_only` -- constructed directly
by the engine but kept OUT of the op registry and the LLM schema. They can no
longer appear in a plan's `operations` list (rejected at parse). Every registered
op is now streamable -- it compiles to an ffmpeg filter or is a per-frame effect.
A new `Operation.internal_only: ClassVar[bool]` gates registration.

### Unified segment compatibility

The min-fps/min-resolution concat policy now lives in one resolver
(`_resolve_matching_target`), shared by the prediction and execution passes so
they cannot drift, and `normalize_dimensions` gained a `"match"` target that
materializes the same policy as explicit resize ops. Additive.

## 0.45.0

Breaking: the "fallback" vocabulary is retired. Since the in-memory path was
removed in 0.44.0 there is nothing to fall back *to* -- an op that cannot stream
is a hard rejection, not a fallback -- so the names now say so.

### Renamed `STREAMING_FALLBACK` -> `STREAMING_UNSUPPORTED`

`PlanErrorCode.STREAMING_FALLBACK` (value `"streaming_fallback"`) is now
`PlanErrorCode.STREAMING_UNSUPPORTED` (value `"streaming_unsupported"`). This is
the only breaking change: consumers matching on the error code -- e.g. LLM refine
loops reading `StreamabilityReport.errors()` or `VideoEdit.check()` output -- must
update to the new name/value.

### Renamed `StreamabilityReport.fallbacks` -> `.unstreamable`

The property listing the ops that cannot stream is now `.unstreamable`, aligned
with the existing `StreamingClass.UNSTREAMABLE`. `.errors()`, `.streamable`, and
the rest of the report are unchanged. Docstrings and docs no longer explain why a
"fallback" doesn't fall back -- the surface is self-describing.

Confirmed cleanups surfaced by the 0.44.0 streaming-only removal -- all additive
or internal refactors, no public behavior change.

### Dead code and inert state

- Removed the dead `_plan_output_seconds` helper (no call sites).
- Removed `PlanError.predicted_duration`: every construction site passed a value
  equal to `limit`, so the field was redundant. Dropped the field, its
  `to_prompt_line()` render clause, and all `predicted_duration=` kwargs.

### Internal dedup (behavior-identical)

- `VideoMetadata.with_frame_count(n)` replaces 3 inline metadata rebuilds in
  `transforms.py` (distinct from `with_duration`, which re-rounds the frame count).
- `volume_envelope(terms)` in `audio_ops.py` is now shared by the audio ducking
  code and the `Fade` / `VolumeAdjust` effects -- the compiled ffmpeg `volume`
  expression is byte-identical.
- `_optional_model_field` collapses the twin optional-field schema closures, and a
  local `make_ctx` builder collapses the 3 identical `FilterCtx` constructions in
  `video_edit.py`.
- The two `TRANSITION_TOO_LONG` guards share one error builder while each call site
  keeps its own fps resolution.
- The generic `escape_filter_value` ffmpeg escaper moved to `base/_ffmpeg` (its
  natural home now that it is borrowed across modules); all importers point there.
- `lazy_exports()` in `ai/_optional.py` collapses the 4 identical PEP-562
  lazy-import blocks across the `ai` subpackages, preserving the exact export
  surface.

### Docs and tests

- Fixed the `SpeechBackend` docstring (`synthesize` -> `generate_audio`) and trimmed
  stale streaming-only transitional comments in `video_edit.py` / `streaming.py`.
- Hoisted the shared `_toplevel_imports` / `_flatten_extra` test helpers into
  `conftest.py`.

## 0.44.0

Breaking: the eager/in-memory editing path is gone. Streaming-to-file
(`run_to_file`) is now the only execution engine -- nothing runs in memory.

### Removed `VideoEdit.run()`

`VideoEdit.run()` (execute the plan into an in-memory `Video`) is gone, along with
its in-memory machinery (`stream_segment_to_frames`, `stream_segment_audio_to_array`,
`stream_transition_frames`, `out_frames_reshape`). It was a hand-synced twin of
`run_to_file` that defeated the engine's O(1)-memory guarantee for the sake of a
convenience return value.

Use `run_to_file` and read the result back when you need a `Video`:

```python
from videopython.base.video import Video

out = edit.run_to_file("output.mp4")
video = Video.from_path(str(out))   # only when you actually need frames in memory
```

`run_to_file` is unchanged: same signature, same structured `STREAMING_FALLBACK`
errors raised before any decode.

### Removed `Operation.apply()` and the eager twins

`Operation.apply(video) -> Video` and its machinery are removed: `Effect.apply`,
`Effect._apply`, `Effect._resolved_window`, every `Transform.apply` /
`_audio_apply` numpy implementation, the effect `apply` / `_audio_apply` overrides
(`Fade`, `VolumeAdjust`, `FullImageOverlay`, `Vignette`), and
`TranscriptionOverlay.apply`. An Operation is now defined purely by its streaming
contract: `to_ffmpeg_filter` / `to_ffmpeg_audio_filter`, `streaming_init` +
`process_frame`, and `predict_metadata`.

To apply an operation, build a `VideoEdit` and stream it:

```python
edit = VideoEdit.from_dict(
    {"segments": [{"source": "in.mp4", "start": 0, "end": 5,
                   "operations": [{"op": "resize", "width": 1280, "height": 720}]}]}
)
edit.run_to_file("out.mp4")
```

`FullImageOverlay`'s shape / fade-time validation moved to `predict_metadata`, so
it still fails fast at `validate()`. The AI pipeline no longer loads video into
memory to edit it: `ai/dubbing` and `ai/video_analysis` slice `Video` directly
(frames + audio) instead of `CutSeconds(...).apply(video)`. `Video.from_frames`,
`Video` slicing, and the per-frame `process_frame` streaming hook are unchanged.

## 0.43.1

Four low-risk consumer wins (P2.14 + P3), all additive or bugfix -- no breaking
changes.

### `PlanError.to_prompt_line()` (P2.14)

A structured `PlanError` can now render itself as a single actionable line for
an LLM refine loop, composed from its fields:
`<CODE> [at <location>] [(op '<op>')][: <clauses>][ -- <detail>]`. Every
`PlanErrorCode` renders a non-empty line and `None` fields are omitted cleanly;
`PlanValidationError.prompt_feedback()` newline-joins the lines over its errors.
Purely additive -- existing per-site messages and `PlanError` fields are
unchanged.

### Predictor context managers

A small `ManagedPredictor` mixin (`videopython.ai._predictor`) gives every
predictor that defines `unload()` an `__enter__`/`__exit__`, so VRAM is released
on block exit:

```python
with SceneVLM() as vlm:
    vlm.detect(frame)
# unload() fires here, on success or exception
```

Mixed into 12 predictors across `ai/understanding` and `ai/generation`;
`__exit__` never suppresses exceptions. Replaces the manual `.unload()`
discipline between Modal pipeline stages.

### Pinned model revisions

HuggingFace loads now pin a commit SHA via a central registry
(`videopython.ai._revisions`, `pinned(model_id) -> str | None`), so silent
upstream weight changes can't alter production analysis. 11 fixed repos are
pinned (Qwen3.5 VLM sizes, pyannote diarization, AST classifier, YOLOv8-face,
Qwen3 GGUF, MusicGen, SDXL, CogVideoX t2v/i2v). Loaders with no fixed HF repo
are left unpinned by design and documented inline: the ultralytics YOLO asset,
dynamic Marian language-pair models, Chatterbox's internal load, and the
openai-whisper CDN download. `pinned()` returns `None` for those (the library
default). Refresh SHAs per the registry docstring.

### `Video.add_audio()` sample-rate negotiation

Fixed a correctness bug: overlaying audio onto a non-silent track at a different
sample rate previously raised or produced silent A/V drift. Incoming audio is
now resampled to the existing track's rate (the rate encoded into the video)
before any length math, via the existing `Audio.resample()`. Pure attach /
replace keep the incoming rate. Behavior-only fix; signature unchanged.

## 0.43.0

The P1 roadmap items, shipped as one
release. Highlights: dubbing dependency isolation, multi-clip per-source
context, and an ffprobe probe cache.

### Dubbing dependency isolation: granular `[ai]` extras + a pluggable TTS backend

The monolithic `[ai]` extra is split into per-capability extras so a consumer
can build a slim, conflict-free image per capability instead of pulling the
entire ML stack (and `chatterbox-tts`'s strict pins) for one feature:

| Extra | Covers |
|---|---|
| `asr` | transcription / diarization (`understanding/audio.py`) |
| `vision` | detection, scene/temporal, VLM (`understanding/{faces,objects,temporal,image}.py`, `video_analysis`) |
| `separation` | source separation (`understanding/separation.py`) |
| `translation` | MarianMT + Qwen3 GGUF (`generation/{translation,qwen3}.py`) |
| `tts` | Chatterbox voice cloning (`generation/audio.py`) — isolated |
| `generation` | SDXL / CogVideoX / MusicGen (`generation/*`) |
| `dub` | the dubbing pipeline (`asr + separation + translation + pyloudnorm`) |
| `ai` | convenience aggregate of every extra (PEP 685 self-references) |

**`[dub]` deliberately excludes `chatterbox-tts`.** The blocker that made the
dubbing wedge undeployable was chatterbox's `torch==2.6.0` pin conflicting with
`pyannote-audio` (`torch>=2.8`) and CogVideoX (`diffusers>=0.30`). Local
synthesis now runs through a new `runtime_checkable` `SpeechBackend` protocol
(`ai/generation/_tts_backend.py`): `VideoDubber` / `LocalDubbingPipeline` accept
a `tts_backend=` injection, so a consumer can run TTS in its own image/function
(install `[tts]` there, or inject a remote backend) while a `[dub]` image
co-resolves cleanly **without** chatterbox. `[tts]` installs standalone on
chatterbox's own pins. The `[tool.uv].override-dependencies` block is now
load-bearing only for the all-in `[ai]` resolve.

**Lazy `ai` imports.** `ai/__init__.py` and the `generation`/`understanding`/
`dubbing` sub-packages are PEP 562 `__getattr__` lazy re-exports, and every
heavy import is guarded by `ai/_optional.require(module, extra)`, which raises
a clear `pip install 'videopython[<extra>]'` hint on a missing dependency.
`import videopython.ai` (and importing a single leaf class) no longer drags in
sibling heavy modules — verified in `test_packaging_extras.py`, which also
drift-guards `union(granular) == [ai]` so the dep lists never hand-desync.

BREAKING: `pip install videopython[ai]` still installs everything, but
consumers pinning a sub-capability must migrate to the new extra names; a bare
`[dub]` install that reaches local synthesis raises an ImportError pointing at
`[tts]`; `numba`/`scipy`/`scikit-learn`/`ollama`/`hf-transfer` drop to
transitive-only.

### `VideoMetadata.from_path` caches ffprobe probes

Every plan traversal (`repair` → `check` → `validate` → `run`) re-probed each
source file per segment; a job that chains those over a handful of clips paid
dozens of redundant ffprobe subprocesses. Probes are now cached in a bounded
LRU keyed by `(resolved path, mtime_ns, size)`, so repeated probes of an
unchanged file in one process collapse to one call, while a file modified in
place is re-probed automatically (the stat key changes). The lock guards only
dict access — the ffprobe subprocess runs outside it, so concurrent probes of
different files never serialize. New `VideoMetadata.clear_cache()` forces a
re-probe for the rare in-place overwrite that preserves both mtime_ns and size.
No API or behavior change otherwise.

### Per-source runtime context for multi-clip plans

A runtime context value may now be a per-source map keyed by
`str(segment.source)`, so a plan that cuts from several sources can feed each
segment its OWN transcription:

```python
edit.run_to_file(out, context={"transcription": {"a.mp4": tx_a, "b.mp4": tx_b}})
```

A bare value still broadcasts to every segment (unchanged), mirroring the
existing `VideoMetadata | dict[str, VideoMetadata]` precedent in
`_resolve_source_metas` — runtime metadata and runtime context now share one
mental model. Resolution happens at the single `_segment_context` chokepoint
(per-source select, then the existing per-segment re-basing), so nothing
downstream changed. This lifts the block on subtitles in multi-clip project
edits, where a single global transcription was wrong for every segment past
the first.

`check()` now reports a structured `CONTEXT_SOURCE_MISSING` error when an op
requires a per-source key whose map omits that segment's source — otherwise an
`add_subtitles` plan would pass validation (subtitles do not change predicted
metadata) and only fail at decode.

BREAKING: the private `_segment_context(context, start, end)` helper gains a
`source` argument (`_segment_context(context, source, start, end)`).

### Segment transitions (`xfade` / `acrossfade`)

Segment boundaries were hard cuts only. A segment may now carry an optional
`transition_in` describing how it enters from the previous segment:

```python
{"source": "b.mp4", "start": 0, "end": 5,
 "transition_in": {"type": "dissolve", "duration": 0.5}}
```

`TransitionSpec` exposes a curated catalog (`fade`, `dissolve`,
`wipeleft/right/up/down`, `slideleft/right`) as a closed enum so the strict
LLM grammar stays tight, a `duration` (seconds of overlap), and `audio`
(acrossfade the audio across the same overlap, else a hard butt-join). It
surfaces automatically in `VideoEdit.json_schema()`. N segments → N-1
transitions; `segments[0].transition_in` must be `None`.

Streaming-first: each transition-adjacent segment is realized by the existing
per-segment engine (effects + audio baked), then a native two-decoder
`xfade`+`acrossfade` pass joins the two finished files. Maximal hard-cut runs
still `concat -c copy`; only transition seams re-encode. `run()` and
`run_to_file` share one `xfade` filter-string builder, so the seam pixels
match. `_assemble_timeline` subtracts each `round(duration*fps)` overlap, so
predict / `check` / `repair` / streamability all see the shortened output.

A transition that would overlap a whole adjacent segment is a structured
`TRANSITION_TOO_LONG` error (the constraint is frame-based: `round(D*fps)`
must be strictly fewer frames than the shorter adjacent segment); `check`
reports it and `repair` clamps it one frame short of the limit. `post_operations`
combined with transitions are reported `UNSTREAMABLE` (a post-op envelope
cannot fold across the separately re-encoded seam).

### Audio in the filter graph

Segment audio moved off the in-memory path into the ffmpeg graph: the original
source is a second `-i` input routed through `-filter_complex` in the **same**
per-segment invocation as the video (no more whole-source `Audio` decode, no
temp-WAV mux). This closes the one honest caveat on "streaming-only" — segment
audio was the last fully-in-memory stage (~1.2 GB/hour stereo). Each
audio-affecting op compiles an `to_ffmpeg_audio_filter` twin at the same plan
stage as its video filter: `speed_change` → `atempo`, `freeze_frame` → silence
splice, `silence_removal` → sample-accurate `atrim`+`concat` keep-windows,
`fade` → a windowed `volume` envelope, `volume_adjust` → `volume`. A
length-pin (`atrim`+`apad` to the folded output duration) plus
`aresample=async=1` hold A/V sync within the existing 0.15s tolerance; a source
with no audio stream gets a native `anullsrc` silent track.

`run()` stays a view over the same compiled graph — it decodes the audio graph
to a PCM buffer for `video.audio` (O(output-duration) memory, in-memory view
only), so `run()` and `run_to_file` cannot diverge.

BREAKING: `Operation.transform_audio` → `to_ffmpeg_audio_filter`;
`StreamingSegmentPlan.audio_ops`/`post_audio_ops` → `af_filters`/`post_af_filters`;
`VideoEdit._load_segment_audio` removed; audio fade/volume curves are now native
ffmpeg (`afade`/`volume`-shaped), not bit-identical to the old numpy ramps (the
libass precedent).

### Music bed + transcription-derived ducking

A plan can now carry a `VideoEdit.music_bed` (a frozen `MusicBed`: `source`,
`gain`, `loop`, `fade_in`/`fade_out`, and an optional `duck`). The bed is mixed
under the **whole assembled** program in a final `amix` pass after concat /
transitions — `run()` and `run_to_file` share one bed-mix filter builder, so
they cannot diverge. `amix=duration=first` clamps the output to the program
length (a short bed loops, a long bed is trimmed) and `normalize=0` keeps the
program dialogue at full level. An unreadable bed source fails `validate`/`check`
with `SOURCE_UNREADABLE` before any decode.

`music_bed.duck` lowers the bed under speech using deterministic,
transcription-derived `volume` automation (no live key signal): the speech
windows come from a shared `speech_windows()` helper that `silence_removal` also
uses. Ducking is single-segment only (the assembled timeline maps cleanly to one
segment's re-based transcription); a multi-segment plan with `duck` set raises a
structured `MUSIC_BED_DUCK_MULTISEGMENT` error (a non-ducked bed on a
multi-segment plan is fine).

Per-segment / per-window **gain** already shipped with the audio-graph migration
above (`volume_adjust` compiles to a `volume` node), so no separate op was
needed.

## 0.42.0

**Streaming-first migration complete: streaming is the only execution
engine.** Every operation compiles to an ffmpeg filter chain or streams as a
per-frame effect; the silent whole-plan eager fallback is gone, and `run()`
is a view over the same engine that streams into memory (lossless rawvideo)
instead of encoding. Plan shapes with no streaming strategy are rejected
before any decode with structured `STREAMING_FALLBACK` errors carrying
reorder hints.

**Every remaining op streams natively:**

- `speed_change`: `setpts` retime (bias-corrected nearest-frame for
  speedups; closed-form log-curve ramps, exact to half a timebase tick) +
  `fps`/`framerate` (blending) resample.
- `freeze_frame`: linear `loop`/`select` chains for all three modes.
- `silence_removal`: the transcription is consumed at plan compile and the
  silent gaps drop via a `select` keep-window cut; gains a real
  `predict_metadata` so validation folds the cut duration.
- `face_crop` (ai extra): the detection pass runs at plan-compile time over
  a bounded decode of exactly the frames the filter sees, compiling the
  smoothed track to a per-frame `sendcmd` crop -- zero per-frame Python at
  render time.

**The engine folds duration and audio.** The plan builder threads real
metadata through the chain (pipe stage vs final stage), the plan carries
authoritative frame counts, effect envelopes size to the post-transform
timeline, and segment audio follows through per-op audio twins
(`Operation.transform_audio`: time-stretch, silence insertion, gap cuts)
replayed in chain order around the effect envelopes.

**Plan-order scheduling across two filter stages.** Transforms ordered
after frame effects join the encoder's filter chain (`FrameEncoder -vf`),
so `[fade, crop]` and `[fade, speed_change]` stream in plan order. Post-op
effects fold across multi-segment plans with globally-rebased frame
offsets -- envelopes continue across concat boundaries -- closing the last
whole-plan fallback trigger (the brand-logo overlay on multi-segment
plans).

**The unstreamable shapes that remain** (all rejected with explicit
errors): a frame effect ordered after encode-stage filters; time-based
context after a duration-changing transform; transform post-ops;
audio-coupled post-ops (`fade`/`volume_adjust`) on multi-segment plans;
`face_crop` behind frame effects; in-plan `cut` ops.

**BREAKING (no deprecation):**

- The `reverse` op is removed (whole-video buffering has no place in a
  streaming-first engine).
- `silence_removal` loses `mode="speed_up"` and `speed_factor`.
- `face_crop`: fixed crop-window size (the size-expansion clause is gone,
  crops are pure slices); `fallback="full_frame"` behaves as `"center"`.
- `strict_streaming` kwargs removed from `run_to_file()` and `check()` --
  strictness is the only behavior; `check()` always reports unstreamable
  ops as plan errors.
- `StreamingClass.EAGER` renamed to `UNSTREAMABLE` (value
  `"unstreamable"`).
- `VideoEdit.run()` executes via the streaming engine: per-op in-memory
  execution inside `VideoEdit` (`SegmentConfig.process`) is deleted, and
  `run()` output pixels come from the decode chain. Per-op `apply()`
  remains as the direct single-op API.
- Plans that previously relied on the silent eager fallback now raise.

## 0.41.0

**The subtitle renderer is now libass.** `add_subtitles` consumes its
transcription when the plan compiles, bakes the look into an ASS document,
and ffmpeg's `subtitles=` filter burns it in — native-speed rendering with
zero per-frame Python (~4x faster than the removed PIL renderer on display
fonts). One pixel path everywhere: the streaming path emits one `-vf` entry,
and the eager `run()` path pipes the in-memory frames through the same
filter. ffmpeg must include libass. The PIL rendering engine
(`videopython.base.ImageText`, ~1,200 lines plus the layout/fit machinery)
is deleted.

- **Filter-class effects.** New `Effect.compiles_to_filter` hook: the op
  joins the filter chain at its plan position instead of scheduling
  `process_frame` — the decode chain normally, or the new encode-stage chain
  (`StreamingSegmentPlan.post_vf_filters`, `FrameEncoder`'s `-vf`) when frame
  effects precede it. Both `[add_subtitles, crop]` and
  `[fade, add_subtitles]` now stream end-to-end in plan order; under the old
  renderer the former forced the whole-plan eager fallback. A frame effect
  ordered *after* the burned-in subtitles is the one shape that still falls
  back to eager (reported as such by `streamability()`). `FilterCtx` gains
  `context` (re-based, segment-local runtime values for compile-time
  consumption) and `owned_files` (compile-time temp artifacts, deleted by the
  runner on every exit path).
- **ASS mapping.** Style presets and region placement map natively
  (`\an`+`\pos` box-center semantics, margins from `box_width`, background
  via libass `BorderStyle=4`); the word-level highlight is word-state
  Dialogue events (active-word color + `\fscx/\fscy` size pop — ASS `\k`
  karaoke can only express stay-lit highlighting). `font_scale` is corrected
  for libass's cell-height font sizing vs PIL's em sizing, so the apparent
  size is unchanged. Bundled fonts ship to libass via `fontsdir`;
  `BUNDLED_FONT_FAMILIES` pins the name-table families it matches on.
  `window` is applied by clipping event times at compile (the `subtitles`
  filter has no timeline support).
- **Breaking — removed without deprecation:**
  - `add_subtitles` fields `min_font_scale`, `text_align`, `margin`, and
    `highlight_bold_font` (libass wraps instead of shrinking; centers text;
    derives margins from `box_width`; uses one font face). Plans carrying
    them now fail to parse.
  - Subtitle fit validation: libass wraps long cues, so `validate()`/
    `check()` no longer reject "unfittable" subtitles;
    `PlanErrorCode.SUBTITLE_UNFITTABLE` is removed.
  - `videopython.base.ImageText`, `TextAlign`, `TextRenderError`,
    `OutOfBoundsError` are removed. `AnchorPoint` (still the `anchor` field
    type) moved to `videopython.editing`.
  - Subtitle pixel output differs from 0.40.0 (human-reviewed side by side;
    placement and box metrics shift slightly).

## 0.40.0

Streamability becomes a contract instead of a silent behavior. Whether
`run_to_file` streams in O(1) memory or eager-loads the whole video is now
inspectable before running anything, and enforceable. Second step of the
streaming-first migration.

- **Per-op streamability report.** `VideoEdit.streamability()` classifies
  every op in plan order by streaming class — `filter` (ffmpeg `-vf`),
  `frame_effect` (`streaming_init`/`process_frame`), or `eager` (no streaming
  strategy; forces the whole-plan fallback) — with the reason on each `eager`
  entry. Purely structural: no source files, metadata, or context needed, so
  admission gating can happen before anything is downloaded.
  `report.streamable` is the plan-level verdict; `report.errors()` renders
  the fallbacks as structured `PlanError`s.
- **`check(..., strict_streaming=True)`** appends one `STREAMING_FALLBACK`
  error per fallback-forcing op (after the regular validity errors, in plan
  order), so an LLM refine loop treats "won't stream" like any other plan
  violation. Default `check()` behavior is unchanged.
- **`run_to_file(..., strict_streaming=True)`** raises `PlanValidationError`
  carrying those errors instead of silently eager-loading — before any
  decode. The former fallback sites are also guarded: if a plan stops
  streaming despite a clean report (e.g. a third-party transform whose
  `streamable` flag disagrees with its `to_ffmpeg_filter`), strict mode
  raises rather than eager-loading. Default behavior is unchanged.
- **New error vocabulary.** `PlanErrorCode.STREAMING_FALLBACK`, plus an
  optional `PlanError.detail: str | None` field carrying a human-readable
  cause for codes where the code alone isn't actionable (populated for
  streaming fallbacks; `None` everywhere else, so existing constructions and
  comparisons are unaffected).
- **The `streamable` flag is now authoritative for transforms.** The plan
  builder consults it before compiling `to_ffmpeg_filter`, so a transform
  declaring `streamable=False` takes the eager path even if its filter
  compiles — the report and the runtime can no longer disagree in that
  direction. Affects only third-party ops with a mismatched declaration
  (a working filter but `streamable=False` previously streamed silently);
  all built-in ops declare coherently, pinned by registry tests in both the
  editing and ai suites.
- New exports: `StreamabilityReport`, `OpStreamability`, `StreamingClass`
  from `videopython.editing`.

## 0.39.0

Context-requiring effects now stream. `add_subtitles` — previously the most
common reason a whole plan silently fell back to eager, in-memory execution —
runs on the O(1)-memory streaming path. First step of the streaming-first
migration.

- **Context threading.** `run_to_file` resolves each segment's `requires`
  context at plan-build time, re-based onto the segment-local timeline by the
  same `_segment_context` machinery the eager path uses (slice to
  `[start, end)`, shift by `-start`, drop empty slices). The resolved kwargs
  travel on `EffectScheduleEntry.context` and are forwarded to the effect's
  `streaming_init` — the streaming twin of `_apply_with_context`. Context
  errors (missing/non-overlapping transcription) raise *before that segment
  decodes*, with the same exception as the eager path. Context-requiring
  *transforms* (`silence_removal`) still fall back to eager: an ffmpeg filter
  string can't consume runtime context.
- **Transform-after-effect plans now fall back to eager** instead of
  streaming with reordered semantics. Streaming hoists all transform filters
  to decode time, *before* every per-frame effect, so a segment like
  `[fade, resample_fps]` or `[add_subtitles, crop]` streamed against a
  different timeline/dims than plan order prescribes — mistimed envelopes,
  and for subtitles a layout resolved at post-transform dims that could fail
  *after* `validate()` passed. Such plans (a pre-existing latent bug for
  context-free effects, newly reachable for `add_subtitles`) now execute
  eagerly, in strict plan order. Transforms-then-effects plans stream as
  before.
- **Bounded subtitle overlay cache.** The per-stream overlay memo evicts on
  cue change (frames arrive in time order; only the active cue's highlight
  variants are needed), bounding it to `max_words_per_cue + 1` full-frame
  overlays regardless of video length — without this, an hour of 1080p
  speech would accumulate gigabytes on the "O(1)-memory" path.
- **Trailing-frame fix in `stream_segment`.** The scheduled frame count is
  an estimate (`round(duration * fps)`); on rounding ties ffmpeg can emit
  one extra frame, which previously escaped every full-range effect (an
  unfaded frame popping after a fade-out, subtitles vanishing on the last
  frame). Full-range entries are now open-ended with a clamped window-local
  index.
- **`TranscriptionOverlay` ported to the streaming contract**
  (`streamable=True`): subtitle layout/cue state precomputes in
  `streaming_init`, per-frame rendering lives in `process_frame`, and the
  eager path replays exactly that contract via the base `Effect._apply` — one
  pixel path, parity by construction (the 0.36.1 principle).
- **`Effect.streaming_init` signature widened** to
  `streaming_init(total_frames, fps, width, height, **context)`; the base
  `apply`/`_apply` thread `**context` through. Effects without `requires` are
  always called without context kwargs. **External `Effect` subclasses must
  widen their `streaming_init`/`_apply` overrides with `**_context: Any`.**
- **Behavior changes in `TranscriptionOverlay`:** `window` is now honored
  (previously ignored — the overlay rendered over the full clip regardless);
  window-local frame indices are mapped back to segment time, so cue timing is
  unchanged. `apply()` now mutates input frames in place, like every other
  effect (it used to return a fresh `Video`); snapshot frames before applying
  if you compare against the input.

## 0.38.0

Validation/repair primitives that let an LLM-driven compiler converge on a valid
`VideoEdit` in a single re-prompt instead of playing whack-a-mole across a retry
budget. The plan skeleton now **parses permissively** (shape only) and owns its
numeric bounds at validation, so every mechanical violation is a structured,
collectable, repairable `PlanError` rather than a hard `from_dict` failure.

- **Permissive parse (breaking).** `TimeRange.start/stop` and
  `SegmentConfig.start/end` drop their parse-time `ge=0` / ordering validators —
  a negative `window.start` or a `start >= end` segment now *parses* and is
  reported by validation. The boundary: parsing owns shape/required-fields
  (still a Pydantic `ValidationError` — e.g. `resize` with no dimension, unknown
  ops, extra fields); validation owns the numeric and metadata-relative bounds
  (`PlanValidationError`).
- **`VideoEdit.check(source_metadata, *, clamp_windows=False) -> list[PlanError]`**
  — the non-raising sibling of `validate`: runs the full dry-run and returns
  **every** error in one pass (`[]` == valid), best-effort isolating failures
  per segment/op. `validate*()` still raises, byte-stable.
- **Structured everywhere.** The remaining bare `ValueError`s on the
  predict/validate path are now `PlanValidationError` with new `PlanErrorCode`s
  (`OP_TIMESTAMP_OUT_OF_RANGE`, `CROP_EXCEEDS_SOURCE`, `DEGENERATE_DURATION`,
  `SOURCE_UNREADABLE`, `WINDOW_NEGATIVE`, `WINDOW_ORDER`, `SEGMENT_NEGATIVE`,
  `SEGMENT_RANGE`, `POST_OP_REQUIRES_CONTEXT`, `OP_PREDICTION_FAILED`), so
  nothing escapes the collect-all walk and consumers branch on `code`.
- **Real `repair()`.** `repair(..., clamp_op_params=True, clamp_segment_end=False)`
  now clamps effect `window.start`/`window.stop` into `[0, duration]` (segment
  ops *and* `post_operations`), op time fields past the clip end
  (`freeze_frame.timestamp`, generic via a per-op `BoundedTimeField`
  declaration), and a negative segment `start` — returning a structured
  `PlanRepair` changelog. **Breaking:** `WindowClamp` is removed; `repair()` now
  returns `list[PlanRepair]` (`location, field, old, new, code`).
- **`VideoEdit.normalize_dimensions(source_metadata, target)`** — appends a
  per-segment `resize` to a common canvas (`(w, h)` / `"first"` / `"largest"`)
  so `CONCAT_MISMATCH` is satisfiable by construction; returns the changelog.
  Best-effort and non-raising like `repair()`/`check()`: a segment it cannot
  predict is left untouched and deferred to `check()`, so the
  `repair -> normalize_dimensions -> check` flow has one non-raising path.
- **Strict schema.** `VideoEdit.json_schema(strict=True)` /
  `Operation.json_schema(strict=True)` emit a *submittable* provider strict-mode
  grammar (closed objects, all-required, union `$defs` hoisted to the document
  root so every `$ref` resolves, `anyOf` union without `discriminator`, numeric
  constraints preserved). Optionality is taken from the Pydantic type, not from
  "has a default": only genuinely `Optional` fields are nullable, so a
  grammar-valid payload always round-trips through `model_validate`.
  `PlanError`/`PlanErrorCode`/`PlanValidationError`/`PlanRepair` are now exported
  from `videopython.base`.

## 0.37.0

Adds `ObjectDetectionOverlay`, an AI-powered, streamable effect (op
`object_detection_overlay`) that detects objects per frame with a YOLOv8-COCO
model and draws labelled bounding boxes. Built as three small,
independently-testable layers mirroring the `FaceTracker` / `FaceTrackingCrop`
split, so the `editing` layer stays AI-free:

- **AI-free renderer** (`videopython.base.draw_detections`): `draw_detections()`
  plus a `DetectionStyle` value object and deterministic per-class
  `class_color()`, reusing the existing `DetectedObject` / `BoundingBox` /
  `load_font` primitives. Anti-aliased PIL label chips in the box colour,
  resolution-scaled stroke/font, and edge-aware label placement; testable with
  synthetic detections, no GPU.
- **`ObjectDetector`** (`videopython.ai.understanding.objects`): a lazy
  YOLOv8-COCO detector returning `list[DetectedObject]` with normalized boxes —
  a near line-for-line counterpart to the face detector (`detect` /
  `detect_batch`, device selection, confidence threshold, optional
  `class_filter`).
- **`ObjectDetectionOverlay`** (`videopython.ai.effects`): a real
  shape-preserving `Effect`. Only `streaming_init` / `process_frame` are
  overridden, so the base `Effect._apply` replays the identical contract and
  eager / streaming cannot drift. `requires=()` keeps it both streamable and
  LLM-exposed. Detection runs on a `detection_interval` cadence (default 2) and
  boxes hold between detections, so cost is compute-bound, not memory-bound; cap
  it with `window`, `detection_interval`, `class_filter`, and `model_size`
  (`n` / `s` / `m`). The infra `backend` field is `llm_hidden`.

Additive only — no wire-format or behavior change to existing ops. Requires the
`[ai]` extra.

## 0.36.1

Internal refactor of the editing core; no wire-format or runtime-behavior
change. Streaming becomes the single source of truth for every `Effect`: the
base `Effect._apply` now replays `streaming_init` + `process_frame` over the
in-memory frames, so the 14 pure-frame effects drop their hand-written eager
`_apply` and can no longer drift from the streamed path. `FullImageOverlay`
(eager-only validation), `Vignette` (batched vectorisation), and the audio
effects (`Fade`/`VolumeAdjust`) keep their bespoke eager paths. `TextOverlay`
and `ImageOverlay` now share placement, off-frame clipping, and alpha blending
via a new `_AnchoredOverlay` base (one `_overlay_for_frame` hook each), closing
the verbatim-copy parity contract behind the 0.34.1 fix. Easing curves are
factored into `editing/_easing.py` (used by `KenBurns` / `PunchIn`). The only
observable difference: `position` / `anchor` sort earlier in the overlay ops'
generated JSON schema (now inherited fields) -- cosmetic, irrelevant to tool
use.

## 0.36.0

Sharpens the LLM-editing surface so consumers write less glue and hit fewer
validate-retry loops. Six changes, several **breaking** on the schema / error
surface:

- **Named bundled fonts.** Poppins-Bold, Lato-Bold, Anton, and BebasNeue ship
  with the package (alongside their OFL licenses). `TextOverlay` and
  `add_subtitles` gain a `font` field — a fixed enum (`anton`, `bebas-neue`,
  `lato-bold`, `poppins-bold`, exposed as `videopython.base.fonts.FONT_NAMES`)
  an LLM can pick from, and stored plans round-trip on the name. `load_font`
  resolves a registered name before the path → DejaVu → PIL fallback and stays
  non-raising. `font_filename` remains as an advanced override (it takes
  precedence over `font`).
- **Server-only ops.** New `llm_exposed: ClassVar[bool]` (default `True`) plus
  `Operation.llm_registry()`. `image_overlay` and `full_image_overlay` are now
  `llm_exposed=False` — they need a server-resolved `source` path the model
  can't supply. **Breaking:** `Operation.json_schema()` and
  `VideoEdit.json_schema()` now cover only LLM-exposed ops by default; pass
  `Operation.json_schema(include_server_only=True)` for the full union.
  `Operation.registry()` and `from_dict` are unchanged (they still see all ops).
  The same hiding works per field: `Field(json_schema_extra={"llm_hidden": True})`
  drops a valid-but-advanced field from the LLM-facing schema. `font_filename`
  (on `text_overlay`/`add_subtitles`) and `highlight_bold_font` are now hidden —
  the `font` name enum is the LLM-facing surface; the raw paths remain parseable.
  New `cls.llm_json_schema()` gives the per-op schema with these stripped.
- **Typed validation errors. (Breaking)** `validate()` now raises
  `PlanValidationError` (a `ValueError` subclass — `str(e)` is unchanged and
  `except ValueError` still works) carrying structured `.errors`: a list of
  `PlanError(code, location, op, field, value, limit, predicted_duration)`, so
  consumers branch on a code instead of substring-matching the message.
- **Window-clamp repair.** `validate(clamp_windows=True)` /
  `validate_with_metadata(..., clamp_windows=True)` clamp an effect `window.stop`
  that overruns a duration-shrunk chain (after `cut` / `speed_change` /
  `silence_removal`) to the run-time value instead of raising — closing a
  validate-vs-`run()` divergence with no extra LLM call. `VideoEdit.repair()`
  returns a corrected copy of the plan plus the applied `WindowClamp`s.
- **Unified duration tolerance.** A single `DURATION_EPS` (1e-3) is applied at
  the segment-end guard, effect-window check, and `CutSeconds`/`CutFrames`, so a
  sub-millisecond boundary value is accepted (and reported) consistently.
- **Drift-proof `VideoEdit.json_schema()`.** Reimplemented as a thin transform
  over `model_json_schema()` so it can't desync from the models; `source` now
  correctly carries `"format": "path"`.

## 0.35.1

Fixes a hard failure in `Qwen3Translator` on long sources. The translator
built one prompt containing every segment, so a long source with hundreds
of dense segments ballooned past the default 8192-token `n_ctx` and
llama.cpp rejected the call with a context-window overflow.
`translate_segments` now splits the input across as many Qwen calls as
needed via a char-budget heuristic (`_chunk_segment_indices`), and the
retry pass is chunked the same way. Indices, progress callback semantics
(ramps 0 → 0.5 across first-pass chunks, then 0.9, then 1.0), and the
Marian fallback path are preserved; the default 8192 `n_ctx` is now safe
for sources of any length.

## 0.35.0

New `image_overlay` operation: a scaled, anchored, time-windowed image
overlay for logos / watermarks / brand marks. Unlike `full_image_overlay`
(full-frame, raises on size mismatch), `scale` is a fraction of frame width
so one config is resolution-independent across 1080p / 4k / vertical /
square; `position`/`anchor` reuse `TextOverlay`'s geometry and off-frame
placement clips to a no-op (never an error). Streams via the `run_to_file`
fast path with eager/streamed parity.

`source` may be a raster image or an **SVG** (detected by the `.svg`
extension), rasterised by `resvg` *at the exact target pixel width* -- crisp
at any frame size, not an upscaled bitmap -- with a transparent background
and no remote-resource fetching (local path only; no SSRF). `predict_metadata`
rejects only a missing/unreadable `source` at `validate()` time (a cheap
header / 1px-SVG parse) -- geometry that `run()` can clip is not rejected.
This codifies the `predict_metadata` contract (reject iff `run()` would fail)
in the `Operation` docstring.

**Breaking:** minimum Python is now **3.11** (was 3.10). `resvg-py` is the
SVG rasteriser and ships no 3.10/macOS wheel; rather than break `pip install`
on that slice or hide SVG behind an extra, the floor moves to 3.11 (3.10 is
near security-EOL). `resvg-py` is a core dependency (1.7 MB, self-contained,
no system libraries).

## 0.34.1

Fixes a parity hole in the 0.34.0 subtitle fit check: `add_subtitles`
enlarges the spoken word by `highlight_size_multiplier` at draw time, but
measurement sized everything at the base font, so a cue that "fit" could
still raise `OutOfBoundsError` mid-render once a word lit up.
`measure_text_box` is now highlight-aware (worst-case over every word via a
shared `_highlighted_line_size`), the renderer positions the highlighted
line by its true extent, and `TranscriptionOverlay` also requires the
worst-case line to fit the box -- so an overflowing cue is auto-shrunk or
rejected in `predict_metadata`/`validate()`, never crashed in `run()`.
Non-highlight callers are byte-identical (multiplier `1.0` default).

## 0.34.0

`TranscriptionOverlay` subtitles are now resolution-independent: size and
place them with `style`/`region`/`font_scale` (fractions of the frame),
not absolute pixels. This closes the validate/run fit gap — `add_subtitles`
no longer crashes mid-render with `OutOfBoundsError`; un-fittable plans
fail fast in `VideoEdit.validate()` (also fixed: `face_crop` now predicts
its real output size). Legacy absolute fields still work as overrides.

**Breaking:** a bare `TranscriptionOverlay()` renders larger and
bottom-aligned; pin `font_size=40, position=(0.5, 0.7)` to restore.

## 0.33.5

`VideoEdit` now re-bases time-based runtime context onto each segment's
local timeline before its operations run. A cut segment is decoded
0-based (its first frame is `t=0`), but a `Transcription` passed via
`run(context=...)` / `validate(context=...)` carries source-absolute
timestamps. Previously the raw transcription was threaded unchanged into
every segment, so `add_subtitles` (and `silence_removal`) on a segment
cut from the middle of a video saw timestamps that never matched the
0-based frames: subtitles rendered **blank**. They are now sliced to the
segment's `[start, end)` and shifted by `-start`, so any segment -- not
just one starting at `t=0` -- gets correctly timed subtitles. Words
outside the segment no longer bleed in, even when `start == 0`.

Re-basing is generic: any runtime-context value implementing
`slice(start, end)` and `offset(delta)` (the new structural
`SegmentRebaseable` protocol) is re-based, not just the concrete
`Transcription` type -- keeping the context mechanism open to future
time-based context and removing a layering dependency from the editing
layer.

**Scope:** re-basing is per-segment only. `post_operations` run on the
assembled, concatenated timeline, which cannot be re-based across a
multi-segment concat. A multi-segment plan with a `post_operation` that
`requires` time-based context now **raises a clear error up front**
(during `validate()` / `run()` / `run_to_file()`) instead of silently
rendering against the wrong timeline; single-segment plans are
unaffected. When a segment contains no overlapping words the
`transcription` key is dropped rather than passed empty, so the
consuming operation raises its own clear "requires ..." error instead of
silently rendering nothing -- request subtitles only on segments that
contain speech.

### Changes

- `SegmentConfig.process` and `VideoEdit` metadata validation
  (`_predict_segment`) derive a per-segment context via the new internal
  `_segment_context` helper (slice to `[start, end)` then `.offset(-start)`).
  `run()` and `validate()` stay consistent.
- New `SegmentRebaseable` protocol + `_segment_context` re-bases every
  matching context value structurally; `VideoEdit` no longer imports or
  type-checks `Transcription`.
- New `VideoEdit._assert_post_ops_supported`, enforced by `validate()`,
  `run()`, and `run_to_file()`: rejects a multi-segment plan whose
  `post_operations` need time-based context.
- Streaming path defers to the eager path for any operation that
  `requires` runtime context (streaming schedules effects by frame range
  with no context, so it cannot supply or re-base one) -- previously safe
  only because such ops happen to be non-streamable; now explicit.
- New `TranscriptionSegment.from_words(...)` centralizes segment
  construction from a word group (used by `_words_to_segments`,
  `standardize_segments`, `slice`, `chunk_segments`). Fixes a latent bug
  where `Transcription.offset()` silently dropped the segment confidence
  fields (`avg_logprob`, `no_speech_prob`, `compression_ratio`); a pure
  timing shift now preserves them. `chunk_segments` no longer aliases a
  words-less source segment into its output.
- No public API or wire-format change. Plans that previously produced
  blank subtitles on mid-video cuts now render them correctly.

## 0.33.4

`TranscriptionOverlay` now normalizes subtitles by default: long
transcription segments are split into short on-screen cues, and sentence
starts are capitalized. This fixes burned-in subtitles that dumped a
whole sentence on screen at once and rendered lowercase sentence starts
from word-level speech-to-text.

**Behavior change:** with default settings, rendered subtitle text now
differs from the raw transcription (chunked to <= 5 words per cue,
sentence-cased). Pass `max_words_per_cue=None` and `capitalize=False` to
restore the previous verbatim rendering.

### Changes

- New `Transcription.capitalize_sentences()`: returns a new
  Transcription with the first word and every word after `.`, `!`, `?`,
  `…` capitalized. Existing caps (acronyms, proper nouns) are preserved;
  timing/speaker/language are carried through. Abbreviations are not
  special-cased.
- New `Transcription.chunk_segments(max_words)`: splits each segment into
  cues of at most `max_words` words **without merging across segments**,
  so silence gaps are preserved and subtitles don't linger over pauses.
  Distinct from `standardize_segments`, which flattens and re-groups all
  words globally.
- `TranscriptionOverlay` gains two fields: `max_words_per_cue` (default
  `5`, `None` disables chunking) and `capitalize` (default `True`). They
  are applied to the transcription at the start of `apply()`.

## 0.33.3

`font_filename` is now optional on `TranscriptionOverlay` (it was
required). When omitted — or set to `None` — text renders with a bundled
DejaVu Sans instead of raising. Backward compatible: existing code that
passes an explicit path is unaffected.

### Changes

- New `videopython.base.fonts.load_font` resolver with a fixed chain:
  explicit path -> bundled DejaVu Sans -> PIL's built-in font. It never
  raises for a missing or unreadable font, so a bad path falls back
  silently rather than failing the render.
- `TextOverlay` and `TranscriptionOverlay` now share this resolver.
  Previously `TextOverlay` relied on a system-installed `DejaVuSans.ttf`,
  which was platform-dependent; the font is now bundled in the wheel.
- DejaVu Sans (`DejaVuSans.ttf`) and its license are packaged into the
  wheel and sdist. This adds ~740 KB to the installed package.

## 0.33.2

Ten new streamable effects aimed at experimental / stylistic edits, all
extending `Effect` with `window` support and per-frame streaming via
`process_frame`. Nothing breaking — purely additive registry entries.

### New effects

- `Shake` — per-frame jitter with `random` / `rhythmic` / `decay` modes.
- `PunchIn` — snap-zoom emphasis with attack/release ramps; distinct
  from the existing continuous `Zoom`.
- `Flash` — solid-color frame flash with attack/decay alpha envelope.
- `ChromaticAberration` — R/B channel split, `horizontal` / `vertical` /
  `radial`.
- `Glitch` — seeded horizontal slice displacement plus channel offsets.
- `FilmGrain` — seeded Gaussian noise, luma-only or per-channel.
- `Sharpen` — unsharp-mask sharpening with odd-kernel validation.
- `Pixelate` — mosaic blocks, full frame or normalized `BoundingBox`
  region (face-censor friendly).
- `MirrorFlip` — full flip or half-mirror (`mirror_left`, `mirror_top`,
  etc.).
- `Kaleidoscope` — N-way radial mirror around the frame center, with
  precomputed remap maps.

### Other

- Fixed a pre-existing test bug in `test_video_resize` that mutated the
  session-scoped `small_video` fixture in place. The corruption
  shadow-failed `test_image_text.py::test_overlaying_video_with_text`
  whenever pytest collected `editing/` before `base/` (e.g. when only
  passing those two paths). CI's default alphabetical order hid it.
- Extended the local `cv2` type stubs with `warpAffine`, `addWeighted`,
  `flip`, `remap`, `rotate`, and the `BORDER_*` / `ROTATE_*` constants
  used by the new effects.

## 0.33.1

Point-fix cleanup of the AI predictor classes. No new abstractions —
just removing six drifts the broader DESIGN.md §4 mixin was meant to
unify, but at a fraction of the churn.

### Changes

- `TextToSpeech._init_model` renamed to `_init_local` for parity with
  the other predictors.
- `SemanticSceneDetector._load_model` renamed to `_init_local`, and
  the `_device` field is now public `device` (matches every other
  predictor in `ai/`).
- `unload()` added to `AudioClassifier`, `TextToMusic`, `TextToImage`,
  `TextToVideo`, `ImageToVideo`, `SemanticSceneDetector` (previously
  missing — these leaked VRAM across pipeline stages in low-memory
  dubbing).
- `SceneVLM.unload` now goes through `release_device_memory` instead of
  open-coding `gc.collect()` + `torch.cuda.empty_cache()`.
- `TextToMusic`, `TextToVideo`, `ImageToVideo` no longer carry a
  redundant `_device` field alongside `self.device`.
- `MarianTranslator` device-init log line corrected from
  `"TextTranslator"` (the old class name) to `"MarianTranslator"`.

These renames are technically breaking for any external code that
subclassed the predictors and overrode `_init_model` / `_load_model`,
or read `SemanticSceneDetector._device` directly. All three surfaces
were underscore-prefixed (internal).

## 0.33.0

`videopython.ai` refactor. The 8.5k-LOC `ai/` subtree had two god
modules (`video_analysis.py` at 1181 LOC, `dubbing/pipeline.py` at
908 LOC), nine analysis/dubbing dataclasses with hand-rolled
`to_dict`/`from_dict`/`save`/`load` plumbing, and a nine-kwarg
configuration block duplicated across `VideoDubber` and
`LocalDubbingPipeline`. This release splits the god modules,
Pydantic-izes the result models, and consolidates the dubbing knobs
into a single `DubbingConfig`.

**This is a breaking release on the dubbing/analysis result APIs.**
The flat kwargs constructors on `VideoDubber` and
`LocalDubbingPipeline` still work, so dubbing call sites don't need
to change.

### Breaking changes

- **Hand-rolled `to_dict` / `from_dict` / `to_json` / `from_json`
  removed from the analysis and dubbing result models.** Use
  Pydantic's `model_dump` / `model_validate` /
  `model_dump_json` / `model_validate_json` instead. Affected:
  `GeoMetadata`, `VideoAnalysisSource`, `AnalysisRunInfo`,
  `VideoAnalysisConfig`, `AudioAnalysisSection`,
  `SceneAnalysisSample`, `SceneAnalysisSection`, `VideoAnalysis`,
  `Expressiveness`, `TranslatedSegment`, `TimingSummary`,
  `DubbingResult`, `RevoiceResult`, `TranscriptQuality`.
  `VideoAnalysis.save()` / `VideoAnalysis.load()` are kept as
  convenience file-I/O wrappers around `model_dump_json` /
  `model_validate_json`.
- **`videopython.ai.video_analysis` is now a package, not a module.**
  All public re-exports (`VideoAnalysis`, `VideoAnalysisConfig`,
  `VideoAnalyzer`, `SCENE_VLM`, `SEMANTIC_SCENE_DETECTOR`, etc.)
  resolve unchanged. The private `_phash_dedup_frames` and
  `_sample_timestamps` helpers move to
  `videopython.ai.video_analysis.sampling` as `phash_dedup_frames`
  and `sample_timestamps`.
- **`videopython.ai.dubbing.pipeline` private helpers moved.**
  `_peak_match` and `_loudness_match` are now
  `videopython.ai.dubbing.loudness.peak_match` /
  `loudness_match`. `_rms` and `_expressiveness_for` are now
  `videopython.ai.dubbing.expressiveness.rms` /
  `expressiveness_for`. The three private voice-sample methods on
  `LocalDubbingPipeline` (`_extract_voice_samples`,
  `_pick_voice_segment`, `_score_voice_segment`) become free
  functions in `videopython.ai.dubbing.voice_sample` (`extract`,
  `pick`, `score`).
- **`VideoDubber.<knob>` and `LocalDubbingPipeline.<knob>`
  attributes moved onto the new `DubbingConfig`.** Reads of
  `dubber.device`, `dubber.low_memory`, `dubber.whisper_model`,
  `dubber.strict_quality`, `dubber.translator`,
  `dubber.condition_on_previous_text`, `dubber.no_speech_threshold`,
  `dubber.logprob_threshold`, `dubber.vocabulary` now go through
  `dubber.config.<knob>`. Same shape on
  `LocalDubbingPipeline`. **Constructor calls are unchanged** —
  `VideoDubber(device="cpu", low_memory=True, ...)` still works
  and builds a `DubbingConfig` internally. Callers can also pass
  `VideoDubber(config=DubbingConfig(...))` explicitly.

### Internal

- `ai/video_analysis/` is now four sibling modules: `models.py`
  (Pydantic result models), `sampling.py` (frame-sampling profile
  + `phash_dedup_frames` / `sample_timestamps`), `stages.py` (the
  five stage runners + `record_stage` helpers as free functions of
  `(config, source_path, video, ...)`), `analyzer.py`
  (`VideoAnalyzer` orchestration only).
- `dubbing/loudness.py`, `dubbing/expressiveness.py`, and
  `dubbing/voice_sample.py` carry the extracted pure functions.
  `LocalDubbingPipeline.process` and `.revoice` share a new
  `_finalise_audio` helper that consolidates their duplicated
  background/loudness-match tail.
- `dubbing/config.py` adds `DubbingConfig` (Pydantic), carrying
  the nine knobs that used to be duplicated across `VideoDubber`,
  `LocalDubbingPipeline`, and their init-log lines. Adding a knob
  now costs one edit.
- Nested non-Pydantic types (`Transcription`, `SceneDescription`,
  `AudioClassification`, `FaceTrack`, `TranscriptionSegment` from
  `videopython.base`) interop with the new Pydantic models via
  `Annotated[..., BeforeValidator, PlainSerializer]` field hooks
  that delegate to their existing `to_dict` / `from_dict`. JSON
  wire format is identical to 0.32.0 (verified by round-tripping a
  pre-refactor `VideoAnalysis` JSON snapshot with zero key drift).

## 0.32.0

Package layout refactor. `videopython.base` had grown into a 5.5k-LOC
kitchen sink holding data containers, every editing primitive, audio,
text rendering, and scene detection. This release splits it into four
top-level subpackages with a clear dependency direction: `base` →
nothing, `audio` → `base`, `editing` → `base` + `audio`, `ai` → all
three.

**This is a breaking release.** Import paths change and no
backwards-compatibility shims are provided.

### Breaking changes

- **`videopython.base.audio` → `videopython.audio`.** Promoted to a
  top-level subpackage. `Audio`, `AudioMetadata`, `AudioLevels`,
  `SilentSegment`, `AudioSegment`, `AudioSegmentType` now live there.
- **Editing primitives moved out of `videopython.base` into
  `videopython.editing`.** `Operation`, `Effect`, `TimeRange`,
  `OpCategory`, `FilterCtx`, all transforms (`CutSeconds`,
  `CutFrames`, `Resize`, `ResampleFPS`, `Crop`, `SpeedChange`,
  `Reverse`, `FreezeFrame`, `SilenceRemoval`), all effects (`Blur`,
  `Zoom`, `ColorGrading`, `Vignette`, `KenBurns`, `FullImageOverlay`,
  `Fade`, `VolumeAdjust`, `TextOverlay`), and `TranscriptionOverlay`
  are imported from `videopython.editing`.
- **`videopython.base.text` package removed.** `Transcription`,
  `TranscriptionSegment`, `TranscriptionWord`, `ImageText`,
  `TextAlign`, and `AnchorPoint` are now imported directly from
  `videopython.base`. `TranscriptionOverlay` moves to
  `videopython.editing` (it's an `Effect`).
- **`videopython.base.SceneDetector` removed.** The histogram-based
  detector duplicated `videopython.ai.SemanticSceneDetector` at lower
  quality. Use `SemanticSceneDetector` instead. `SceneBoundary` (the
  result dataclass) stays in `videopython.base`.

### Migration

```python
# Before (0.31.x)
from videopython.base import (
    Audio, AudioMetadata,
    Operation, Effect, TimeRange,
    CutSeconds, Resize, Blur, Fade,
    VideoEdit,
)
from videopython.base.text import Transcription, ImageText
from videopython.base.scene import SceneDetector

# After (0.32.0)
from videopython.base import Transcription, ImageText
from videopython.audio import Audio, AudioMetadata
from videopython.editing import (
    Operation, Effect, TimeRange,
    CutSeconds, Resize, Blur, Fade,
    VideoEdit,
)
from videopython.ai import SemanticSceneDetector  # replaces SceneDetector
```

### Internal

- Tests mirror the new package tree: `src/tests/audio/`,
  `src/tests/editing/`, `src/tests/base/`. The import-isolation check
  is parametrized over `base`/`audio`/`editing` at
  `src/tests/test_import_isolation.py`.
- CI now runs `pytest --ignore=src/tests/ai` so the moved audio and
  editing tests are exercised on every push.
- README, full `docs/` tree, and every mkdocstrings directive point
  at the new canonical homes.

## 0.31.3

Internal refactor: `ImageText` (the PIL rendering primitive) moves
out of `base/text/overlay.py` into its own module
`base/text/image_text.py`. `overlay.py` keeps only the
`TranscriptionOverlay` subtitle effect that consumes it. Public
imports through `videopython.base` and `videopython.base.text` are
unchanged.

### Changed

- New module `videopython.base.text.image_text` owns `ImageText`,
  `TextAlign`, and `AnchorPoint`. The package `__init__` re-exports
  them so existing imports (`from videopython.base import ImageText`,
  `from videopython.base.text import ImageText`) keep working.
- `videopython.base.text.overlay` shrank from ~1090 LOC to ~160 LOC
  and now contains only `TranscriptionOverlay`.

## 0.31.2

Internal refactor: `Video.from_path` and `Video.save` are now thin
wrappers around plain functions in a new internal `base/_video_io.py`
module. The `Video` data class is back to being a frame/audio
container (~150 LOC) instead of a 440-LOC mix of state, decoder, and
encoder. No public API changes.

### Changed

- New internal module `videopython.base._video_io` holds
  `decode_video` and `encode_video`. `Video.from_path` and
  `Video.save` delegate to them.
- `ALLOWED_VIDEO_FORMATS` and `ALLOWED_VIDEO_PRESETS` are now defined
  in `_video_io` and re-exported from `videopython.base.video` —
  existing import paths keep working.

## 0.31.1

Internal refactor: ffmpeg/ffprobe subprocess plumbing and dimension
math are now centralised in two new modules. No public API changes;
behaviour is preserved aside from a few latent-bug fixes called out
below.

### Added

- `videopython.base.exceptions.FFmpegError` with `FFmpegProbeError`
  and `FFmpegRunError` subclasses. Existing public exceptions
  (`VideoMetadataError`, `AudioLoadError`, `VideoLoadError`,
  `RemuxError`) still wrap them at call sites — no external catch
  blocks need to change.

### Changed

- All ffprobe invocations (`VideoMetadata`, `Audio`,
  `VideoAnalyzer._extract_source_tags`) share one helper.
- All blocking ffmpeg runs (`concat_files`,
  `replace_audio_stream`, `replace_audio_stream_from_audio`) share
  one helper that pipes optional stdin bytes.
- All streaming ffmpeg decode/encode lifecycles (`FrameIterator`,
  `extract_frames_at_indices`, `Video.from_path`, `FrameEncoder`,
  `Video.save`) share `popen_decode` / `popen_encode` context
  managers that own the `Popen` + terminate/kill cleanup.
- `round_to_even` / `floor_to_even` replace three divergent
  "round to even" helpers (`_round_dimension_to_even`, `_make_even`,
  and an inline `(cw // 2) * 2`).
- `Video.save` and `FrameEncoder` now raise `FFmpegRunError` instead
  of `RuntimeError` on non-zero ffmpeg exit. `concat_files` likewise
  raises `FFmpegRunError`.

### Fixed

- `FrameIterator` previously opened ffmpeg's stderr as a pipe but
  never drained it, risking a buffer-fill deadlock on chatty input.
  Stderr is now `DEVNULL`.
- `extract_frames_at_indices`'s cleanup `wait()` had no timeout and
  could hang indefinitely; cleanup now caps at 5s with a kill
  fallback.
- `FrameEncoder`'s post-stdin-close wait was capped at 30s, which
  could prematurely kill long encodes. The cap is removed to match
  `Video.save`'s unbounded wait.

## 0.31.0

Operation Unification — single sweep, breaking. Every editing primitive
is now an `Operation` subclass — a Pydantic `BaseModel` whose fields ARE
the JSON wire format. The class is the single source of truth: schema,
validation, and (de)serialisation are free; subclasses just declare
fields and implement `apply` / `predict_metadata`.

### Added

- New `videopython.base.Operation` Pydantic foundation: auto-registry
  via `__pydantic_init_subclass__`, discriminated-union schema via
  `Operation.json_schema()`, `Operation.registry()`, `Operation.get()`.
- `Effect(Operation)` with a `window: TimeRange | None` field —
  effect-time ranges are no longer a separate `apply: {start, stop}`
  slot.
- `videopython.base.TimeRange`, `videopython.base.OpCategory`,
  `videopython.base.FilterCtx` are now public.
- Context injection on the runner: ops declare
  `requires: ClassVar[tuple[str, ...]] = ("transcription",)`, the
  `VideoEdit` runner pulls matching keys from `context` and threads them
  into `apply` / `predict_metadata`. Replaces the
  `requires_transcript` registry tag.

### Changed

- **Breaking:** `VideoEdit` wire format flattens. `op` is inline per
  operation, fields are hoisted (no more nested `args` / `apply`),
  segments carry a single `operations` list (no more
  `transforms` / `effects` split), and top-level
  `post_transforms` / `post_effects` collapse into `post_operations`.

  Old (`0.30.x`):

  ```json
  {"op": "blur_effect",
   "args": {"mode": "constant", "iterations": 5},
   "apply": {"start": 1, "stop": 3}}
  ```

  New (`0.31.0`):

  ```json
  {"op": "blur_effect", "mode": "constant", "iterations": 5,
   "window": {"start": 1, "stop": 3}}
  ```

- **Breaking:** `Operation` subclasses construct from keyword args only,
  positional args no longer work for the Pydantic shape. Most
  call-sites already use kwargs; double-check inline construction.
- `Crop.predict_metadata` now mirrors `apply`'s odd-dim center-crop
  rounding (`(cw // 2) * 2`). Fixes a long-standing
  validation-vs-runtime divergence for odd target dimensions.

### Removed

- **Breaking:** `videopython.base.registry` is gone. `OperationCategory`,
  `OperationSpec`, `ParamSpec`, `register`, `get_operation_spec`,
  `get_operation_specs`, `get_specs_by_category`, `get_specs_by_tag`,
  `spec_from_class` are no longer importable. Use
  `Operation.registry()`, `Operation.get()`, and `Operation.json_schema()`.
- **Breaking:** `videopython.base.transitions` deleted —
  `Transition`, `InstantTransition`, `FadeTransition`, `BlurTransition`.
  Their 2→1 shape doesn't fit the single-input `Operation` contract.
  Multi-source ops will return in a future minor with a coherent
  redesign; revive from git history if you need them sooner.
- **Breaking:** `videopython.editing.multicam` deleted —
  `MultiCamEdit`, `CutPoint`. Same reasoning as transitions.
- **Breaking:** `PictureInPicture` and `SplitScreenComposite` deleted —
  no live consumers and a multi-source shape that fights the single-input
  contract.
- **Breaking:** `VideoMetadata` fluent prediction methods deleted:
  `cut`, `cut_frames`, `resize`, `crop`, `resample_fps`, `speed_change`,
  `reverse`, `freeze_frame`, `silence_removal`, `crop_to_aspect_even`,
  `transition_to`, `can_be_merged_with`. Use
  `Operation.predict_metadata(meta)` on the matching op instead. The
  inner builders `with_duration`, `with_dimensions`, `with_fps` stay.
- **Breaking:** `videopython.base.InsufficientDurationError` and
  `videopython.base.IncompatibleVideoError` removed — no live consumers.
- **Breaking:** Transitional aliases `Transformation = Operation` and
  `AudioEffect = Effect` removed. Use `Operation` and `Effect` directly.
- `videopython.ai.registry` deleted — AI ops now auto-register via
  `Operation.__pydantic_init_subclass__` on `import videopython.ai`.

### Migration

Most user code only needs the wire-format update. For programmatic use,
replace fluent metadata calls with the corresponding op:

```python
# Before
new_meta = meta.cut(0, 5).resize(width=1280)

# After
from videopython.base import CutSeconds, Resize
new_meta = CutSeconds(start=0, end=5).predict_metadata(meta)
new_meta = Resize(width=1280).predict_metadata(new_meta)
```

For JSON plans, flatten op shapes and merge `transforms`/`effects` into
`operations`. See `docs/api/editing.md` for the new schema and
`docs/api/operations.md` for the `Operation` base contract.

## 0.30.0

### Removed

- Removed a lot of unused/overcomplicated/dead code:
- **Breaking:** `to_premiere_xml` (Premiere/FCP7 XML export). `videopython.editing.to_premiere_xml` and `videopython.editing.premiere_xml` deleted. Known correctness bug + unclear demand.
- **Breaking:** `ObjectSwapper` and the `videopython.ai.swapping` subpackage (SAM2 + GroundingDINO + LaMa inpainter). `videopython.ai.ObjectSwapper` no longer re-exported. SAM2/GroundingDINO ride `transformers`; no `[ai]` extra change.
- **Breaking:** `StackVideos` and `videopython.base.combine` deleted. `stack_videos` registry entry gone.
- **Breaking:** Dub cache. `cache_dir` kwarg removed from `VideoDubber` and `LocalDubbingPipeline`. `DubCache` and `dub_cache_clear` no longer importable from `videopython.ai.dubbing`. Resume-after-crash is no longer supported.
- **Breaking:** `videopython.base.progress` module deleted. `configure`, `set_verbose`, `set_progress` no longer re-exported from `videopython.base`. Call sites now use stdlib `logging` (configure via `logging.basicConfig`) and `tqdm` directly (suppress with `TQDM_DISABLE=1`); progress bars are now on by default.
- Dead `lead_room` parameter on `FaceTrackingCrop`.

### Changed

- Narrowed 12 bare `except Exception:` clauses in `videopython.ai.video_analysis` and one in `videopython.base.streaming` to specific runtime exceptions so unexpected programming errors propagate.
- Replaced stray `print()` calls in `FaceTrackingCrop` and `SplitScreenComposite` with `logger.info`.

## 0.29.1

### Added

- `vocabulary: list[str] | None` kwarg on `AudioToText`, `VideoDubber`, and `LocalDubbingPipeline`. Forwarded to Whisper as `initial_prompt` to bias the first-window decoder toward stylized brand and proper-noun spellings. Per-call override on `AudioToText.transcribe()`. Recovers near-mishears; not a hotword decoder. Plumbs the API surface for the brand-name-recognition roadmap.

### Changed

- `DubCache.SCHEMA_VERSION` 1 → 2; the transcription cache key now includes vocabulary. Pre-0.29.1 transcription artifacts re-run once on next cached call; translation and TTS artifacts survive.

## 0.29.0

### Added

- `SceneDescription` dataclass on `videopython.base.description` (caption + open-list `subjects` + closed-enum `shot_type`). `SceneVLM.analyze_scene` and `analyze_frame` now return `SceneDescription` instead of a free-form string. Few-shot JSON prompting + tolerant parse + one retry; the final fallback returns the raw text inside the caption field so a scene always gets *something*.
- `SceneVLM.model_size` is now `Literal["4b", "9b", "27b"]` (Qwen3.5-4B/9B/27B). `"4b"` is the default. The previous `"2b"` value is gone — 2B was not reliable enough for structured JSON output, and shipping a tier with a different return shape was an API smell. Constructor logs a loud WARNING when `"27b"` is requested with less than ~45 GB free VRAM (does not raise — knowledgeable users may run with their own quantization layer).
- `SceneVLM.unload()` for `low_memory` parity with the Marian/Qwen translator backends and other unloadable stages.
- `VideoAnalyzer(sampling="low" | "medium" | "high")` kwarg controls the per-scene SceneVLM frame budget. `"low"` is a fast preview pass for long videos, `"high"` keeps talking-head depth, `"medium"` is the previous default. Replaces the four `_SCENE_VLM_*` module constants with a `_SAMPLING_PRESETS` table.
- Perceptual-hash dedup inside `_run_scene_vlm_batched` — talking-head shots collapse to 1-2 frames so action shots keep their budget. Hard floor of one frame survives per group. Threshold lives in `PHASH_DEDUP_DISTANCE` (module constant).
- New `videopython.ai.understanding.faces` module with `FaceTracker.track_shot(frames, frame_indices)` that returns a list of per-shot `FaceTrack` objects (id-stable within a scene via IoU association, no embedding re-id). New `face_tracker` analyzer id wired into `VideoAnalyzer`; results land on `SceneAnalysisSample.faces`.
- `FaceTrack` dataclass on `videopython.base.description`.
- `SceneDescription` and `FaceTrack` are also re-exported from `videopython.base`.

### Changed

- **Breaking:** `SceneAnalysisSample.caption: str | None` renamed to `scene_description: SceneDescription | None`. `to_dict` / `from_dict` updated. Persisted JSON from 0.28.x will no longer round-trip.
- **Breaking:** `FaceTracker` import path moved from `videopython.ai.transforms` to `videopython.ai.understanding.faces`. `videopython.ai.FaceTracker` still works (re-exported from `videopython/ai/__init__.py`). The internal `_FaceDetectionBackend` was renamed to `_FaceDetector`.
- `qwen-vl-utils` is now a pinned dependency in the `ai` extra. The previous transitive-import fallback inside `_generate_from_message_batch` was removed — install issues now surface immediately instead of silently using a manual-image-construction shim.

### Removed

- **Breaking:** `ActionRecognizer` (VideoMAE/Kinetics-400) deleted from `videopython.ai.understanding.temporal`. It was never wired into `VideoAnalyzer`. Direct importers must remove the call.
- **Breaking:** `DetectedAction` dataclass deleted from `videopython.base.description`. Nothing in the codebase produced or consumed it after `ActionRecognizer` was removed.
- **Breaking:** `easyocr` removed from the `ai` extra. The dependency had zero source references; the VLM does OCR inline.

### Notes

- All six pieces of M5 ship together in this 0.29.0 minor bump. No staged patch releases, no back-compat shims.
- `model_size="9b"` weights are ~18 GB FP16, fitting a 24 GB GPU only when SceneVLM runs solo. The analyzer already releases Whisper/TransNetV2 GPU memory before SceneVLM loads (`gc.collect()` + `torch.cuda.empty_cache()`).
- A caption-quality A/B harness lives at `scripts/eval_scene_vlm.py` for by-eye review across `model_size` and `sampling` combinations. Manual run, not a CI gate.

## 0.28.3

### Added

- `Expressiveness` dataclass exposed from `videopython.ai.dubbing` carrying the three Chatterbox `generate()` knobs (`exaggeration`, `cfg_weight`, `temperature`) as `Optional[float]`. `None` means "don't pass the kwarg, use Chatterbox's default". Same three knobs added as optional kwargs on `TextToSpeech.generate_audio`.

### Changed

- Dubbing pipeline now derives a per-segment `Expressiveness` from source vocals RMS and forwards it to Chatterbox, so the dub tracks the source's loud/quiet shape instead of using flat defaults. Three buckets: <0.7× baseline → `exaggeration=0.3, cfg_weight=0.7`; >1.3× → `exaggeration=0.85, cfg_weight=0.35`; in between → Chatterbox defaults. Knob values picked by-ear on `cam1_1min.mp4`. Pure numpy, no new dep.
- `DubCache.tts_key` folds the three knobs into the hash. Pre-0.28.3 cache entries miss on first hit and re-synthesize; old WAVs stay on disk until `dub_cache_clear`. All-`None` profile hashes the same as absent kwargs.

## 0.28.2

### Added

- `cache_dir: str | Path | None = None` constructor kwarg on `VideoDubber` and `LocalDubbingPipeline`. When set, transcription, translated segments, and per-segment TTS WAVs are persisted under that directory and skipped on subsequent runs whose hash inputs match. Designed for resuming crashed long runs and iterating on dub configuration without re-paying transcription cost. Cache invalidates conservatively on whisper/translator/voice-sample/text changes — false misses are cheap; false hits would be bugs. Cache grows unbounded; clear with the new `dub_cache_clear` helper.
- `videopython.ai.dubbing.cache.DubCache` and `dub_cache_clear()` exported from `videopython.ai.dubbing`. `DubCache` exposes the three artifact getters/setters and the key-derivation static methods so external callers can pre-warm or inspect cache state.
- `keep_original_audio: bool = False` kwarg on `VideoDubber.dub_file` and the two `videopython.ai.dubbing.remux` helpers (`replace_audio_stream`, `replace_audio_stream_from_audio`). When True, the source audio is retained as a secondary audio track in the output (dubbed audio remains the default playback track). Useful for editorial A/B.
- `TranslatedSegment.to_dict` / `from_dict` for round-tripping translations through the dub cache. Symmetric with `Transcription` and `TimingSummary`.

### Changed

- Replaced the post-mix peak-amplitude match with BS.1770 integrated-loudness matching via `pyloudnorm` (BSD-3, added to the `ai` extra). The dubbed track now lands within ~1 LU of the source on dialogue-heavy mixes, fixing the perceptually-thinner output that peak ratio left behind. Falls back to peak-match for clips shorter than the BS.1770 gating block (400 ms) or when measurement returns -inf. Post-gain peaks are clamped to 0.99 to keep BS.1770's lack of a peak ceiling from clipping quiet sources.
- `replace_audio_stream` and `replace_audio_stream_from_audio` now carry subtitle streams from the source video through stream-copy by default (`-map 0:s? -c:s copy`). Sources without subtitles are tolerated by the `?` modifier, so existing fixtures aren't disturbed.

## 0.28.1

### Added

- `Qwen3Translator` translation backend (`videopython.ai.generation.qwen3.Qwen3Translator`). Uses Qwen3-4B-Instruct-2507 via `llama-cpp-python` (Q4_K_M GGUF, ~2.4 GB; downloaded on first use to the standard HuggingFace cache). Produces context-aware, length-budgeted translation: the prompt includes a per-segment `target_chars` budget derived from source duration × language-specific speech rate, and an optional `low_confidence` flag for segments whose `avg_logprob` is unhealthy. Output is JSON-line, with parse retry and optional per-segment Marian fallback when both Qwen attempts fail. Apache-2.0 model + MIT inference framework. Adds `llama-cpp-python>=0.3` to the `ai` extra.
- `TranslationBackend` runtime-checkable Protocol (`videopython.ai.generation.translation.TranslationBackend`). Captures the three-method contract `translate_segments` / `unload` / `get_supported_languages`. Both `MarianTranslator` and `Qwen3Translator` satisfy it; the pipeline depends only on the protocol.
- `translator: Literal["auto", "marian", "qwen3"] = "auto"` constructor kwarg on `VideoDubber` and `LocalDubbingPipeline`. The `"auto"` resolver picks based on language coverage **and** device: GPU + Qwen-supported pair → Qwen3 (best quality); Marian-supported pair → Marian (~10-15× faster on CPU); CPU + Qwen-only pair → Qwen3 with a loud WARNING. The integration spike on `cam1_1min.mp4` (Polish→Spanish) showed Qwen3-4B Q4_K_M on CPU runs ~2.8× the wall time of Marian end-to-end (and ~13× on the translation stage in isolation), so the auto resolver intentionally prefers Marian on CPU even when Qwen would also cover the pair.
- `UnsupportedLanguageError(ValueError)` exposed from `videopython.ai.dubbing`. Carries `source_lang` and `target_lang` so callers can introspect the unsupported pair without parsing the message. Raised by the auto resolver when neither backend covers the pair.
- `translation_failures: list[int]` field on `DubbingResult`. Populated by `Qwen3Translator` for segments where both the primary call and the per-segment Marian fallback failed — those segments land on the result with `translated_text=""`. Empty list under `MarianTranslator`.

### Changed

- `_transcribe_with_diarization` now re-attaches per-segment Whisper confidence (`avg_logprob`, `no_speech_prob`, `compression_ratio`) to the diarization-rebuilt segments by max-overlap match. Without this fix the segments-from-words rebuild dropped the metadata that 0.28.0 plumbed through, so on every diarized run the M1.3 confidence fields were `None`. M1.5's logprob-based reject rule and any prompt that branches on confidence now have signal on the diarized path too.
- `TextTranslator` is renamed to `MarianTranslator`. The old name remains as a back-compat alias through 0.28.x; remove in 0.30.0. Existing imports continue to work unchanged.

### Known limitations

- Qwen3 on CPU is impractical for everyday use — wall time on a 60-second source ran ~25 minutes (vs ~9 minutes for Marian) in the integration spike. The auto resolver prevents accidental opt-in; explicit `translator="qwen3"` on CPU triggers a startup WARNING.
- On the same spike, Qwen3 produced clearly better translations on idiomatic and question-vs-statement segments but **regressed slightly on truncation rate** (worst-case segment lost 2.18 s of content vs Marian's 0.86 s) and **mistranslated a single-word segment** (`narzekamy` → `mejoramos`, opposite meaning). The prompt's character-budget instruction isn't strict enough on long segments, and very short segments give the LLM too little anchor to disambiguate. Both will be revisited as follow-ups; for now, the auto resolver's CPU preference for Marian is the practical mitigation.

## 0.28.0

### Added

- `transcript_quality` field on `DubbingResult`. A cheap heuristic over the Whisper transcription, exposed as `TranscriptQuality(recommendation, dominant_phrase, dominant_phrase_fraction, median_avg_logprob, speech_fraction, flags)`. Three checks fire flags: a single phrase covers ≥70% of segment chars (the YouTube-outro `「ご視聴ありがとうございました」` cascade); median per-segment `avg_logprob` falls below `-1.5`; speech-region duration is <5% of the clip's wall-clock duration on inputs >30s. The recommendation is `"reject"` when the dominance flag fires together with at least one other flag, `"warn"` when any single flag fires, `"ok"` otherwise — single repetition alone (chants, song lyrics) only warns.
- `strict_quality: bool = False` constructor kwarg on `VideoDubber` and `LocalDubbingPipeline`. With `strict_quality=True`, a `"reject"` raises `GarbageTranscriptError` *before* Demucs/translation/TTS run, attaching the triggering `TranscriptQuality` as `error.quality` for caller introspection. Default `False` matches existing behaviour; reject-graded transcripts log a WARNING and processing continues. Either way the assessment is exposed on `DubbingResult` for inspection.
- `timing_summary: TimingSummary | None` field on `DubbingResult`. Aggregates the per-segment `TimingAdjustment` list that `synchronize_segments` was already producing but the pipeline previously discarded: `total_segments`, `clean_count` (speed factor in [0.99, 1.01]), `stretched_count`, `truncated_count`, `mean_speed_factor`, `max_truncation_seconds`. Truncation/clamp counts are quality red flags — high values mean translation produced text too long for the source duration.
- Optional confidence fields on `TranscriptionSegment`: `avg_logprob`, `no_speech_prob`, `compression_ratio` (all `float | None`, default `None`). Populated from raw Whisper segment dicts in `AudioToText._process_transcription_result`. Existing constructions (positional or kwargs) keep working since the fields all have defaults; `to_dict`/`from_dict` round-trip them and accept old persisted JSON without the keys.
- Optional `progress_callback: Callable[[float], None]` on `TextTranslator.translate_batch` and `translate_segments`. Fires once per batch with a fraction in `[0, 1]` representing translation-stage progress. `LocalDubbingPipeline.process` uses it to map progress onto its `[0.35, 0.50]` overall window and emits `"Translating text (N%)"` instead of sitting silently on `0.35` for minutes during MarianMT runs on long sources.

### Changed

- `LocalDubbingPipeline._extract_voice_samples` now scores candidate segments and rejects clipped slices (peak ≥ 0.99) and slices where Demucs left the background louder than the vocals (`vocal_rms / bg_rms < 1.5`). Segments are scored by `vocal_rms - 0.05 × |duration − 6s|`, and the highest score wins. When every candidate fails, falls back to the longest segment overall and logs a WARNING — previously a speaker with no qualifying segment got silently dropped from `voice_samples` and the dubbing loop fell back to Chatterbox's default voice without any indication. The signature gains a `background_audio: Audio | None` parameter; the dubbing path passes the separated background, the `revoice()` path passes `None` (which silently degrades to "no RMS check"). Existing dubs may pick a different sample for the same input.
- `LocalDubbingPipeline.process` reports finer-grained translation progress (`"Translating text (37%)"` etc.) instead of a single `"Translating text"` event at `0.35`. Downstream consumers that pattern-match on the exact stage string will need updating.

## 0.27.2

### Changed

- `AudioToText` now defaults to `condition_on_previous_text=False` (Whisper's own default is `True`). With conditioning on, a single hallucinated filler phrase cascades through the rest of the file because each window's decoder is primed by the previous window's decoded text — the textbook Whisper-on-noisy-Japanese failure where the YouTube outro `「ご視聴ありがとうございました」` ("Thank you for watching") repeats across every window. The cost on clean audio is small (slightly less context for ambiguous homophones across sentence boundaries); the benefit on degenerate audio is large. Pass `condition_on_previous_text=True` to restore the previous default.

### Added

- `AudioToText` now exposes three Whisper decoder kwargs: `condition_on_previous_text`, `no_speech_threshold` (default `0.6`, matching Whisper), and `logprob_threshold` (default `-1.0`, matching Whisper). The latter two are surfaced for per-job tuning without behaviour change. All three are forwarded to both the plain and the diarization transcribe paths.
- `VideoDubber` and `LocalDubbingPipeline` accept the same three kwargs and forward them through to `AudioToText`, so dubbing benefits without a separate code path.

## 0.27.1

### Changed

- Default Whisper model bumped from `small` to `turbo` across `AudioToText`, `VideoDubber`, and `LocalDubbingPipeline`. `turbo` (a distilled large-v3) gives large-v3-class accuracy at roughly 8x the speed of `large` and ~2x the speed of `small`, with a comparable VRAM footprint to `small`. The transcription pass in the dubbing pipeline gets faster *and* more accurate out of the box. Pass `whisper_model="small"` (or `model_name="small"` on `AudioToText`) to restore the previous default.

## 0.27.0

### Changed

- `AudioToText` now runs Silero VAD before Whisper to gate language detection. Whisper's auto-detect only inspects the first 30s of input; on movies/podcasts/vlogs that open with silence, music, or non-vocal credits, detection used to lock onto the wrong language (typically English) and the rest of the file was decoded as that wrong language. VAD locates voiced regions, builds a 30s mel from concatenated voiced audio, and the detected language is passed into `whisper.transcribe(language=...)` for both the diarization and plain branches. If VAD finds no speech, an empty `Transcription` is returned without invoking Whisper — tighter than the existing energy-based `is_silent` guard which lets through music/noise. Default-on; opt out with `AudioToText(enable_vad=False)`. The `Transcription.language` field now carries the correctly-detected language even on hard inputs (Japanese movies opening with credits etc.). `VideoDubber` benefits transparently — no API changes there.

### Added

- `AudioToText(enable_vad: bool = True)` constructor flag. Mirrors the existing `enable_diarization` precedent. New dependency: `silero-vad>=5.1` (MIT, ~2 MB JIT model, CPU-fast).

## 0.26.10

### Added

- `AnalysisRunInfo.stage_durations_seconds: dict[str, float]` and `AnalysisRunInfo.total_duration_seconds: float | None`. `VideoAnalyzer._analyze` now records per-stage wall-clock times (`whisper`, `scene_detection`, `whisper_and_scene_detection_parallel` when both run together, `scene_analysis`, `scene_vlm`, `audio_classification`) into the run info that ships with every `VideoAnalysis`. Consumers can now persist or aggregate these timings (e.g. via `to_dict()`) to track pipeline performance over time without parsing logs. Note: the per-stage `"<stage> completed in X.XXs"` log lines now use the stage key as the prefix (e.g. `whisper completed in 1.23s` instead of `Whisper transcription completed in 1.23s`); grep/alerting on the old strings will need to be updated.

## 0.26.9

### Changed

- `LocalDubbingPipeline` TTS loop now catches per-segment exceptions, logs the failing segment index/speaker/text at WARNING, and continues. Chatterbox occasionally crashes inside `alignment_stream_analyzer.step` on short translated text (an `IndexError` from a zero-column tensor reduction) — previously one bad segment aborted the entire run. Skipped segments leave a gap where source audio plays through.

## 0.26.8

### Changed

- `TextTranslator.translate_segments` now skips empty and punctuation-only segments (text with fewer than 2 alphanumeric characters) instead of sending them to MarianMT. Whisper routinely emits `" ."`, `"..."`, `"?"`, and single-token segments which the model could hallucinate full sentences from — those hallucinations were then TTS'd into the dubbed track. Skipped segments produce `TranslatedSegment(translated_text="")` so timing/speaker metadata stays parallel to the input list. The pipeline TTS loop also skips empty translated text.
- `LocalDubbingPipeline` now gates Demucs to the speech-bearing portion of the audio. Speech regions are derived from the transcription, merged, and padded by 0.5s for context. Non-speech gaps pass through as background unchanged. On talk-heavy sources with silence/music gaps this roughly halves separation time. When speech covers ≥90% of the track, falls back to full-track separation (the slicing+stitching overhead would exceed the savings on near-continuous speech).
- Final dubbed audio is now peak-matched against the source audio so the dub doesn't land quieter than the original. Demucs background normalization and the timing-assembler peak guard each clamp at 1.0 instead of restoring headroom; without this match the dubbed mix consistently came out perceptually thinner. Applied to both `process()` (full dub) and `revoice()`.

### Added

- `AudioSeparator.separate_regions(audio, regions, full_separation_threshold=0.9)` — separates only the given `(start, end)` regions and passes through the rest as background. Used internally by the dubbing pipeline; available for callers that want region-gated source separation directly.

## 0.26.7

### Changed

- `TimingSynchronizer.assemble_with_timing` now uses a single-pass assembler (one buffer + in-place add per segment) instead of repeatedly calling `Audio.overlay`. The old loop allocated a fresh `np.zeros(total_length)` and copied the entire base track on every segment; cumulative cost was O(N × total_samples) for N segments. For long dubs (thousands of segments at typical sample rates) the assembly stage now runs in roughly constant memory and a fraction of the wall time.
- Dubbing pipeline encodes each speaker's voice sample to a temp WAV exactly once and reuses the path across all of that speaker's segments. Previously `TextToSpeech.generate_audio` re-encoded the same sample on every TTS call. Public TTS API stays compatible: `generate_audio` gains an optional `voice_sample_path` argument that takes precedence over `voice_sample` when set.
- `VideoDubber.dub_file` now streams the dubbed audio directly to ffmpeg via stdin instead of writing it to a temp WAV first. Drops two full-track copies (~10 GB on a 2h dub) from the disk path.

### Added

- `videopython.ai.dubbing.remux.replace_audio_stream_from_audio(video_path, audio, output_path)` — variant of `replace_audio_stream` that accepts an in-memory `Audio` and pipes WAV bytes to ffmpeg's stdin. Used by `dub_file` to skip the disk round-trip.

## 0.26.6

### Added

- `AudioToText.diarize_transcription(audio, transcription)` runs pyannote standalone on the supplied audio and overlays speaker labels onto a pre-computed transcription's words. Useful when callers have a transcription (e.g. pre-computed and edited) but no speakers, and want per-speaker voice cloning without re-running Whisper. Requires word-level timings; rejects SRT-loaded transcriptions.
- Dubbing now diarizes a supplied transcription on demand. When `transcription` has no speakers and `enable_diarization=True`, `LocalDubbingPipeline.process()` runs pyannote standalone and attaches speakers to the supplied words. With `enable_diarization=False`, the supplied transcription is used as-is (single shared voice clone). Speaker labels already present on the supplied transcription always take precedence — `enable_diarization` is logged-and-ignored in that case.

## 0.26.5

### Added

- `VideoDubber(whisper_model=...)` and `LocalDubbingPipeline(whisper_model=...)` expose the Whisper model size (`tiny`, `base`, `small`, `medium`, `large`, `turbo`) used for transcription. Default remains `small`.

### Changed

- Reduced peak memory for dubbing long sources. `AudioSeparator` now keeps the input tensor on CPU and passes `device=` to Demucs' `apply_model` so per-chunk compute runs on GPU while the full output stays in CPU RAM, avoiding GPU OOM on long sources. Voice samples extracted in `LocalDubbingPipeline` are now copied so they no longer keep the full vocals array (~1.3 GB for a 2h source) alive across translation and TTS. The unused `music` stem is no longer computed by the separator (`SeparatedAudio.music` is `None`). In `low_memory=True` mode, the returned `DubbingResult.separated_audio` is now `None` so vocals and background buffers can be released as soon as they're no longer needed.

## 0.26.4

### Changed

- `TextToSpeech` is now Chatterbox Multilingual only. Bark (`suno/bark`, `suno/bark-small`) is removed — it was unmaintained, slower, and lower quality than Chatterbox, and the dubbing pipeline already routed all real traffic through Chatterbox. The `model_size` and `voice_preset` arguments are gone; `voice` now takes an optional `Audio` reference clip instead of a Bark preset string. `voice_sample` on `generate_audio()` remains optional — without one, Chatterbox uses its built-in default speaker. Existing `TextToSpeech().generate_audio("text")` calls continue to work.

## 0.26.3

### Added

- `VideoDubber.dub_file(input_path, output_path, target_lang, ...)` dubs a video file on disk without loading any video frames into Python memory. Extracts audio via ffmpeg, runs the dubbing pipeline on the audio only, then muxes the dubbed audio back into the source video using ffmpeg stream-copy (no video re-encode). Peak memory is bounded by model weights and the audio track, independent of video length and resolution. Use instead of `dub_and_replace` for long or high-resolution sources.
- `videopython.ai.dubbing.remux.replace_audio_stream(video_path, audio_path, output_path)` helper that stream-copies a video track while replacing its audio.

### Changed

- `LocalDubbingPipeline.process()` and `LocalDubbingPipeline.revoice()` now take a `source_audio: Audio` argument instead of `video: Video`. `VideoDubber.dub()`/`revoice()` public signatures are unchanged — they pass `video.audio` at the boundary.

## 0.26.2

### Added

- `VideoDubber(low_memory=True)` unloads each pipeline stage's model (Whisper, Demucs, MarianMT, Chatterbox) after it runs, so only one model is resident at a time. Trades per-run latency for a lower memory ceiling. Recommended for GPUs with <=12GB VRAM or hosts with <32GB RAM. Each model class (`AudioToText`, `AudioSeparator`, `TextTranslator`, `TextToSpeech`) now exposes an `unload()` method that clears cached weights and releases CUDA/MPS allocator cache.

## 0.26.1

### Added

- `VideoDubber.dub()`, `dub_and_replace()`, and `LocalDubbingPipeline.process()` accept an optional `transcription` parameter. When a pre-computed `Transcription` object is provided, the internal Whisper transcription step is skipped, saving time and VRAM.

### Fixed

- `TextTranslator` now resolves correct HuggingFace model names for languages without a direct `opus-mt-en-{lang}` model. Portuguese uses `opus-mt-tc-big-en-pt`, Korean uses `opus-mt-tc-big-en-ko`, Japanese uses `opus-mt-en-jap`, and Polish uses `opus-mt-en-zlw`. Previously these languages failed with a 404 at runtime.

## 0.26.0

### Added

- `VideoEdit.run_to_file(output_path)` streams frames directly from ffmpeg decode through per-frame effect processing to ffmpeg encode, keeping only one frame in memory at a time. Memory usage is now constant regardless of video length (264 MB for any duration, vs 9+ GB previously for a 1-minute 720p video). Falls back to eager mode automatically for non-streamable operations.
- Streaming interface on `Effect`: `supports_streaming`, `streaming_init()`, `process_frame()`. All built-in effects support streaming: `ColorGrading`, `Blur`, `Zoom`, `Vignette`, `KenBurns`, `Fade`, `FullImageOverlay`, `TextOverlay`, `VolumeAdjust`.
- `FrameIterator` now accepts `-vf` filter parameters for applying ffmpeg filters (scale, crop, fps) during decode.
- `FrameEncoder` class for writing raw frames to ffmpeg encode via stdin pipe.
- `stream_segment()` and `concat_files()` for low-level streaming pipeline control.
- Streamable transform compilation: `Resize`, `Crop`, `ResampleFPS`, and constant `SpeedChange` compile to ffmpeg `-vf` filters for zero-memory transform application.

## 0.25.8

### Changed

- Effects now mutate frames in-place instead of creating full array copies. Affected effects: `ColorGrading`, `Blur`, `Zoom`, `FullImageOverlay`, `KenBurns`. This reduces peak memory by ~39% and speeds up effect application by ~65% on a 1-minute 720p benchmark.
- `Fade` and `Vignette` now process frames in batches instead of allocating a full float32 intermediate array, significantly reducing memory spikes.
- `Effect.apply()` skips the `np.r_` slice-and-reassemble path when the effect covers the full video (no `start`/`stop`), avoiding an unnecessary full-frame copy.
- Dropped `multiprocessing.Pool` from `ColorGrading` and `Blur` in favor of serial in-place processing. The IPC serialization overhead of Pool was slower than direct computation for typical video lengths.

## 0.25.7

### Fixed

- `Transcription.standardize_segments()` now preserves speaker information. Segments are split on speaker changes in addition to time/word-count limits, and each segment's `speaker` field is set from its words. Previously speaker info was dropped during regrouping.

## 0.25.6

### Fixed

- Dubbing pipeline now reinitializes transcriber and TTS models when configuration changes between calls (`enable_diarization`, `voice_clone`, `language`), fixing stale cache bug.
- Multicam timeline duration now uses the shortest source instead of the first source's duration, preventing cuts from exceeding shorter sources.
- Fixed `ImageToVideo` example in README (removed non-existent `fps` parameter).
- Fixed Python version range in docs (`<3.14`, not `<3.13`).

### Changed

- Extracted `_resolve_time_range` helper in effects to deduplicate time validation across `Effect`, `Fade`, and `AudioEffect`.
- Exposed `MultiCamEdit` metadata as public properties (`source_meta`, `source_duration`, `source_metas`); `premiere_xml` no longer reaches into private attributes.
- Extracted `_cut_ranges()` method in `MultiCamEdit` to deduplicate the cut-range loop between `_validate()` and `run()`.
- Marked `Transition._from_dict` as `@abstractmethod` for consistency with `to_dict`.
- Merged duplicate `_coerce_optional_number` functions in `video_edit.py`.
- Removed unused `torchcodec` from base dependencies (was never imported, pulled in torch for all users).
- Removed redundant `[project.optional-dependencies] dev` (contributors use `uv sync --dev`).
- Aligned AI dependency floors with uv override-dependencies (`torch>=2.8`, `diffusers>=0.30`, added `torchaudio>=2.8`).
- Added `editing/` tests and coverage to CI pipeline.

## 0.25.5

### Fixed

- Dubbing pipeline now uses the language detected by Whisper instead of hardcoding English as the source language.
- Translation is skipped when source and target languages are the same, avoiding a crash when trying to load a nonexistent same-language translation model.
- Use explicit `MarianTokenizer`/`MarianMTModel` for translation, fixing compatibility with transformers 5.x (`AutoTokenizer` no longer resolves `MarianConfig`).
- `TimingSynchronizer` no longer crashes on zero-duration target segments (produced by diarization edge cases).

### Changed

- Replaced Coqui TTS (XTTS) with Chatterbox Multilingual (MIT license, 23 languages) for voice cloning. Coqui TTS was unmaintained and incompatible with transformers 5.x.
- Added `sentencepiece` dependency required by the Marian MT tokenizer.
- Speaker diarization is now available as an opt-in parameter (`enable_diarization`) on `VideoDubber.dub()` and `dub_and_replace()`. When enabled, each detected speaker gets their own cloned voice.
- Segments shorter than 100ms are skipped during TTS generation to avoid generating speech for near-empty diarization artifacts.

## 0.25.4

### Added

- `to_premiere_xml(edit)` -- export a `MultiCamEdit` plan to FCP7 XML (xmeml) for direct import into Adobe Premiere Pro. Supports all sources, cuts, source offsets, external audio, and fade transitions (cross dissolve). Available via `from videopython.editing import to_premiere_xml`.

## 0.25.3

### Added

- Python 3.13 support. Updated `requires-python` to `>=3.10, <3.14` and added 3.13 to CI test matrix.

## 0.25.2

### Added

- `MultiCamEdit` now accepts an optional `source_offsets` parameter -- a dict mapping camera names to time offsets in seconds. This allows aligning sources that started recording at different times without pre-trimming files. Fully supported in validation, execution, serialization, and JSON schema.

## 0.25.1

### Added

- `MultiCamEdit.validate()` -- predict output `VideoMetadata` (dimensions, fps, duration) without loading video frames. Accounts for duration consumed by fade/blur transitions.
- `MultiCamEdit.json_schema()` -- returns a JSON Schema for MultiCamEdit plans, suitable for LLM tool definitions or structured-output formats.

## 0.25.0

### Added

- `MultiCamEdit` and `CutPoint` for multicam podcast-style editing -- switch between synchronized camera angles at specified cut points with transitions, and replace audio with an external track.
- `Transition.to_dict()` and `Transition.from_dict()` for JSON serialization of all transition types (`InstantTransition`, `FadeTransition`, `BlurTransition`).
- New `videopython.editing` package grouping all editing plan logic (`VideoEdit`, `SegmentConfig`, `MultiCamEdit`, `CutPoint`).

### Changed

- **Breaking:** `VideoEdit` and `SegmentConfig` moved from `videopython.base.edit` to `videopython.editing`. Update imports from `from videopython.base import VideoEdit` to `from videopython.editing import VideoEdit`.

## 0.24.2

### Added

- `Transcription.to_srt()` -- export transcription as an SRT subtitle string.
- `Transcription.from_srt()` -- parse an SRT string into a Transcription (one segment per subtitle block; word-level timing is not available in SRT).
- `Transcription.save_srt()` -- write transcription directly to an SRT file.

## 0.24.1

### Fixed

- Fixed `AudioEffect.apply()` and `Fade.apply()` crashing with "Video is only X long, but passed stop: Y" when effect stop time slightly exceeded video duration due to floating-point rounding after segment assembly. Both now clamp start/stop to video duration, matching the existing behavior in `Effect.apply()`.

### Improved

- Updated `Effect.apply()`, `AudioEffect.apply()`, and `Fade.apply()` docstrings to recommend omitting `start`/`stop` for full-range effects instead of passing explicit values. This guidance flows through to the JSON schema descriptions used by LLM integrations.

## 0.24.0

### Added

- `VideoEdit` now automatically matches segments before concatenation with two new parameters (both `True` by default):
  - `match_to_lowest_fps` -- resamples all segments to the lowest fps among them.
  - `match_to_lowest_resolution` -- resizes all segments to the smallest width and height among them.
- Matching is applied at the ffmpeg decode level (first step in the pipeline), avoiding unnecessary memory allocation for full-resolution frames.
- `Video.from_path` now accepts optional `fps`, `width`, and `height` parameters for ffmpeg-level resampling and scaling during decode.
- Set either matching flag to `False` to get the previous strict behavior that raises on mismatched segments.
- Both flags are supported in JSON plans (`match_to_lowest_fps`, `match_to_lowest_resolution`) and only serialized when `False`.

## 0.23.3

### Fixed

- Fixed `RuntimeError: expected scalar type Float but found BFloat16` when Whisper and SceneVLM run concurrently. Removed concurrent SceneVLM preload during Whisper/TransNetV2 phase -- `transformers.from_pretrained(torch_dtype="auto")` mutates the process-global `torch.get_default_dtype()` to BFloat16 during model construction, corrupting Whisper's model weights when initialized in a parallel thread. SceneVLM now loads sequentially after Whisper/TransNetV2 complete.

## 0.23.2

### Fixed

- Fixed `RuntimeError: expected scalar type Float but found BFloat16` when Whisper and SceneVLM run concurrently. `AutoModelForImageTextToText.from_pretrained(torch_dtype="auto")` mutated the global `torch.get_default_dtype()` to BFloat16, causing Whisper's LayerNorm to fail. The default dtype is now saved and restored around model loading.

## 0.23.1

### Fixed

- Switched core dependency from `opencv-python` to `opencv-python-headless` to resolve conflict with `easyocr` (which pulls in `opencv-python-headless`). Both packages provide the `cv2` module and clash when installed side by side. The headless variant works identically for videopython's use case (no GUI bindings needed).

## 0.23.0

### Changed

- SceneVLM upgraded from Qwen3-VL to Qwen 3.5 unified vision-language models (`Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-2B`). Requires `transformers>=5.2.0`.
- Replaced `whisperx` with `openai-whisper` + `pyannote-audio` for transcription and speaker diarization. whisperx pinned `huggingface-hub<1.0.0`, incompatible with transformers 5.x.
- Speaker diarization now uses `pyannote/speaker-diarization-community-1` directly with overlap-based word-speaker assignment.
- Removed `compute_type` parameter from `AudioToText` (was whisperx-specific).

## 0.22.8

### Fixed

- `AudioMetadata.duration_seconds` is now rounded to 4 decimal places, matching `VideoMetadata` convention. Previously unrounded sample-count division caused audio/video duration drift after slicing (e.g. `36.04997916666667s` vs `36.05s`), leading to `end_seconds cannot exceed audio duration` errors during effect application.
- `Audio.slice()` tolerance increased from 1 microsecond to 100ms. Container formats (.mov, .mp4) routinely have audio/video stream duration mismatches of 1-50ms; the old tolerance rejected valid slices. Values within tolerance are clamped, not silently accepted.
- `Effect.apply()` now clamps `start`/`stop` to video duration instead of raising when they slightly exceed it. This handles frame-rounding mismatches where post-effects carry `stop` values from the plan that exceed the assembled video duration after segment cutting (e.g. plan says `stop=36.05` but 746 frames at 20.69fps = 36.0567s video with 36.05s audio).

## 0.22.7

### Improved

- Operation registry now exposes rich JSON Schema constraints (`exclusiveMinimum`, `minimum`, `maximum`, `enum`) for all operations, preventing invalid parameter values (e.g. `zoom_factor <= 1`).
- All operation and parameter descriptions are auto-extracted from class docstrings, making docstrings the single source of truth.
- Refined all operation docstrings to be actionable and LLM-friendly, with concrete guidance on parameter values and their visual effects.

## 0.22.6

### Fixed

- Standardized duration rounding to 4 decimal places across all `VideoMetadata` methods (`from_path`, `from_video`, `cut_frames`, `with_duration`). Previously some paths used 2 decimals, some 4, and some none, causing effect stop times to exceed segment duration in validation.
- Added 1ms tolerance to effect bounds validation to prevent false positives from residual floating-point arithmetic.

## 0.22.5

### Added

- Metadata prediction for `reverse`, `freeze_frame`, and `silence_removal` transforms, enabling `VideoEdit.validate()` and `validate_with_metadata()` to accept plans containing these operations.
- `validate()` and `validate_with_metadata()` now accept an optional `context` parameter to pass runtime data (e.g. transcription) needed by context-dependent operations like `silence_removal`.

## 0.22.4

### Performance

- SceneVLM frames are now downscaled to a 384x384 pixel budget before processing, reducing vision token count and inference time (~5.5x faster VLM inference, device-independent).
- All scene frame timestamps are collected upfront and extracted in a single ffmpeg call instead of one process per scene.
- Adjacent short scenes (< 10s combined) are grouped into a single VLM call to reduce per-call overhead.
- SceneVLM model weights now load in a background thread overlapping with Whisper and scene detection.

### Changed

- `SceneVLM` now accepts a `max_image_pixels` parameter to control the per-image pixel budget (default: 147456). Set to a large value to disable downscaling.
- `SceneVLM` now allows MPS device selection (previously forced CPU).

## 0.22.3

### Fixed

- `Fade` and `VolumeAdjust` audio envelope off-by-one: when audio sample count doesn't perfectly match `round(duration * sample_rate)` (common with real-world files), the envelope could be 1 sample longer than the actual audio slice, causing a NumPy broadcast shape mismatch.

## 0.22.2

### Added

- `VideoEdit.validate_with_metadata()` validates editing plans using pre-built `VideoMetadata` instead of loading from disk. Accepts a single `VideoMetadata` for single-source plans or a `dict[str, VideoMetadata]` for multi-source plans. Eliminates the need for downstream projects to import private validation functions.

## 0.22.1

### Added

- `Transcription` now stores the detected `language` (ISO 639-1 code) from Whisper/WhisperX. The field is propagated through `offset()`, `standardize_segments()`, `slice()`, and serialized via `to_dict()`/`from_dict()`. Defaults to `None` for backwards compatibility.

## 0.22.0

### New Operations

- **Fade** effect: fades video to/from black with synchronized audio fade. Supports `in`, `out`, and `in_out` modes with `sqrt`, `linear`, and `exponential` curves.
- **VolumeAdjust** effect: adjusts audio volume within a time range with optional smooth ramps. Does not affect video frames.
- **TextOverlay** effect: renders text on video frames with configurable position, font size, color, background box, word wrap, and 6 anchor points.
- **Reverse** transform: reverses video frame order with optional audio reversal.
- **FreezeFrame** transform: holds a single frame for a specified duration, inserting it before, after, or replacing frames at the given timestamp.
- **SilenceRemoval** transform: removes or speeds up silent gaps between speech using word-level transcription timing. Supports `cut` and `speed_up` modes.

### Changed

- `Effect.apply()` is no longer `@final`, allowing subclasses like `Fade` and `AudioEffect` to override it when frame-only processing is insufficient.
- `AudioEffect` base class added for effects that modify only audio (inherits from `Effect` for execution engine compatibility).
- `VideoEdit.run()` and `SegmentConfig.process_segment()` accept an optional `context` dict for passing side-channel data (e.g. transcription) to context-dependent operations.

## 0.21.6

### Dependencies

- Bumped `whisperx>=3.8.1` (from `>=3.4.2`) for pyannote-audio 4.x compatibility, fixing `torchaudio.AudioMetaData` and `use_auth_token` errors when speaker diarization is enabled.

## 0.21.5

### Breaking Changes

- Removed `ActionRecognizer` from the `VideoAnalysis` pipeline. The standalone `ActionRecognizer` class remains available for direct use.
- `SceneAnalysisSample.visual_segments` replaced with `SceneAnalysisSample.caption: str | None`. Each scene now produces a single caption instead of a list of per-segment captions.
- Removed `SceneVisualSegment` dataclass.
- `action_recognizer` is no longer a valid analyzer id in `VideoAnalysisConfig.enabled_analyzers`.

### Changed

- SceneVLM now sends all frames for a scene in a single multi-image inference call instead of splitting into 10-second windows with separate calls. Frame count scales with scene duration (1 frame per 5 seconds).
- Whisper transcription and TransNetV2 scene detection now run in parallel when both are enabled.
- After TransNetV2 finishes, global torch deterministic mode and cuDNN benchmark settings are reset to defaults for better SceneVLM throughput.
- Whisper and TransNetV2 GPU memory is explicitly freed before SceneVLM loads.
- Added per-step timing logs throughout the analysis pipeline and inside SceneVLM (model load and inference time with image count).

## 0.21.4

### Breaking Changes

- `SceneVLM.analyze_scene()` and `SceneVLM.analyze_frame()` now return `str` (plain-text caption) instead of `SceneVLMResult`.
- Removed `SceneVLMResult` dataclass.

### Changed

- SceneVLM now generates plain-text captions instead of structured JSON. Removes fragile JSON parsing that failed on most segments due to Qwen3 thinking mode output.
- Disabled Qwen3 thinking mode (`enable_thinking=False`) to avoid `<think>` token overhead.

## 0.21.3

### Fixed

- Removed unsupported `use_model_defaults` kwarg from SceneVLM generation call that caused `ValueError` with transformers versions that don't recognize it.

## 0.21.2

### Fixed

- Added logging warnings to all bare `except` blocks in `VideoAnalyzer` so failures in transcription, scene detection, VLM, action recognition, and audio classification are no longer silently swallowed.

## 0.21.1

### Changed

- `VideoAnalysisConfig` now accepts `analyzer_params` to configure individual predictors (model size, device, thresholds, etc.) without subclassing or patching internals.
- Removed hardcoded `_SCENE_VLM_MODEL_SIZE` override in `VideoAnalyzer`; `SceneVLM` uses its own default unless overridden via `analyzer_params`.

## 0.21.0

### Breaking Changes

- `VideoAnalysis` was rewritten to a scene-first schema focused on `audio.transcription` and `scenes.samples`.
- Removed legacy `VideoAnalysis` payload/flow elements, including filtering-focused exports, motion outputs, and per-step timing/status data.
- `VideoAnalysisConfig` is now minimal and primarily driven by `enabled_analyzers`.
- Removed understanding APIs from public exports:
  - `ImageToText`
  - `ObjectDetector`
  - `FaceDetector`
  - `TextDetector`
  - `CameraMotionDetector`
  - `MotionAnalyzer`
- Removed modules:
  - `videopython.ai.understanding.detection`
  - `videopython.ai.understanding.motion`

### Changed

- Added `SceneVLM` (Qwen3-VL) as the primary visual-understanding component, with structured JSON output parsing/validation.
- `VideoAnalyzer` now runs global transcription/scene detection, then scene-level analysis with chunked visual segments (`_SCENE_VLM_MAX_SEGMENT_SECONDS`).
- Device selection behavior was tightened:
  - `SceneVLM` auto-selects `cuda -> cpu` (MPS disabled)
  - `AudioToText` now also disables MPS auto-selection
  - explicit `cuda`/`mps` requests now validate backend availability in shared device resolution

### Dependencies

- Added `pydantic>=2.8.0` to AI dependencies.
- Bumped `transformers` minimum version to `>=4.57.0`.

### Tests

- Added focused tests for shared device resolution, `SceneVLM` initialization/parsing, and scene-first `VideoAnalysis`.
- Removed legacy tests tied to deleted detection/motion APIs and pre-rewrite `VideoAnalysis` behavior.

## 0.20.5

### Changed

- `VideoAnalysis.summary` is now always computed at access time from the current analysis state (including filtered analyses), while keeping serialized `summary` output for compatibility.
- Added `VideoAnalysis.filter(target="editing")` to produce an editing-focused `VideoAnalysis` view that filters low-confidence/low-signal predictions and keeps all high-quality signals (no hard cap on retained good moments).
- Reduced editing payload noise by stripping low-value frame details (for example, bounding boxes and repeated near-duplicate samples), compacting transcription/audio events to useful signals, and merging adjacent camera-motion spans.

### Documentation

- Updated `VideoAnalysis` API docs to describe the filtering-first workflow for editing use cases.

### Tests

- Added and updated tests for runtime summary computation, legacy-summary deserialization compatibility, editing filter behavior, no-cap retention of high-confidence actions, and motion timeline fallback.

## 0.20.4

### Changed

- Reimplemented `VideoAnalysis.summary` to prioritize high-level understanding outputs (overview narrative, primary subjects/actions, audio cues, pacing, themes, and timestamped highlights) instead of mostly low-level distributions.
- `VideoAnalysis.summary` now conditionally includes `transcript_full` when transcript quality appears reliable, and exposes gating details in `transcript_reliability`.

### Tests

- Added summary-focused tests covering high-level narrative outputs plus reliable vs. unreliable full-transcript inclusion behavior.

## 0.20.3

### Changed

- Simplified several `videopython.base` frame-processing paths to reduce overhead and code size.
- Removed unnecessary float64 conversion in `ColorGrading` saturation processing.

### Performance

- `SceneDetector` now reuses the previous frame histogram in `detect`, `detect_streaming`, and worker-based detection instead of recomputing both sides for each frame pair.
- `PictureInPicture` rounded-mask and masked-border logic is now vectorized with OpenCV/NumPy operations instead of nested Python pixel loops.
- `Resize` now uses a direct in-process frame resize path, avoiding multiprocessing setup and frame pickling overhead for common workloads.

## 0.20.2

### Changed

- Face detection/tracking transform defaults now use automatic backend/device resolution instead of hard-coded CPU defaults:
  - `FaceDetector(backend="auto")`
  - `FaceTracker(backend="auto")`
  - `FaceTrackingCrop(backend="auto")`
  - `SplitScreenComposite(backend="auto")`
- Added consistent initialization logging across AI generation, understanding, swapping, and dubbing classes to report selected execution device (and backend resolution where applicable).
- `ObjectSwapper` now propagates its configured device to internal `TextToImage` creation.

### Fixed

- `AudioToText` now consistently loads Whisper on the resolved runtime device.
- `AudioToText` and `AudioClassifier` constructor device handling now consistently uses shared device resolution.

## 0.20.1

### New Features

- Added `VideoAnalysisConfig.rich_understanding_preset()` for high-coverage, cross-domain video understanding defaults
- Added structured OCR output support via `DetectedText` and `TextDetector.detect_detailed()`, while preserving backward-compatible `TextDetector.detect() -> list[str]`

### Changed

- `VideoAnalyzer` now supports memory-budgeted sampling/chunking through `VideoAnalysisConfig.max_memory_mb` with `effective_max_frames` surfaced in `FrameSamplingReport`
- Action recognition orchestration supports `action_scope` (`video`, `scene`, `adaptive`) with `max_action_scenes` to improve temporal understanding quality on multi-scene videos
- `FrameAnalysisSample` now includes optional `text_regions` for OCR confidence + region geometry
- `VideoAnalysis` summary now includes richer aggregate signals (top actions/objects, OCR terms, face presence, motion distributions)

### Fixed

- Geo redaction now sanitizes `source.raw_tags` when `redact_geo=True`, preventing location metadata leakage
- Added stronger `VideoAnalysisConfig` validation for analyzer IDs and sampling/runtime constraints to fail fast on invalid configurations

### Tests

- Added explicit full `VideoAnalysis` JSON roundtrip test
- Added tests for scene-scoped action execution, memory-budgeted sampling behavior, geo redaction, and rich preset coverage
- Added lightweight unit test for structured OCR regions without model downloads

## 0.20.0

### Breaking Changes

- `videopython.ai` is now local-only. All non-local/cloud provider backends were removed.
- Removed backend/config/error compatibility modules:
  - `videopython.ai.backends`
  - `videopython.ai.config`
  - `videopython.ai.exceptions`
- Removed cloud-only analyzers:
  - `ShotTypeClassifier`
  - `CombinedFrameAnalyzer`
  - `CombinedFrameAnalysis`
- Removed backend-related API surface from constructors and orchestration where no longer applicable (`backend`, `api_key`, backend override/fallback settings).

### Changed

- Simplified AI runtime paths to direct local inference across generation, understanding, dubbing, and swapping modules.
- `VideoAnalyzer` frame-analysis orchestration now runs only local analyzers.
- Introduced shared local device selection helper with consistent automatic behavior:
  - default auto-selection: `cuda` -> `cpu`
  - optional `mps` in auto-selection only for models where explicitly allowed

### Dependencies

- Removed cloud SDK dependencies from AI extras/groups:
  - `openai`, `google-generativeai`, `protobuf`, `elevenlabs`, `runwayml`, `lumaai`, `replicate`, `requests`

### Documentation

- Rewrote README and AI docs/examples for local-only usage.
- Added explicit project positioning in README: open-source and local-first.

## 0.19.0

### New Features

- Added `VideoAnalysis`, `VideoAnalysisConfig`, and `VideoAnalyzer` to aggregate understanding outputs into a serializable analysis object
- Added `CombinedFrameAnalysis.to_dict()` / `from_dict()` and `AudioClassification.to_dict()` / `from_dict()` for complete analysis serialization support
- Added bounded-memory `VideoAnalyzer.analyze_path()` orchestration that prefers path-based analyzers and streaming/chunked frame sampling

### Documentation

- Added `docs/api/ai/video_analysis.md` with usage and API references
- Linked AI Video Analysis in API overview and MkDocs navigation

## 0.18.3

### Changed

- Consolidated face-framing behavior into `FaceTrackingCrop` by adding framing controls (`framing_rule`, `headroom`, `lead_room`) and optional movement speed clamping (`max_speed`)

### Removed

- Removed `AutoFramingCrop` and the `auto_framing` AI registry operation; use `FaceTrackingCrop` for face-based framing/cropping

### Documentation

- Updated AI transform docs/registry docs/examples to reflect `FaceTrackingCrop` as the single face-based crop/framing transform

## 0.18.2

### Fixed

- `Resize` now rounds output dimensions to even values by default (runtime and `VideoMetadata.resize()` prediction), preventing width-only/height-only resizes from producing encoder-incompatible odd dimensions
- `SplitScreenComposite` now snaps final composite canvas dimensions to even values by default for H.264 / `yuv420p` compatibility
- `Video.save()` now raises a clear error for odd frame dimensions before invoking FFmpeg and surfaces FFmpeg stderr when the encoder terminates early (instead of a bare `BrokenPipeError`)

## 0.18.1

### Fixed

- Added `VideoEdit.validate()` metadata prediction support for AI aspect-crop transforms `auto_framing` and `face_crop`
- `VideoEdit.validate()` no longer fails on `auto_framing` / `face_crop` plans after `import videopython.ai` due to missing registry `metadata_method`

## 0.18.0

### New Features

- **VideoEdit editing plans**: New multi-segment editing plan API for assembling clips from one or more source videos
  - `VideoEdit.run()` executes segment extraction, per-segment transforms/effects, concatenation, and post-assembly operations
  - `VideoEdit.validate()` performs dry-run validation using `VideoMetadata` and checks concatenation compatibility (exact fps and dimensions)
  - JSON plan support via `VideoEdit.from_dict()`, `VideoEdit.from_json()`, and canonical serialization via `VideoEdit.to_dict()`
  - Registry-backed parsing with canonical op IDs, alias normalization, and clear errors for unsupported operations
- **New base exports**: `VideoEdit` and `SegmentConfig` are now exported from `videopython.base`
- **VideoEdit JSON Schema generation**: `VideoEdit.json_schema()` builds a parser-aligned plan schema from the operation registry
  - Canonical op IDs only
  - Excludes unsupported categories/tags and non-JSON-instantiable operations
  - Reflects currently registered operations (AI ops appear after `import videopython.ai`)

### Improved Validation

- Stronger parse-time JSON plan validation for `VideoEdit`
  - Validates parameter value types, enums, array item types, and selected numeric minimum constraints using registry `ParamSpec` metadata
  - Normalizes JSON values for constructor compatibility where needed (for example enum values and tuple-like args)
  - Nullable registry params (e.g. effect `apply.start` / `apply.stop`) now emit correct JSON Schema and validate `null` inputs
- Added metadata prediction support for core transforms used by `VideoEdit.validate()` (`cut`, `cut_frames`, `resize`, `crop`, `resample_fps`, `speed_change`)
- Added `VideoMetadata.speed_change()` for metadata-only validation of speed changes

### Fixed

- `VideoEdit.validate()` metadata prediction now matches runtime frame-rounding semantics for time-based cuts (`cut`)
- `VideoEdit.validate()` crop metadata prediction now matches runtime crop behavior (including odd-size center crops and slice clipping)
- `VideoEdit.json_schema()` and parser semantics are aligned for `resize` by requiring at least one non-null dimension (`width` or `height`)

### Documentation

- Added `docs/api/editing.md` with `VideoEdit` JSON plan format, validation, and schema generation examples
- Linked editing plans from `docs/api/index.md`
- Added a `VideoEdit` JSON plan example to `docs/examples/social-clip.md`

## 0.17.0

### New Features

- **Operation Registry**: Machine-readable metadata for all video operations, enabling downstream tools to discover operations, parameters, and capabilities without importing internal modules
  - `OperationSpec` dataclass with stable operation IDs, category, tags, and aliases
  - `ParamSpec` dataclass for constructor and apply method parameter schemas
  - JSON Schema generation via `to_json_schema()` (constructor args) and `to_apply_json_schema()` (apply args)
  - Registry API: `get_operation_specs()`, `get_operation_spec()`, `get_specs_by_category()`, `get_specs_by_tag()`
  - `spec_from_class()` helper to introspect class signatures and build specs automatically
  - `register()` for adding custom operations with collision detection
  - 18 base operations and 3 AI operations registered with stable IDs
  - AI operations registered lazily on `import videopython.ai`

### New Exports from `videopython.base`

- `OperationCategory`, `OperationSpec`, `ParamSpec`
- `get_operation_specs`, `get_operation_spec`, `get_specs_by_category`, `get_specs_by_tag`
- `register`, `spec_from_class`

## 0.16.6

### Fixed

- `AudioToText` and `AudioClassifier` now auto-detect the best available device (CUDA, MPS, CPU) instead of defaulting to CPU

## 0.16.5

### Security

- Upgrade protobuf minimum version to 5.29.6 to fix DoS vulnerability via recursive `Any` messages in `json_format.ParseDict()` (affects protobuf < 5.29.6)

## 0.16.4

### New Features

- **Progress and verbosity configuration**: New `configure()`, `set_verbose()`, and `set_progress()` functions to control logging and progress bars in base operations (off by default)

### Fixed

- Fix off-by-one in `ResampleFPS` frame interpolation that dropped the last frame
- Sync audio duration after FPS resampling to prevent audio/video drift

### Changed

- `Video.save()` now streams raw frames to FFmpeg via stdin pipe instead of writing a temporary file to disk
- Suppress noisy FFmpeg output during save

## 0.16.3

### Security

- Upgrade Pillow minimum version to 12.1.1 to fix out-of-bounds write when loading PSD images (affects Pillow >= 10.3.0, < 12.1.1)

## 0.16.2

### Fixed

- Prevent audio slicing near the end of a clip from failing due to floating point precision mismatch

## 0.16.1

### New Features

- **GPU-accelerated face detection**: `FaceDetector` now supports GPU acceleration via YOLOv8-face model
  - New `backend` parameter: `"cpu"` (default, Haar cascade), `"gpu"` (YOLOv8-face), or `"auto"`
  - New `detect_batch()` method for efficient batched detection on video frames
  - Uses `arnabdhar/YOLOv8-Face-Detection` model from Hugging Face

- **GPU support for face tracking transforms**: `FaceTrackingCrop`, `SplitScreenComposite`, `AutoFramingCrop`
  - New `backend` parameter to enable GPU acceleration
  - New `sample_rate` parameter for frame sampling with interpolation (GPU only)
  - Backward compatible - defaults to CPU backend

- **Video-level face tracking**: New `FaceTracker.track_video()` method
  - Batched detection for optimal GPU utilization
  - Frame sampling with smooth interpolation between detected frames
  - Configurable `batch_size` parameter

### Example

```python
# CPU (default, backward compatible)
video = FaceTrackingCrop().apply(video)

# GPU with frame sampling for speed
video = FaceTrackingCrop(backend="gpu", sample_rate=5).apply(video)
```

## 0.16.0

### Breaking Changes

- **VideoAnalyzer removed**: The `VideoAnalyzer` class and all orchestration types have been extracted to a separate package. Use individual backbone tools directly:
  - Scene detection: `SceneDetector`, `SemanticSceneDetector` (now return `SceneBoundary`)
  - Image analysis: `ImageToText.describe_image()`
  - Object detection: `ObjectDetector`, `FaceDetector`, `TextDetector`
  - Audio: `AudioToText`, `AudioClassifier`
  - Actions: `ActionRecognizer`

- **Removed types**: `VideoDescription`, `SceneDescription`, `FrameDescription`, `SceneUnderstanding`, `VisualEvent`, `LLMSummarizer`
- **New type**: `SceneBoundary` - lightweight timing structure for scene boundaries

## 0.15.6

### New Features

- **Key Frame Extraction**: Extract representative frames from each scene during video analysis
  - New `extract_key_frames` parameter in `VideoAnalyzer.analyze_path()` (default: False)
  - New `key_frame_width` parameter to control output size (default: 640px, height auto-scaled)
  - Extracts middle frame of each scene as JPEG (quality 85)
  - New `SceneDescription.key_frame` field containing JPEG bytes
  - New `SceneDescription.key_frame_timestamp` field with frame timestamp
  - Serialization support: `to_dict()` encodes as base64, `from_dict()` decodes back to bytes

## 0.15.5

### New Features

- **Crop transform**: Enhanced with normalized coordinates and custom positioning
  - Now accepts both pixel values (int) and normalized coordinates (float 0-1)
  - Float values in range (0, 1] are interpreted as percentages of video dimensions
  - New `x` and `y` parameters for custom crop positioning
  - New `CropMode.CUSTOM` mode for arbitrary crop regions
  - Example: `Crop(width=0.5, height=0.5)` crops to 50% of original size
  - Example: `Crop(width=0.5, height=1.0, x=0.5, y=0.0, mode=CropMode.CUSTOM)` crops right half

## 0.15.4

### Fixed

- Suppress `use_fast` deprecation warning from BlipProcessor by explicitly setting `use_fast=True`

## 0.15.3

### Changed

- **Audio Classification Backend**: Replaced PANNs with Audio Spectrogram Transformer (AST)
  - Uses `MIT/ast-finetuned-audioset-10-10-0.4593` model from HuggingFace
  - Same 527 AudioSet classes with state-of-the-art performance (0.485 mAP)
  - More reliable model downloads (PANNs used Zenodo which had timeout issues in CI)
  - Uses sliding window approach for temporal event detection

### Dependencies

- Removed `panns-inference` dependency (AST uses `transformers` which is already included)

## 0.15.2

### New Features

- **JSON Serialization**: Added `to_dict()` and `from_dict()` methods to all description and transcription classes for easy JSON serialization of analysis results
  - `VideoDescription.to_dict()` / `VideoDescription.from_dict()` - Full video analysis roundtrip
  - `SceneDescription.to_dict()` / `SceneDescription.from_dict()` - Scene-level serialization
  - `FrameDescription.to_dict()` / `FrameDescription.from_dict()` - Frame-level serialization
  - `Transcription.to_dict()` / `Transcription.from_dict()` - Audio transcription serialization
  - All nested dataclasses (`BoundingBox`, `DetectedObject`, `DetectedFace`, `ColorHistogram`, `AudioEvent`, `MotionInfo`, `DetectedAction`, `TranscriptionSegment`, `TranscriptionWord`) also support serialization

### Removed

- **`ColorHistogram.hsv_histogram`**: Removed unused field that stored raw HSV histogram numpy arrays. The field was never read after being set. Color analysis still provides `dominant_colors`, `avg_hue`, `avg_saturation`, and `avg_value`.
- **`include_full_histogram` parameter**: Removed from `VideoAnalyzer.analyze()`, `VideoAnalyzer.analyze_path()`, `ImageAnalyzer.describe_frame()`, `ImageAnalyzer.describe_frames()`, `ImageAnalyzer.describe_scene()`, and `ColorAnalyzer.extract_color_features()`.

## 0.15.1

### New Features

- **Adaptive Frame Sampling**: New `sampling_strategy` parameter for `VideoAnalyzer`
  - `'fixed'`: Original behavior - sample at fixed FPS rate
  - `'adaptive'`: Smart sampling using start + ln(1+duration) + end formula
  - Reduces frames by ~27% while maintaining scene coverage
  - Short scenes (<=2s): 1-2 frames
  - Longer scenes: start frame + logarithmic middle frames + end frame

## 0.15.0

### New Features

- **ObjectSwapper**: Replace objects in videos using AI-powered segmentation and inpainting
  - `ObjectSwapper.swap()` - Replace object with AI-generated content from text prompt
  - `ObjectSwapper.swap_with_image()` - Replace object with provided image
  - `ObjectSwapper.remove_object()` - Remove object and fill with background
  - `ObjectSwapper.segment_only()` - Get object masks without modification
  - `ObjectSwapper.visualize_track()` - Debug visualization of tracked object

- **ObjectSegmenter**: SAM2-based video object segmentation
  - Text prompts via GroundingDINO (e.g., "red car", "person")
  - Point and bounding box prompt support
  - Automatic tracking across video frames

- **VideoInpainter**: SDXL-based video inpainting
  - Remove objects and fill with generated background
  - Mask dilation for cleaner edges
  - Optional temporal consistency blending

### New Data Structures

- **ObjectMask**: Single-frame object mask with confidence and bounding box
- **ObjectTrack**: Tracked object across multiple frames
- **SwapResult**: Result containing swapped frames and metadata
- **SegmentationConfig**: Configuration for SAM2 segmentation
- **InpaintingConfig**: Configuration for SDXL inpainting
- **SwapConfig**: Combined configuration for full pipeline

### New Backends

- Added Replicate backend for ObjectSwapper (cloud-based, no local GPU required)

### Dependencies

- Added `replicate>=0.20.0` for cloud backend

## 0.14.1

### New Features

- **Voice Revoicing**: Replace speech with custom text using voice cloning
  - `VideoDubber.revoice()` - Generate new speech with cloned voice
  - `VideoDubber.revoice_and_replace()` - Convenience method returning video with new audio
  - Extracts voice sample from original speaker automatically
  - Preserves background audio (music, sound effects) via Demucs separation
  - Natural pacing - speech duration matches text length

- **Audio.silence()**: New class method to create silent audio tracks
  - Configurable duration, sample rate, and channels
  - Useful for padding audio tracks

### New Data Structures

- **RevoiceResult**: Result of voice replacement operation
  - `revoiced_audio`: Final audio with new speech
  - `text`: The text that was spoken
  - `voice_sample`: Voice sample used for cloning
  - `speech_duration`: Duration of generated speech

## 0.14.0

### New Features

- **VideoDubber**: Automatic video dubbing with translation and voice synthesis
  - Transcribes speech using Whisper
  - Translates text to target language (OpenAI GPT-4o or local Ollama)
  - Generates dubbed speech with natural timing
  - Supports 50+ languages

- **Voice Cloning**: Clone original speaker's voice for dubbed audio
  - Uses XTTS-v2 model from Coqui TTS
  - Extracts voice samples from separated vocals
  - Preserves speaker characteristics in translated speech

- **Background Preservation**: Keep music and sound effects while replacing speech
  - Uses Demucs for audio source separation (vocals vs background)
  - Mixes dubbed speech with original background audio
  - Maintains audio atmosphere of original video

- **Multiple Backends**:
  - **ElevenLabs**: Cloud-based dubbing with professional voice quality
  - **Local Pipeline**: Fully offline dubbing using Whisper + XTTS + Demucs
  - Configurable translation backend (OpenAI or Ollama)

- **Timing Synchronization**: Dubbed speech matches original timing
  - Analyzes original segment durations
  - Adjusts speech speed to fit within segment boundaries
  - Maintains natural pacing and pauses

### New Modules

- `videopython.ai.dubbing` - Video dubbing pipeline
  - `VideoDubber` - Main API for dubbing videos
  - `DubbingResult` - Result with dubbed audio and metadata
  - `TranslatedSegment` - Individual translated speech segment

- `videopython.ai.generation.translation` - Text translation
  - `TextTranslator` - Translate text between languages
  - Backends: OpenAI (gpt-4o-mini) and Ollama (local models)

- `videopython.ai.understanding.separation` - Audio source separation
  - `AudioSeparator` - Separate vocals from background using Demucs
  - `SeparatedAudio` - Container for separated audio tracks

### Dependencies

- Added `coqui-tts>=0.24.0` for voice cloning TTS
- Added `demucs>=4.0.0` for audio source separation
- Added `requests>=2.28.0` for ElevenLabs API

## 0.13.0

### New Features

- **ActionRecognizer**: Recognize actions and activities in video clips
  - Uses VideoMAE model fine-tuned on Kinetics-400 (400 action classes)
  - Supports both "base" and "large" model variants
  - Per-scene action recognition via `recognize_scenes()`
  - Memory-efficient `recognize_path()` for file-based analysis
  - Example actions: "walking", "running", "dancing", "answering questions", "using computer"

- **SemanticSceneDetector**: ML-based scene boundary detection using TransNetV2
  - More accurate than histogram-based detection, especially for gradual transitions
  - Uses pretrained weights from `transnetv2-pytorch` package
  - Competitive F1 scores: 77.9 (ClipShots), 96.2 (BBC Planet Earth), 93.9 (RAI)

- **VideoAnalyzer enhancements**:
  - New `use_semantic_scenes` parameter to use ML-based scene detection
  - New `recognize_actions` parameter to enable action recognition per scene
  - New `action_confidence_threshold` parameter (default: 0.1)

- **MPS (Apple Silicon) support**: Automatic device detection for CUDA, MPS, and CPU
  - ActionRecognizer and SemanticSceneDetector work on Apple Silicon Macs

### New Data Structures

- **DetectedAction**: Action detected in a video segment
  - `label`: Action name (e.g., "running", "talking")
  - `confidence`: Detection confidence (0-1)
  - `start_frame`, `end_frame`: Frame range
  - `start_time`, `end_time`: Time range in seconds

- **SceneDescription**: Added `detected_actions` field for per-scene actions

### Dependencies

- Added `transnetv2-pytorch>=1.0.5` to AI optional dependencies

## 0.12.0

### Breaking Changes

- **FrameDescription**: Removed `camera_motion` field - use `motion.motion_type` instead
- **FrameDescription**: Changed `detected_faces` from `int | None` to `list[DetectedFace] | None`
  - Use `len(detected_faces)` to get the count
  - Local backend now returns faces with bounding boxes
  - Cloud backends return faces without bounding boxes (count only)
- **DetectedFace**: Made `bounding_box` optional (can be `None` for cloud backends)
- **FaceDetector**: Removed `count()` method - use `len(detect(image))` instead

### Changed

- Extracted duplicate `_adjust_audio_duration` methods to `Audio.fit_to_duration()`
- Vectorized `Vignette` effect (removed per-frame loop)
- Added `MIN_FRAMES_FOR_MULTIPROCESSING` threshold (100 frames) to `Resize`, `Blur`, and `ColorGrading` to avoid multiprocessing overhead for short videos

## 0.11.1

### New Features

- **Exception hierarchy**: Proper exception classes for better error handling
  - Base module: `VideoPythonError`, `VideoError`, `VideoLoadError`, `VideoMetadataError`, `AudioError`, `AudioLoadError`, `TransformError`, `InsufficientDurationError`, `IncompatibleVideoError`, `TextRenderError`, `OutOfBoundsError`
  - AI module: `BackendError`, `MissingAPIKeyError`, `UnsupportedBackendError`, `GenerationError`, `LumaGenerationError`, `RunwayGenerationError`, `ConfigError`
  - All exceptions exported from `videopython.base` and `videopython.ai`

- **AI module tests in CI**: Added lightweight AI tests to CI pipeline
  - Tests for models <100MB run in CI (YOLO, PANNs, OpenCV)
  - Tests for models 100MB+ excluded via `@pytest.mark.requires_model_download` marker
  - New `ai_tests` job in CI workflow

### Fixed

- Replaced broad `except Exception` patterns with specific exception types
- Config file parsing errors now emit warnings instead of failing silently
- LLM summarization failures now emit warnings with fallback behavior

### Changed

- Moved `VideoMetadataError` from `video.py` to `exceptions.py`
- Moved `AudioLoadError` from `audio/audio.py` to `exceptions.py`
- Moved AI backend exceptions from `backends.py` to `exceptions.py`
- Transitions now raise `InsufficientDurationError` instead of `RuntimeError`
- Transitions now raise `IncompatibleVideoError` instead of `ValueError`
- Video loading now raises `VideoLoadError` instead of `ValueError`

## 0.11.0

### New Features

- **SpeedChange transform**: Change video playback speed
  - Constant speed changes (e.g., 2x faster, 0.5x slower)
  - Smooth speed ramping for cinematic effects

- **ColorGrading effect**: Adjust video color properties
  - Brightness, contrast, saturation adjustments
  - Color temperature control (warm/cool tones)

- **Vignette effect**: Add darkened edges to frames
  - Configurable strength and radius

- **KenBurns effect**: Cinematic pan-and-zoom effect
  - Animate between two regions using normalized BoundingBox coordinates
  - Easing functions: linear, ease_in, ease_out, ease_in_out
  - Fluent API: `video.ken_burns(start_region, end_region, easing="ease_in_out")`

- **PictureInPicture transform**: Overlay video on main video
  - Configurable position (normalized 0-1), scale, border, rounded corners, opacity
  - Overlay loops automatically if shorter than main video
  - Fluent API: `video.picture_in_picture(overlay, position=(0.7, 0.7), scale=0.25)`

- **FaceDetector bounding boxes**: Now returns detailed face information
  - Returns `list[DetectedFace]` with normalized bounding box coordinates
  - Use `.count()` for backward compatibility with previous API

- **SpeedChange audio synchronization**: Audio now adjusts with video speed
  - Pitch-preserving time stretch using FFmpeg atempo filter
  - New `adjust_audio: bool = True` parameter (enabled by default)
  - For speed ramps, uses average speed for audio adjustment

- **PictureInPicture audio mixing**: Configurable audio handling for overlays
  - New `audio_mode` parameter: `"main"` (default), `"overlay"`, or `"mix"`
  - New `audio_mix` parameter for volume factors in mix mode, e.g. `(0.8, 0.5)`
  - Overlay audio loops automatically if shorter than main video

- **FaceTrackingCrop transform** (in `videopython.ai`): Crop video to follow detected faces
  - Create vertical (9:16) content from horizontal (16:9) by tracking speaker
  - Configurable face selection: largest, centered, or by index
  - Smooth position tracking with exponential moving average
  - Fallback options when face not detected: center, last_position, full_frame

- **SplitScreenComposite transform** (in `videopython.ai`): Arrange multiple videos in grid layouts
  - Layouts: 2x1, 1x2, 2x2, 1+2, 2+1
  - Face tracking for each cell to keep subjects centered
  - Configurable gap, border, and colors

- **AutoFramingCrop transform** (in `videopython.ai`): Intelligent cropping with cinematographic rules
  - Framing rules: thirds, center, headroom, dynamic
  - Configurable headroom and lead room
  - Smooth camera movement with speed limiting

- **FaceTracker utility** (in `videopython.ai`): Frame-by-frame face tracking with smoothing
  - Selection strategies: largest, centered, by index
  - Detection interval to reduce processing load
  - Exponential moving average for jitter-free tracking

- **Audio.time_stretch()**: New method for pitch-preserving time stretching
  - Supports extreme speeds via chained atempo filters (e.g., 4x, 0.25x)

- **Audio.scale_volume()**: New method for volume adjustment

## 0.10.0

### Breaking Changes

- Removed `TransformationPipeline` class - use the new fluent API instead

### New Features

- **Fluent API for Video**: Chain transformations directly on Video objects
  - `video.cut(start, end)` - cut by time range
  - `video.cut_frames(start, end)` - cut by frame range
  - `video.resize(width, height)` - resize (aspect ratio preserved if only one dimension given)
  - `video.crop(width, height)` - center crop
  - `video.resample_fps(fps)` - change frame rate
  - `video.transition_to(other, transition)` - combine with another video

- **Fluent API for VideoMetadata**: Validate operations before execution
  - Same methods as Video, but only transforms metadata (fast, no frame processing)
  - Raises `ValueError` for invalid operations (e.g., incompatible dimensions for transitions)
  - Example: `video.metadata.cut(0, 10).resize(1280, 720)` validates the operation chain

- **Transcription**: Added `words` property to access all words across segments

### Fixed

- Fixed PyTorch 2.6+ compatibility for speaker diarization (omegaconf serialization)

### Migration Guide

```python
# Before (0.9.x)
from videopython.base import TransformationPipeline, CutSeconds, Resize
pipeline = TransformationPipeline([CutSeconds(0, 10), Resize(1280, 720)])
result = pipeline.run(video)

# After (0.10.0)
result = video.cut(0, 10).resize(1280, 720)

# Validate before executing (optional)
output_meta = video.metadata.cut(0, 10).resize(1280, 720)
```

## 0.9.1

- Re-release of 0.9.0 (PyPI publish failed)

## 0.9.0

### Breaking Changes

- `add_audio()` and `add_audio_from_file()` now return a new `Video` instance instead of mutating in place
- Reduced public API exports from `videopython.base` (items still importable, just not in `__all__`)

### Deprecated

- `Audio.from_file()` is deprecated, use `Audio.from_path()` instead

### Fixed

- Security: Replaced `eval()` with `json.loads()` when parsing ffprobe output in audio loading
- `Audio.is_silent` now returns Python `bool` instead of `np.bool`
- Exception handling now uses specific exceptions instead of generic `except Exception`

### Changed

- Large video loading (>10GB estimated RAM) now emits `ResourceWarning` suggesting `FrameIterator`

## 0.8.3

- Added `preset` and `crf` parameters to `Video.save()` for encoding control
  - `preset`: Speed/compression tradeoff (ultrafast to veryslow), default "medium"
  - `crf`: Quality control (0-51), default 23. Lower values = better quality, larger files
  - Changed default preset from "ultrafast" to "medium" for better compression
  - Removed `-tune zerolatency` for improved compression efficiency

## 0.8.2

- Fixed missing `MotionInfo` export from `videopython.base`
- Added documentation build check to CI pipeline

## 0.8.1

- Added `MotionAnalyzer` for motion detection via optical flow analysis (Farneback method)
  - Detects motion types: static, pan, tilt, zoom, complex
  - Returns normalized motion magnitude (0-1) and raw pixel displacement
  - Frame-level analysis with `analyze_frames()` and `analyze_frame_sequence()`
  - Memory-efficient `analyze_video_path()` for long videos
  - Scene-level aggregation via `aggregate_motion()`
- Added `analyze_motion` parameter to `VideoAnalyzer.analyze()` and `VideoAnalyzer.analyze_path()`
  - Motion info automatically distributed to scene descriptions
  - New `avg_motion_magnitude` and `dominant_motion_type` fields on `SceneDescription`
- New dataclass: `MotionInfo`

## 0.8.0

- Added `AudioClassifier` for sound event detection using PANNs (Pretrained Audio Neural Networks)
  - Detects 527 AudioSet sound classes (speech, music, animals, vehicles, alarms, etc.)
  - Returns timestamped `AudioEvent` objects with start/end times and confidence scores
  - Configurable confidence threshold and model selection (Cnn14, Cnn10, ResNet38, MobileNetV2)
  - Frame-level predictions (~10ms resolution) automatically merged into coherent events
- Added `classify_audio` parameter to `VideoAnalyzer.analyze()` and `VideoAnalyzer.analyze_path()`
  - Audio events are automatically distributed to scene descriptions
  - New `audio_events` field on `SceneDescription`
- New dataclasses: `AudioEvent`, `AudioClassification`
- Added `panns-inference` to AI dependencies
- Fixed audio slicing issue in Blur transition

## 0.7.3

- Fixed thread safety issue in `ImageToText.describe_frames()` that caused meta tensor errors during concurrent model initialization
- Removed thread parallelism from frame captioning (was causing race conditions with lazy model loading)
- BLIP captioning now defaults to CPU instead of MPS on Apple Silicon (benchmarks show CPU is ~2x faster for this model)
- Parallel processing remains available for scene detection via `SceneDetector.detect_parallel()` (histogram-based, no AI models)

## 0.7.2

- Fixed compatibility with transformers 4.52+ (updated BLIP import path)

## 0.7.1

- Added `SceneDetector.detect_parallel()` for multi-core scene detection (3.5x speedup on 8 cores)
- Added `SceneDetector.detect_streaming()` for memory-efficient frame-by-frame processing
- Added `FrameIterator` for streaming video frames without loading entire video into memory
- Added `VideoAnalyzer.analyze_path()` for memory-efficient analysis of long videos
- Scene detection now works on video files directly without loading all frames into RAM

## 0.7.0

- **Breaking change**: Removed async/await from all AI module APIs - all functions are now synchronous
- Simplified API: no more `asyncio.run()` boilerplate required
- `describe_frames()` uses `ThreadPoolExecutor` for parallel frame processing (maintains performance)
- Removed `pytest-asyncio` dependency
- Fixed whisper type stubs (renamed to `__init__.pyi`)

## 0.6.3

- Removed `VideoUpscaler` - MMagic/mmcv has compatibility issues with NumPy 2.x and is unmaintained
- Removed MPS support for CogVideoX models (TextToVideo, ImageToVideo) - these require CUDA due to 364GB+ memory requirements on MPS
- Fixed `TextToMusic` crash on MPS - added missing `.cpu()` call before numpy conversion
- Removed `mmagic`, `mmcv`, `mmengine` dependencies (reduces install size significantly)
- Added MPS backend testing documentation (`scripts/mps_tests/MPS_TESTING.md`)

## 0.6.2

- Added silence detection in audio analysis (`Audio.get_silence_intervals()`)
- Added scene type detection for videos (`SceneType` enum with OUTDOOR, INDOOR, CLOSEUP, etc.)
- Added comprehensive tests for audio and scene functionality
- Added import isolation tests to ensure optional dependencies don't break core imports

## 0.6.1

- Added video understanding capabilities including scene analysis, color histograms, and object/text/face detection
- Added speaker diarization support for audio/video transcription
- Added video upscaling with RealBasicVSR model
- Added new AI backends: Luma Dream Machine and Runway Gen-4 Turbo for video generation
- Added Polish and German text-to-speech support
- Dropped `soundpython` dependency - audio functionality now built into videopython
- Extended documentation with new examples and API reference

# TODO: `VideoEdit` Editing Plan Representation

## Status

### Phase 1 (implemented)

Core `VideoEdit` execution + validation is now in place:

- `src/videopython/base/edit.py`
  - `EffectApplication`
  - `SegmentConfig.process_segment()`
  - `VideoEdit.run()`
  - `VideoEdit.validate()` with registry-driven `metadata_method` prediction
- `src/videopython/base/video.py`
  - `VideoMetadata.speed_change()` metadata prediction
- `src/videopython/base/registry.py`
  - `metadata_method` wired for core transforms (including `speed_change`)
- `src/videopython/base/__init__.py`
  - exports added
- `src/tests/base/test_video_edit.py`
  - Phase 1 coverage for execution, validation, cache behavior, and metadata prediction

### Phase 2 (next)

Add registry-backed JSON plan parsing and serialization for LLM-generated editing plans:

- `VideoEdit.from_dict(...)`
- `VideoEdit.from_json(...)`
- `VideoEdit.to_dict()` (canonicalized output)
- Stable roundtrips with canonical registry IDs and effect apply params

This plan is based on `REQUEST.md`, `MULTICUT_DESIGN.md`, and the current Phase 1 implementation.

## Current Architecture Facts (relevant to Phase 2)

- `Video.from_path(...)` already supports `start_second` / `end_second` segment loading in `src/videopython/base/video.py`.
- `Video.__add__` concatenation is strict: same `fps` and same `frame_shape` required (`src/videopython/base/video.py`).
- Registry already exposes constructor/apply JSON schemas via `OperationSpec` (`src/videopython/base/registry.py`).
- Registry `apply_params` naturally models effect `start`/`stop` (`Effect.apply(video, start=None, stop=None)`).
- `VideoMetadata` already provides fluent prediction for core operations (`cut`, `cut_frames`, `resize`, `crop`, `resample_fps`) in `src/videopython/base/video.py`.
- Base module must remain independent from `videopython.ai` (enforced by `src/tests/base/test_import_isolation.py`).
- Registry `metadata_method` is now populated for core transform validation paths used by `VideoEdit.validate()`.

## Phase 2 Scope

### In scope

- JSON deserialization into working `VideoEdit` objects using the operation registry
- Canonical JSON serialization (`to_dict`) for roundtrip/debugging
- Alias normalization (`blur` -> `blur_effect`, etc.)
- Category/tag validation for JSON plans (transforms vs effects, reject unsupported ops)
- Error messages tailored for registry-driven plans (including AI lazy-registration hint)
- Tests for parse/serialize/roundtrip/validation-after-parse
- Docs for JSON plan format and usage
- Internal model simplification: registry-backed step records are the primary representation

### Explicitly out of scope (Phase 2)

- Transitions between segments
- Auto-resizing/normalization of incompatible segments
- Full JSON Schema generation for the entire `VideoEdit` plan (nice-to-have, can be Phase 3)
- Generic resolver hooks for non-JSON constructor args (e.g. numpy arrays / `BoundingBox`)
- Primitive type validation of JSON args against `ParamSpec.json_type` (Python constructors catch type errors at instantiation; defer jsonschema-level validation to Phase 3)
- Backward compatibility with pre-merge Phase 1 internal object-only construction/storage shape

## Phase 2 Locked Decisions

- File: extend existing `src/videopython/base/edit.py` (already created in Phase 1).
- API: `from_dict(data)`, `from_json(text)`, `to_dict()`. No `registry_lookup` parameter -- use the global registry directly. Simpler API; tests can monkeypatch if needed.
- `to_dict()` normalizes to canonical `OperationSpec.id` (never emits aliases).
- Internal storage is `_StepRecord`-backed and is the source of truth for serialization/execution/validation.
- `_StepRecord` is the only step representation in Phase 2 internals (no dual object-list + sidecar model).
- `to_dict()` serializes from `_StepRecord` only (no introspection fallback).
- `SegmentConfig` / `VideoEdit` direct constructors accept `_StepRecord` lists (or internal normalized forms), not raw operation-object lists.
- Public construction path is `VideoEdit.from_dict(...)` / `VideoEdit.from_json(...)`.
- `EffectApplication` is removed from the Phase 2 internal model (the `_StepRecord` carries effect + apply bounds).
- `SegmentConfig` remains an exported type but is not a user-facing construction API in Phase 2; users are expected to construct via `VideoEdit.from_dict(...)` / `from_json(...)`.
- `validate()` behavior unchanged: raise on invalid, return predicted metadata on success.
- `run()` does not implicitly validate. Keep as-is.
- Unknown top-level keys in JSON plans: ignore (forward-compatible). Unknown keys inside step dicts: reject (catch typos).

## Phase 2 Detailed Implementation Plan

### A. Make `_StepRecord` the primary step representation

Store canonical step data and the live operation object together. `_StepRecord` becomes the internal source of truth.

- [ ] Add `_StepRecord` (private, not exported) as a simple frozen dataclass:
  ```python
  @dataclass(frozen=True)
  class _StepRecord:
      op_id: str           # canonical registry ID
      args: dict           # constructor args (deep-copied snapshot)
      apply_args: dict     # effect start/stop (deep-copied snapshot; empty for transforms)
      operation: Transformation | Effect  # typed live operation

      @classmethod
      def create(cls, op_id, args, apply_args, operation):
          return cls(
              op_id=op_id,
              args=copy.deepcopy(args),
              apply_args=copy.deepcopy(apply_args),
              operation=operation,
          )
  ```
- [ ] **Snapshot semantics**: `_StepRecord` captures the JSON as parsed. If the caller later mutates the live operation object (e.g. `transform.width = ...`), `to_dict()` still emits the original parsed values. This is intentional -- `_StepRecord` is a parse-time snapshot, not a live binding. Document this in the `to_dict()` docstring.
- [ ] **Defensive deep copies on ingest and emit**: all internal callers construct records via `_StepRecord.create(...)` so `args` and `apply_args` are deep-copied on ingest. `to_dict()` must also deepcopy on emit.
- [ ] Refactor `SegmentConfig` to store `transform_records: tuple[_StepRecord, ...]` and `effect_records: tuple[_StepRecord, ...]` as primary internals.
- [ ] Refactor `VideoEdit` to store `post_transform_records: tuple[_StepRecord, ...]` and `post_effect_records: tuple[_StepRecord, ...]` as primary internals.
- [ ] Change direct constructors to accept record lists/tuples (no raw operation-object list constructor path).
- [ ] `from_dict` constructs normalized record-backed objects directly.
- [ ] Merge execution/validation integration into this refactor:
  - `SegmentConfig.process_segment()` iterates over `_StepRecord.operation`
  - `VideoEdit.run()` iterates over record-held operations
  - `VideoEdit.validate()` predicts transforms using `_StepRecord.op_id` + `_StepRecord.args` (no constructor arg introspection)
  - Effects execute as `record.operation.apply(video, start=record.apply_args.get("start"), stop=record.apply_args.get("stop"))`
- [ ] Simplify metadata prediction helpers:
  - `_predict_transform_metadata(...)` accepts `(meta, op_id, args)` (minimal interface)
  - `_prepare_metadata_args(...)` uses `op_id` + record args directly
  - Special-case handling in `_prepare_metadata_args(...)` uses `op_id` string checks (`"crop"`, `"speed_change"`) instead of `isinstance(...)`
  - Crop normalized-float conversion remains
  - SpeedChange ramp averaging uses `args["speed"]` + `args.get("end_speed")`
- [ ] Remove Phase 1 class/introspection-based validation helpers and cache (hard requirement):
  - `_extract_transform_args` / `_extract_init_args`
  - `_get_metadata_method_for_class`
  - `_SPEC_CACHE`

### B. Implement JSON parsing entrypoints

- [ ] Add `VideoEdit.from_dict(data: dict) -> VideoEdit`:
  - Validate top-level: `segments` required and non-empty
  - `post_transforms` and `post_effects` optional, default `[]`
  - Ignore unknown top-level keys (forward-compatible)
- [ ] Add `VideoEdit.from_json(text: str) -> VideoEdit`:
  - `json.loads(text)` with wrapped decode error
  - Delegates to `from_dict`
- [ ] Parse each segment:
  - Required: `source` (str -> Path), `start` (number), `end` (number)
  - Optional: `transforms` (list, default `[]`), `effects` (list, default `[]`)
  - Reject unknown segment keys
- [ ] Coerce/validate `start`, `end` are numeric (not strings)
- [ ] Build `SegmentConfig` / `VideoEdit` in normalized `_StepRecord`-backed form (not object-list + optional sidecars)

### C. Parse registry-backed operation steps

- [ ] Implement `_parse_transform_step(step: dict, location: str) -> _StepRecord`:
  - Required key: `op`
  - Optional key: `args` (default `{}`)
  - Reject `apply` key on transforms (error: "transforms do not accept apply params")
  - Reject unknown keys in step dict
  - Resolve `op` via `get_operation_spec()`, canonicalize to `spec.id`
  - Enforce support-boundary checks inline (see Section E for check order):
    1. category check (must be `TRANSFORMATION`)
    2. tag check (`multi_source`, `multi_source_only`)
    3. JSON-instantiability check
  - Instantiate via `importlib.import_module(spec.module_path)` + `getattr(..., spec.class_name)`
  - Validate constructor `args` keys against `spec.params` (required present, no unknowns)
  - **Wrap constructor/import errors**: catch `ImportError`, `TypeError`, `ValueError` from instantiation and re-raise as `ValueError` with location, requested op name, canonical op_id, and the original error message. Use `raise ... from e` to preserve the original traceback for debugging. Since we defer type validation, constructor errors from bad arg types are expected and must be actionable.
  - Return `_StepRecord.create(op_id, args, {}, transform_instance)`
- [ ] Implement `_parse_effect_step(step: dict, location: str) -> _StepRecord`:
  - Required key: `op`
  - Optional: `args` (default `{}`), `apply` (default `{}`)
  - Reject unknown keys in step dict
  - Resolve + canonicalize via registry
  - Enforce support-boundary checks inline (same order as transform parser; category must be `EFFECT`)
  - Instantiate effect (raw `Effect`; do not wrap in `EffectApplication`)
  - Validate `args` keys against `spec.params`, `apply` keys against `spec.apply_params`
  - **Wrap constructor/import errors** (same pattern as transform steps)
  - Return `_StepRecord.create(op_id, args, apply, effect_instance)`
- [ ] Use `location` strings like `segments[1].transforms[0]` for path-specific errors

### D. Validate step args against registry specs (structural only)

- [ ] Check required `spec.params` keys are present in `args`
- [ ] Check no unknown keys in `args` (not in `spec.params` names)
- [ ] Check required `spec.apply_params` keys are present in `apply` (effects)
- [ ] Check no unknown keys in `apply` (effects)
- [ ] Do NOT validate value types against `ParamSpec.json_type` in Phase 2 -- rely on Python constructor to catch type errors at instantiation time

### E. Enforce JSON-plan support boundaries

These checks are performed inside `_parse_transform_step` / `_parse_effect_step` (Section C), not as a separate post-parse validation pass.

**Check order is deterministic** (first failing check produces the error):
1. Category check (reject `TRANSITION`, `SPECIAL`)
2. Tag check (reject `multi_source` / `multi_source_only`)
3. JSON-instantiability check (required params excluded from spec)

This means `picture_in_picture` is rejected for `multi_source` tag (step 2), not for excluded `overlay` param (step 3). Tests should assert the specific error message.

- [ ] Reject `TRANSITION` and `SPECIAL` category ops with clear message
- [ ] Reject `multi_source` / `multi_source_only` tagged ops with message naming the tag
- [ ] Detect non-JSON-instantiable ops (required constructor params excluded from spec):
  - Get actual required `__init__` param names (params with no default, excluding `self`)
  - Compare against `spec.params` names
  - If any required param is missing from spec, op is not JSON-instantiable
  - Error: "Operation '{op_id}' is registered but not JSON-instantiable because required constructor parameter '{param}' is not included in the registry spec."
  - This catches: `ken_burns` (excluded `start_region`, `end_region`), `full_image_overlay` (excluded `overlay_image`)

### F. AI lazy-registration compatibility

- [ ] When `get_operation_spec(op)` returns `None`:
  - Error message: "Unknown operation '{op}'. If this is an AI operation (e.g. face_crop, auto_framing), ensure `import videopython.ai` is called before parsing the plan."
- [ ] Do NOT auto-import `videopython.ai` (preserve base import isolation)
- [ ] Add test asserting the hint message text on unknown op

### G. Implement `VideoEdit.to_dict()`

- [ ] Return canonical JSON-ready dict with structure:
  ```python
  {
      "segments": [...],
      "post_transforms": [...],
      "post_effects": [...]
  }
  ```
- [ ] For each transform/effect step, serialize from `_StepRecord` only:
  - `op_id`
  - deep-copied `args`
  - deep-copied `apply_args`
- [ ] No introspection fallback path in Phase 2 (single invariant, simpler implementation)
- [ ] Transform step format: `{"op": op_id, "args": {...}}`
- [ ] Effect step format: `{"op": op_id, "args": {...}, "apply": {...}}`
- [ ] Omit `args`/`apply` keys when empty dict (cleaner output, optional)
- [ ] Segment format: `{"source": str, "start": float, "end": float, "transforms": [...], "effects": [...]}`

### H. Tests (`src/tests/base/test_video_edit.py`)

Add Phase 2 tests in existing file (split to `test_video_edit_json.py` if it gets large).

- [ ] Rewrite Phase 1 construction-heavy tests to use `from_dict` / `from_json` as the primary construction path (execution/validation assertions stay, construction API changes)

- [ ] **Parsing happy paths**
  - Single-segment plan, no ops
  - Plan with transform and effect steps (including `apply` bounds)
  - Alias input (`blur`) canonicalizes to `blur_effect` in parsed record
  - Post-transforms and post-effects
  - `from_json` string input
  - `_StepRecord` snapshot semantics: mutate live op after parse, `to_dict()` still emits original parsed values
- [ ] **Parse + validate + run integration**
  - `from_dict` -> `validate()` -> returns correct predicted metadata
  - `from_dict` -> `run()` -> produces Video output (test on real video assets)
  - Parsed plan with `speed_change` validates via registry metadata_method
  - `from_dict` -> `run()` is treated as the primary construction/execution path in Phase 2
- [ ] **Structural validation errors**
  - Missing `segments` key
  - Empty segments list
  - Missing `source` / `start` / `end`
  - Unknown segment key (e.g. `transforms` typo as `tranforms`)
  - Non-numeric `start` / `end`
  - Invalid `transforms` / `effects` type (not list)
- [ ] **Step validation errors**
  - Missing `op` key
  - Unknown step key
  - Transform step with `apply` key rejected
  - Missing required constructor arg
  - Unknown constructor arg
  - Unknown `apply` arg on effect
- [ ] **Registry/category/tag errors**
  - Unknown op includes AI lazy-registration hint
  - Transform list containing an effect op
  - Effect list containing a transform op
  - Transition op rejected
  - `multi_source` / `multi_source_only` tagged op rejected
  - Non-JSON-instantiable op (`ken_burns`, `full_image_overlay`) rejected with clear message
- [ ] **Serialization tests**
  - `from_dict(...).to_dict()` roundtrip stability (canonical output)
  - `to_dict()` emits deep copies (mutating returned dict does not alter future `to_dict()` output)
  - Alias input serializes as canonical op ID
- [ ] **Normalized storage invariants**
  - Internal step storage is `_StepRecord`-backed after construction/parsing

### I. Docs and release notes

- [ ] Remove `EffectApplication` from `videopython.base.__all__` exports and docs in Phase 2 (records carry effect apply bounds)
- [ ] Add `docs/api/editing.md`:
  - `VideoEdit` and `SegmentConfig` API reference
  - JSON plan schema with annotated example
  - `from_dict` / `from_json` / `to_dict` usage
  - Alias normalization behavior
  - Unsupported ops in JSON plans (multi-source, special, excluded-arg ops)
  - Effect time semantics: segment-local vs post-assembly timeline
- [ ] Link from `docs/api/index.md`
- [ ] Add short README example: JSON plan -> `from_dict` -> `validate` -> `run`
- [ ] Add release notes entry

## Phase 2 Implementation Sequence

1. Refactor `SegmentConfig` / `VideoEdit` internals to `_StepRecord`-backed storage
2. Implement step parsers (`_parse_transform_step`, `_parse_effect_step`) with registry/category/tag/instantiability checks
3. Implement `VideoEdit.from_dict` / `from_json` (construct normalized record-backed objects)
4. Implement `VideoEdit.to_dict` (records-only serialization)
5. Add tests for parsing/serialization/roundtrip/integration
6. Add docs + release notes

## Phase 2 Acceptance Criteria

- [ ] `VideoEdit.from_dict(plan)` builds a runnable/validatable `VideoEdit` for supported base transform/effect plans
- [ ] `VideoEdit.from_json(text)` wraps JSON decode and parse errors with actionable messages
- [ ] `VideoEdit.to_dict()` emits canonical op IDs and preserves effect `apply` bounds
- [ ] `from_dict(...).to_dict()` roundtrips stably for supported plans
- [ ] `_StepRecord` snapshot semantics hold (live-op mutation and returned-dict mutation do not affect future `to_dict()` output)
- [ ] Unsupported categories/tags/non-JSON-instantiable ops fail with clear, path-specific messages
- [ ] Unknown ops mention AI lazy registration (`import videopython.ai`) in the error guidance
- [ ] Base import isolation remains intact (`videopython.base` does not import `videopython.ai`)

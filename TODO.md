# TODO: `VideoEdit` Editing Plan Representation

## Status

### Phase 1 (implemented)

Core execution and dry-run validation landed:

- `src/videopython/base/edit.py`
  - `SegmentConfig.process_segment()`
  - `VideoEdit.run()`
  - `VideoEdit.validate()` with `VideoMetadata` prediction
- `src/videopython/base/video.py`
  - `VideoMetadata.speed_change()` metadata prediction
- `src/videopython/base/registry.py`
  - `metadata_method` populated for core transforms used by validation
- `src/tests/base/test_video_edit.py`
  - execution + validation coverage

### Phase 2 (implemented)

Registry-backed JSON parsing and canonical serialization landed (records-first internals):

- `VideoEdit.from_dict(...)`
- `VideoEdit.from_json(...)`
- `VideoEdit.to_dict()`
- `_StepRecord` as the internal source of truth (canonical `op_id`, parsed `args`, parsed `apply_args`, live op object)
- Registry-driven parsing with category/tag/JSON-instantiability checks
- Path-specific parse errors and AI lazy-registration hint
- Records-first tests in `src/tests/base/test_video_edit.py`

## Current Architecture Facts (Phase 3 baseline)

- `VideoEdit` internals are `_StepRecord`-backed (`src/videopython/base/edit.py`).
- `EffectApplication` is no longer part of the `VideoEdit` internal model.
- `from_dict` / `from_json` are the intended construction path; `SegmentConfig` remains exported but is not the primary user-facing constructor API.
- Parsing currently validates structure (required/unknown keys) and support boundaries (category/tags/JSON-instantiability), then instantiates operations.
- Effect `apply.start` / `apply.stop` are validated/coerced at parse time because they bypass constructor validation.
- Value/type validation for constructor/apply args is still partial:
  - no generic `ParamSpec.json_type`/`enum`/`minimum`/`maximum`/`items_type` enforcement yet
  - JSON values are passed into constructors mostly unchanged
- `OperationSpec` / `ParamSpec` already support per-operation JSON schema generation for constructor/apply args (`spec.to_json_schema()`, `spec.to_apply_json_schema()`).
- There is no `VideoEdit` plan-level JSON schema API yet.
- There is no `docs/api/editing.md` yet, and `docs/api/index.md` does not link editing-plan docs.
- There is no `jsonschema` dependency in `pyproject.toml`.

## Phase 3 Goal

Make `VideoEdit` JSON plans self-describing and safer to consume by adding:

1. Full plan-level JSON schema generation (registry-backed, canonical, parser-aligned)
2. Stronger parse-time value validation and normalization (type/enum/range/items)
3. Docs + release notes for the JSON plan workflow

This phase should iterate on the current Phase 1/2 implementation without backward-compatibility constraints.

## Phase 3 Scope

### In scope

- `VideoEdit` plan-level JSON schema generation API
- Shared filtering logic so schema and parser expose the same supported operation set
- Parse-time value validation against `ParamSpec` metadata:
  - `json_type`
  - `enum`
  - `minimum` / `maximum`
  - `items_type` (array elements)
- Parse-time JSON-to-Python normalization for registry-serializable constructor args where needed (notably enums / tuples)
- Registry metadata enhancements needed to express schema accurately (nullable support)
- Tests for schema generation + value validation + normalization
- Docs and release notes

### Explicitly out of scope (Phase 3)

- Transitions in `VideoEdit` JSON plans
- Auto-normalization of incompatible segment outputs (resize/fps harmonization)
- Generic resolver hooks for non-JSON constructor args (numpy arrays, `BoundingBox`, raw `Video`)
- Full semantic validation of every operation argument beyond available registry metadata (constructors remain the final authority)
- Introducing automatic `videopython.ai` imports from `videopython.base`

## Phase 3 Locked Decisions

- No backward compatibility constraints: iterate directly on the Phase 2 records-first design.
- No runtime `jsonschema` dependency in `videopython.base` for Phase 3.
  - Schema generation returns plain dicts.
  - Parsing/validation remains internal code.
- `VideoEdit` remains the public entrypoint for plans (`from_dict`, `from_json`, `to_dict`).
- Add a class-level schema API on `VideoEdit` (recommended: `VideoEdit.json_schema()`).
- Schema and parser must share the same support-boundary rules (category/tag/JSON-instantiability) to avoid drift.
- Canonical op IDs only in schemas and serialization (aliases accepted on input only).
- Top-level unknown keys remain ignored by parser (forward-compatible); generated top-level schema should reflect this.
- Step-level and segment-level unknown keys remain rejected; generated schemas should reflect this.
- `to_dict()` remains canonical output, not necessarily byte-for-byte identical to original JSON if parse-time normalization occurs.

## Phase 3 Detailed Implementation Plan

### A. Refactor parser support checks into reusable helpers (schema + parser alignment)

Current parser support checks live inline in `_resolve_and_validate_step_spec(...)` plus `_ensure_json_instantiable(...)`.
Phase 3 needs the same rules when generating the plan schema.

- [ ] Extract a reusable helper for JSON-plan support checks on a resolved `OperationSpec`:
  - Example shape:
    - `_ensure_videoedit_json_supported(spec, expected_category, location, op_cls=None)`
    - or split into:
      - `_validate_videoedit_step_category(...)`
      - `_validate_videoedit_step_tags(...)`
      - `_ensure_json_instantiable(...)`
- [ ] Add a schema-facing predicate/helper that uses the same rules but returns bool/reason instead of raising:
  - Example: `_is_videoedit_json_supported_spec(spec, expected_category) -> tuple[bool, str | None]`
- [ ] Keep deterministic check order consistent across parser and schema filtering:
  1. category
  2. tags (`multi_source`, `multi_source_only`)
  3. JSON-instantiability
- [ ] Reuse `_ensure_json_instantiable(...)` in both parser and schema generation (schema generation will need `op_cls`, so class loading happens there too).
- [ ] Preserve AI lazy-registration behavior:
  - schema generation only includes AI ops if already registered
  - no auto-import of `videopython.ai`

### B. Add registry metadata support for nullability (needed for accurate schema + value validation)

Current `ParamSpec` loses `Optional[...]` / `None` information (`float | None` becomes `"number"` only).
That makes generated schemas for effect apply params (`start`, `stop`) stricter than current parser behavior.

- [ ] Extend `ParamSpec` with `nullable: bool = False`
- [ ] Update `ParamSpec.to_json_schema()` to emit nullable JSON schema:
  - Option 1 (recommended for broad compatibility): `"type": [json_type, "null"]`
  - Option 2: `anyOf` with `{type: ...}` + `{type: "null"}`
- [ ] Update `_annotation_to_schema(...)` to return nullability info (or add a separate nullable detector)
- [ ] Preferred implementation: update `_annotation_to_schema(...)` to return nullability info directly (4-tuple/expanded return shape) instead of adding a separate nullable detector, so annotations are inspected once.
  - `float | None` => `json_type="number", nullable=True`
  - `Enum | None` => enum schema + nullable
  - `Literal[...] | None` => literal schema + nullable
- [ ] Update `_spec_params_from_callable(...)` / `spec_from_class(...)` plumbing for the new return shape
- [ ] Add/adjust registry tests in `src/tests/base/test_registry.py`:
  - `blur_effect` apply params `start`/`stop` are nullable
  - `spec.to_apply_json_schema()` exposes nullable schema for `start`/`stop`

### C. Implement `VideoEdit` plan-level JSON schema generation

Add a schema API that downstream systems (LLM orchestration, UI builders, validation layers) can use directly.

- [ ] Add `@classmethod VideoEdit.json_schema(cls) -> dict[str, Any]` in `src/videopython/base/edit.py`
  - Returns a canonical JSON Schema for the full plan object
  - Must not instantiate or load videos
- [ ] Build plan schema from registry specs dynamically:
  - gather supported transform specs (`OperationCategory.TRANSFORMATION`)
  - gather supported effect specs (`OperationCategory.EFFECT`)
  - apply the same support-boundary filtering as parser (category/tag/JSON-instantiability)
- [ ] Generate per-op step schemas:
  - transform step schema shape:
    - `{"type":"object","properties":{"op":{"const": "..."},"args": ...},"required":["op"],"additionalProperties":False}`
    - no `apply` property
  - effect step schema shape:
    - same plus `apply` from `spec.to_apply_json_schema()`
  - include operation descriptions (where useful) from `OperationSpec.description`
- [ ] Step schema required-ness must reflect registry param requirements (parser-aligned):
  - If `spec.params` contains any required params, include `"args"` in step-schema `required`
  - Otherwise `args` remains optional
  - If `spec.apply_params` contains any required params (effects), include `"apply"` in step-schema `required`
  - Otherwise `apply` remains optional
- [ ] Use canonical IDs only in schema (`spec.id`), not aliases
- [ ] Compose union schemas:
  - `transforms.items = {"oneOf": [per-op transform step schemas...]}`
  - `effects.items = {"oneOf": [per-op effect step schemas...]}`
  - same for `post_transforms` / `post_effects`
- [ ] Generate segment schema aligned with parser:
  - required: `source`, `start`, `end`
  - optional: `transforms`, `effects`
  - `additionalProperties: false`
- [ ] Generate top-level plan schema aligned with parser:
  - required: `segments`
  - `segments` minItems = 1
  - optional: `post_transforms`, `post_effects`
  - top-level unknown keys allowed (do not set `additionalProperties: false`)
- [ ] Decide schema draft marker and document it (recommended: include `$schema` draft-07 URI for compatibility)
- [ ] Add helper internals (private) to keep schema code readable:
  - `_videoedit_step_schema_from_spec(...)`
  - `_videoedit_supported_specs_for_category(...)`
  - `_videoedit_plan_schema_components(...)` (optional)

### D. Add parse-time value validation against `ParamSpec` metadata

Phase 2 validates structure and relies on constructors for most bad values. Phase 3 should catch JSON-type/enum/range errors before constructor execution for clearer messages and better LLM feedback loops.

- [ ] Add a reusable validator for arg maps against `ParamSpec` metadata:
  - Example: `_validate_param_values(value: Mapping[str, Any], params: tuple[ParamSpec, ...], location: str)`
- [ ] Validate per-key values after structural validation and before instantiation:
  - `_parse_transform_step(...)`: `args`
  - `_parse_effect_step(...)`: `args` and `apply`
- [ ] Type validation rules (JSON-centric, path-specific errors):
  - `integer`: `int` only (reject `bool`)
  - `number`: `int | float` (reject `bool`)
  - `string`: `str`
  - `boolean`: `bool`
  - `array`: `list` (JSON arrays parse as Python lists)
  - `object`: `dict`
- [ ] Array item validation (when `items_type` is set):
  - validate each element type
  - error includes index path (e.g. `...kernel_size[1]`)
- [ ] Enum validation:
  - if `ParamSpec.enum` is set, require membership in enum values
  - error message shows allowed values
- [ ] Numeric range validation:
  - enforce `minimum` / `maximum` when present
  - keep constructor validation as the source of truth for stricter/non-linear constraints
- [ ] Respect `nullable=True` (if implemented in Section B)
  - `None` accepted only for nullable params
- [ ] Preserve deterministic error precedence:
  - support-boundary checks -> structural key checks -> value validation -> constructor/import errors

### E. Add parse-time JSON-to-Python normalization for constructor args (Enum / tuple support)

Phase 2 stores JSON snapshots and instantiates constructors with raw JSON values. Some constructors accept Python types that do not map 1:1 from JSON (e.g. enum members, tuples).
This is already observable with `Crop.mode` (JSON string vs `CropMode`) and tuple-like params such as `Blur.kernel_size`.

- [ ] Add a constructor-arg normalizer using `op_cls.__init__` type hints (applied to a copy):
  - Example: `_normalize_constructor_args_for_class(op_cls, args, location) -> dict[str, Any]`
- [ ] Normalization rules (Phase 3 targeted set):
  - `Enum` parameters: convert JSON enum value (string/number) to enum member
  - `tuple[...]` parameters: convert JSON list to tuple (shallow conversion)
  - `tuple[T, T]` / `tuple[int, int]`: keep element types validated by Section D, convert container only
  - Leave unsupported complex annotations unchanged (constructor will remain fallback validator)
- [ ] Apply normalization only to constructor call inputs, not `_StepRecord.args`
  - `_StepRecord.args` must remain the canonical parsed JSON snapshot used by `to_dict()`
- [ ] Keep `apply` arg normalization separate:
  - `apply` stays JSON-native in `_StepRecord.apply_args` (already parse-time numeric coercion for `start`/`stop`)
  - optionally collapse `_normalize_effect_apply_args(...)` into the new validation/normalization flow for consistency
- [ ] Add explicit tests for:
  - `crop.mode: "center"` produces a `Crop` op with `CropMode.CENTER` internally and validates/runs correctly
  - invalid enum literal for `crop.mode` fails at parse time with allowed values listed
  - `blur_effect.kernel_size: [5, 5]` is accepted and normalized to tuple for constructor input (while `to_dict()` preserves JSON array snapshot)

### F. Enrich registry constraints with `param_overrides` where high-value (optional but recommended in Phase 3)

The value validator can enforce `minimum` / `maximum`, but current base registrations mostly do not populate these fields.
Adding a small, high-signal set of overrides will improve schema usefulness and parse-time feedback.

- [ ] Add `param_overrides` / `apply_param_overrides` for core base ops where constraints are clear and stable
  - Examples (illustrative; confirm exact constraints from constructors/runtime behavior):
    - `cut.start`, `cut.end` minimum `0`
    - `cut_frames.start`, `cut_frames.end` minimum `0`
    - `resample_fps.fps` minimum `1`
    - `speed_change.speed` minimum `0` (schema cannot express strict `> 0` with current `ParamSpec`; constructor remains authority for rejecting `0`)
    - `speed_change.end_speed` minimum `0` (same caveat; constructor enforces strict positivity)
    - `blur_effect.iterations` minimum `1`
    - effect `apply.start`, `apply.stop` minimum `0`
- [ ] Prefer exact bounds only; avoid speculative limits that constructors do not enforce
- [ ] Add registry tests verifying selected overrides appear in generated schemas

### G. Update `VideoEdit` parser internals to use new validation/normalization flow

Wire the new pieces into the existing parser with minimal churn to the records-first model.

- [ ] `_parse_transform_step(...)`
  - keep current support-boundary + structural checks
  - add value validation for `args`
  - normalize constructor args into a temporary dict
  - instantiate using normalized constructor args
  - keep `_StepRecord.create(spec.id, original_args, {}, operation)` using original parsed JSON args
- [ ] `_parse_effect_step(...)`
  - keep current support-boundary + structural checks
  - add value validation for `args` and `apply`
  - normalize constructor args into temporary dict
  - normalize `apply` args through existing/start-stop path (or unified normalizer)
  - instantiate effect using normalized constructor args
  - keep `_StepRecord` snapshots JSON-native/canonical
- [ ] Revisit helper naming after Phase 3 wiring:
  - avoid overlapping “validate” vs “normalize” responsibilities
  - make call order obvious in code

### H. Tests (`src/tests/base/`)

Add focused Phase 3 coverage. Keep records-first Phase 2 tests and extend them rather than introducing compatibility layers.

- [ ] `src/tests/base/test_video_edit.py` (or split `test_video_edit_schema.py` if it gets too large)

- [ ] **Plan schema generation**
  - `VideoEdit.json_schema()` returns object schema with required `segments`
  - `segments` has `minItems: 1`
  - top-level unknown keys are allowed (schema matches parser)
  - segment/step `additionalProperties: false`
  - transform step schema excludes `apply`
  - effect step schema includes `apply`
  - aliases are not emitted (canonical IDs only)
  - unsupported ops excluded from schema:
    - transitions
    - `multi_source` / `multi_source_only`
    - non-JSON-instantiable (`ken_burns`, `full_image_overlay`) with explicit coverage for both transform/effect filtering paths
  - AI ops absent before `import videopython.ai`, present after import (if supported)

- [ ] **Value validation**
  - wrong primitive types rejected with path-specific messages (`.args.width`, `.apply.start`, etc.)
  - `bool` rejected where `number`/`integer` expected
  - enum validation for `crop.mode`
  - array element type validation for tuple-backed params like `blur_effect.kernel_size`
  - numeric min/max enforcement for any registry overrides added in Section F
  - nullable params (`apply.start`, `apply.stop`) accept `null` if Section B is implemented

- [ ] **Normalization**
  - enum coercion for constructor args (`crop.mode`)
  - tuple coercion for constructor args (`blur_effect.kernel_size`)
  - `to_dict()` still emits JSON-native snapshots after normalization (e.g. list remains list)

- [ ] **Parser/schema alignment smoke tests**
  - op IDs accepted by parser are represented in schema for supported categories
  - unsupported ops rejected by parser are absent from schema

- [ ] **Regression checks**
  - Phase 2 `from_dict -> validate -> run` happy paths still pass
  - import isolation (`src/tests/base/test_import_isolation.py`) still passes

### I. Docs and release notes (carry forward the unfinished Phase 2 docs work)

- [ ] Add `docs/api/editing.md`
  - `VideoEdit` overview (segments -> concat -> post-ops)
  - JSON plan format with annotated examples
  - `from_dict`, `from_json`, `to_dict`
  - `validate()` dry-run usage and compatibility checks
  - `VideoEdit.json_schema()` usage (LLM/UI integration)
  - alias normalization behavior (input aliases accepted, output/schema canonicalized)
  - unsupported ops and AI lazy-registration note (`import videopython.ai`)
  - effect time semantics (segment-local vs post-assembly)
- [ ] Link `docs/api/editing.md` from `docs/api/index.md`
- [ ] Add a short example to `docs/examples/social-clip.md` or a new example showing JSON plan -> validate -> run
- [ ] Add release notes entry in `RELEASE_NOTES.md` for the next version:
  - `VideoEdit` JSON schema generation
  - stronger parse-time plan validation

## Suggested Phase 3 Implementation Sequence (PR-friendly)

1. Add `ParamSpec.nullable` + registry/schema plumbing + tests
2. Add reusable VideoEdit support-filter helpers (parser/schema shared)
3. Implement `VideoEdit.json_schema()` + schema tests
4. Implement `ParamSpec`-based value validation in parser
5. Implement constructor arg normalization (enum/tuple) + tests
6. Add selected registry `param_overrides` for high-value constraints
7. Add docs + release notes

## Phase 3 Acceptance Criteria

- [ ] `VideoEdit.json_schema()` returns a parser-aligned plan schema using canonical op IDs only
- [ ] Generated plan schema excludes unsupported categories/tags/non-JSON-instantiable ops
- [ ] Parser rejects bad arg values (type/enum/range/items) with path-specific errors before constructor execution
- [ ] JSON enum/tuple values needed by constructors are normalized correctly for instantiation while `to_dict()` preserves JSON-native snapshots
- [ ] AI lazy-registration behavior remains unchanged (no base auto-import), and schema/parser both reflect only currently registered ops
- [ ] Existing Phase 2 parse/validate/run behavior remains intact for supported plans
- [ ] Docs for `VideoEdit` JSON plans and schema generation are published and linked from `docs/api/index.md`

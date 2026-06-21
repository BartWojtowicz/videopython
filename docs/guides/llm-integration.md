# LLM & AI Agent Integration

videopython is designed to be controlled by LLMs. Every operation is a
Pydantic `BaseModel` whose fields ARE the JSON wire format, so structural
rules, parameter types, and value constraints surface as standard JSON
Schema. An LLM can generate, validate, and execute editing plans without
needing to learn the surface from examples.

There are three ways to put an LLM in the loop:

1. **Bring your own LLM** *(this guide)* — videopython hands your model the JSON
   Schema and the validate/repair/normalize refine loop; your model authors plans.
2. **[Automatic Editing](auto-editing.md)** — `AutoEditor` runs a *local* Ollama
   vision model as the planner; give it sources + a brief and it returns a cut.
3. **[MCP Server](mcp.md)** — `videopython-mcp` exposes the pipeline as Model
   Context Protocol tools so an external agent (its own model the planner) drives it.

The rest of this guide covers approach 1: the schema, the refine loop, and the
operation registry the other two modes build on.

## Workflow

1. **Generate** — pass `VideoEdit.json_schema()` to the LLM as a tool /
   structured-output schema.
2. **Validate** — call `edit.validate()` for a dry-run via metadata. No
   frames load.
3. **Execute** — `edit.run_to_file()` streams directly to disk.

```python
from videopython.editing import VideoEdit

schema = VideoEdit.json_schema()
plan = call_your_llm(schema=schema,
                     prompt="Create a 15s highlight reel from input.mp4")

edit = VideoEdit.from_dict(plan)
predicted = edit.validate()           # catches bad plans before any I/O
print(predicted)
edit.run_to_file("output.mp4")
```

## Passing the Schema

`VideoEdit.json_schema()` returns a JSON Schema (Draft-07 compatible)
covering segments, post-operations, the matching flags, and a
discriminated union over every **LLM-exposed** `Operation`. Server-only
ops (those needing a server-resolved path, e.g. `image_overlay` /
`full_image_overlay`) are excluded by default so the model never emits a
plan it cannot fill in; pass `include_server_only=True` to
`Operation.json_schema()` for the full union. AI ops appear in the union
only after `import videopython.ai`.

For providers with a strict structured-output **grammar**
(`response_format: json_schema`, strict mode), pass `strict=True`:
`VideoEdit.json_schema(strict=True)` / `Operation.json_schema(strict=True)`
emit a submittable closed grammar (every object `additionalProperties:
false`, every property `required`, the op union as an `anyOf` of closed
variants with no `discriminator`, and the union's `$defs` hoisted to the
document root so every `$ref` resolves). Optionality follows the Pydantic
type — a genuinely optional field stays nullable, a defaulted-but-required
one keeps its concrete type — so a grammar-valid response always parses
back. Grammar-constraining
the decode makes a whole class of bound violations (`window.start >= 0`,
enums, required fields) impossible up front. Cross-field constraints
(`timestamp < duration`, segment-dim equality) can't live in a grammar —
those stay with `check()` / `repair()` / `normalize_dimensions()`.

### Anthropic tool use

```python
import anthropic
from videopython.editing import VideoEdit

client = anthropic.Anthropic()
schema = VideoEdit.json_schema()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=[{
        "name": "create_video_edit",
        "description": "Create a video editing plan",
        "input_schema": schema,
    }],
    messages=[{"role": "user", "content":
               "Cut input.mp4 to the first 10 seconds, resize to 1080x1920, fade in."}],
)

tool_block = next(b for b in response.content if b.type == "tool_use")
edit = VideoEdit.from_dict(tool_block.input)
edit.validate()
edit.run_to_file("output.mp4")
```

### OpenAI function calling

```python
import json
from openai import OpenAI
from videopython.editing import VideoEdit

client = OpenAI()
schema = VideoEdit.json_schema()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a video editor."},
        {"role": "user", "content":
         "Cut input.mp4 to the first 10 seconds, resize to 1080x1920, fade in."},
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "create_video_edit",
            "description": "Create a video editing plan",
            "parameters": schema,
        },
    }],
)

plan = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
edit = VideoEdit.from_dict(plan)
edit.validate()
edit.run_to_file("output.mp4")
```

## Discovering Operations

```python
from videopython.editing import Operation, OpCategory

# All registered ops
for op_id, cls in Operation.registry().items():
    doc = (cls.__doc__ or "").splitlines()[0].strip()
    print(f"{op_id}: {doc}")

# By category
transforms = {k: v for k, v in Operation.registry().items()
              if v.category is OpCategory.TRANSFORM}

# Per-op JSON Schema: model_json_schema() is the full Pydantic schema;
# llm_json_schema() is the LLM-facing variant (drops `llm_hidden` advanced
# fields like raw font paths), so prefer it for tool/function definitions.
Operation.get("color_adjust").model_json_schema()
Operation.get("text_overlay").llm_json_schema()
```

For per-op tool definitions, enumerate `Operation.llm_registry()` (the
LLM-safe subset of `registry()` — it omits server-only ops the model
can't fill in):

```python
tools = []
for op_id, cls in Operation.llm_registry().items():
    if cls.category is not OpCategory.TRANSFORM:
        continue
    tools.append({
        "name": f"transform_{op_id}",
        "description": (cls.__doc__ or "").splitlines()[0],
        "input_schema": cls.llm_json_schema(),   # drops llm_hidden advanced fields
    })
```

## Validation Before Execution

### Parse vs. validate

Parsing (`from_dict`) owns the **shape**: field types, required fields,
unknown ops, extra fields, and op-local structural rules (e.g. `resize`
needs at least one dimension) all surface as a Pydantic `ValidationError`.
The numeric **bounds** of the plan skeleton — segment `start`/`end` and
effect `window` ranges — are deliberately *not* enforced at parse. A
negative `window.start` or a `start >= end` segment now *parses fine* and
is reported by validation instead. This keeps one code path for the
refine loop and makes every numeric violation collectable and repairable.

`VideoEdit.validate()` (or `validate_with_metadata(meta)` to avoid disk)
chains each op's `predict_metadata` across the plan and checks segment
bounds, effect windows, and concatenation compatibility. It raises
`PlanValidationError` on the **first** failure — a `ValueError` subclass
(so `except ValueError` still works) carrying structured `PlanError`s:
`code` (a small enum), `location` (e.g. `"segments[1].operations[0]"`),
`field`, `value`, `limit`. Branch on `code` instead of matching prose.

### Collect every error at once: `check()`

For a refine loop, raising on the first problem means whack-a-mole across
your retry budget. `check()` is the non-raising sibling: it runs the same
dry-run but **accumulates every error in one pass** and returns the
`PlanError` list (empty means valid). Every plan-validation failure is
structured, so nothing escapes as a bare `ValueError`.

```python
errors = edit.check(source_metadata)          # [] == valid
for err in errors:
    print(f"{err.code} at {err.location}: {err.field}={err.value}")
# Re-prompt once with the full structured list instead of one-at-a-time
```

Ops that cannot stream at their plan position are real plan errors:
`check()` reports one `STREAMING_UNSUPPORTED` error per offending op, with
the actionable cause in
`err.detail` (e.g. "move the op before the duration-changing transform to
stream"), and the refine loop treats "won't stream" like any other
violation. The full per-op classification (including the ops that do
stream) is `edit.streamability()`.

### Auto-repair the mechanical violations: `repair()`

Most mechanical faults need no LLM at all — clamping is the obvious fix.
`repair(source_metadata)` returns a corrected copy of the plan plus a
structured changelog (`list[PlanRepair]`), clamping only the unambiguous
cases and never inventing intent:

- effect `window.start`/`window.stop` into `[0, duration]` (negatives →
  `0`, overruns → `duration`), for both segment and `post_operations`;
- time-valued op params past the clip end (e.g. `freeze_frame.timestamp`)
  into range — generic via each op's declared bounded time fields;
- a negative segment `start` → `0`;
- with `clamp_segment_end=True`, a segment `end` past the source → the
  source end (off by default, since it changes editorial intent).

```python
fixed_edit, repairs = edit.repair(source_metadata)   # clamp the mechanical majority
for r in repairs:
    print(f"{r.code}: {r.location}.{r.field} {r.old} -> {r.new}")  # surface to the user
```

Genuinely semantic problems (a concat dimension mismatch, an
`end <= start` range) are left intact for re-prompting, so always
`check()`/`validate()` the returned plan before running it. `repair()`
never raises on an unrepairable op — it leaves it for `check()` to report.

### Normalize concat geometry: `normalize_dimensions()`

The one class you cannot cleanly repair in your own layer is a
`CONCAT_MISMATCH` — detecting it needs each segment's *predicted post-op*
dimensions and fixing it needs a per-segment resize inserted *before*
concat. `normalize_dimensions(source_metadata, target)` does it for you,
appending a `resize` to every segment whose predicted output differs from
the `target` (an explicit `(w, h)`, `"first"`, or `"largest"`) and
returning the same `PlanRepair` changelog. The "all segments share
dimensions" invariant becomes satisfiable by construction. Like `repair()`
and `check()` it is best-effort and non-raising: a segment it can't predict
yet is left untouched and deferred to `check()`.

```python
norm_edit, repairs = edit.normalize_dimensions(source_metadata, (1080, 1920))
```

### The full refine loop

```python
edit = VideoEdit.from_dict(plan)                       # permissive parse
edit, repairs = edit.repair(source_metadata)           # clamp the mechanical ones
edit, dim_repairs = edit.normalize_dimensions(source_metadata, "largest")
errors = edit.check(source_metadata)                   # whatever's left, all at once
if errors:
    ...  # re-prompt with the previous plan + the full structured error list
```

`check()` and `normalize_dimensions()` never raise. `repair()` raises in exactly
one case — a segment `end` past the source — which it treats as an intent error
to re-prompt; pass `clamp_segment_end=True` if you'd rather clamp it to the source
end and keep the loop raise-free. `source_metadata` leads each call for a
consistent signature across the family.

A clampable `window.stop` overrun (a duration-shrinking op like `speed_change` /
`freeze_frame` ordered before a windowed effect leaves the stop past the
now-shorter clip) is the one case `run_to_file()` already tolerates by clamping.
`validate(clamp_windows=True)` / `check(..., clamp_windows=True)` won't
report it either; `repair()` clamps it in the returned plan.

## Context Data

Operations that need side-channel data declare it via
`requires: ClassVar[tuple[str, ...]]`. The runner pulls matching keys
out of the `context` dict and threads them into the op:

```python
# silence_removal and add_subtitles both need a transcription
edit = VideoEdit.from_dict(plan)
# context-requiring effects stream too
edit.run_to_file("out.mp4", context={"transcription": transcription})
```

Time-based context values (e.g. a `Transcription` with source-absolute
timestamps) are re-based onto each cut segment's local timeline
automatically, on the streaming engine that backs `run_to_file()`.

Discover requires-aware ops via the registry:

```python
needs_transcript = [op_id for op_id, cls in Operation.registry().items()
                    if "transcription" in cls.requires]
```

## AI Operations

AI-powered ops (`face_crop`, ...) are registered only when
`videopython.ai` is imported. If your plans use them, import AI first
so the schema includes them:

```python
import videopython.ai   # registers AI ops
from videopython.editing import VideoEdit

schema = VideoEdit.json_schema()    # now includes face_crop
```

## Tips

- **Start with the schema.** Pass `VideoEdit.json_schema()` as the tool
  schema — it encodes all structural rules so the LLM doesn't need
  examples.
- **Always validate.** Call `edit.validate()` before `edit.run_to_file()`.
  Validation is fast and catches most errors.
- **Use the error loop.** If validation fails, feed the error back to
  the LLM and ask it to fix the plan. Most issues correct in one retry.
- **Provide source metadata.** Tell the LLM the source duration,
  dimensions, and fps so it can generate sensible time ranges and
  resize targets.
- **Expose the registry.** For agents, let the LLM call into
  `Operation.llm_registry()` instead of hardcoding the op list — it omits
  server-only ops the model can't supply. Use `Operation.registry()` only
  when you need *every* op (e.g. the worker that executes a stored plan).

# LLM & AI Agent Integration

videopython is designed to be controlled by LLMs. Every operation is a
Pydantic `BaseModel` whose fields ARE the JSON wire format, so structural
rules, parameter types, and value constraints surface as standard JSON
Schema. An LLM can generate, validate, and execute editing plans without
needing to learn the surface from examples.

## Workflow

1. **Generate** — pass `VideoEdit.json_schema()` to the LLM as a tool /
   structured-output schema.
2. **Validate** — call `edit.validate()` for a dry-run via metadata. No
   frames load.
3. **Execute** — `edit.run()` returns a `Video`; `edit.run_to_file()`
   streams directly to disk.

```python
from videopython.editing import VideoEdit

schema = VideoEdit.json_schema()
plan = call_your_llm(schema=schema,
                     prompt="Create a 15s highlight reel from input.mp4")

edit = VideoEdit.from_dict(plan)
predicted = edit.validate()           # catches bad plans before any I/O
print(predicted)
video = edit.run()
video.save("output.mp4")
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
edit.run().save("output.mp4")
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
edit.run().save("output.mp4")
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

`VideoEdit.validate()` chains each op's `predict_metadata` across the
plan and checks segment bounds, effect windows, and concatenation
compatibility. Catches:

- Invalid time ranges (`start >= end`, `end > source duration`)
- Effect `window` outside the predicted segment duration
- Incompatible segment dimensions/fps for concatenation
- Unknown operation IDs (`Pydantic ValidationError` raised by
  `from_dict`)
- Out-of-range parameter values (also at `from_dict` time)

Validation failures raise `PlanValidationError`, which **subclasses
`ValueError`** (so `except ValueError` still works) and additionally
carries a list of structured `PlanError`s — `code` (a small enum),
`location` (e.g. `"segments[1].operations[0]"`), `field`, `value`,
`limit` — so an agent can branch on the failure class instead of
substring-matching the message:

```python
from videopython.base.exceptions import PlanValidationError

edit = VideoEdit.from_dict(plan)
try:
    predicted = edit.validate()
    print(f"Output: {predicted.width}x{predicted.height}, "
          f"{predicted.total_seconds:.1f}s")
except PlanValidationError as e:
    for err in e.errors:
        print(f"{err.code} at {err.location}: {err.field}={err.value}")
    # Feed `str(e)` (the human message) or `e.errors` back to the LLM to retry
```

This makes it cheap to let an LLM retry: validate, return the error,
ask the LLM to fix it.

### Auto-repairing window overruns

A common, mechanical failure: a duration-shrinking op (`cut`,
`speed_change`, `silence_removal`) ordered *before* a windowed effect
leaves the effect's `window.stop` past the now-shorter clip. `run()`
silently clamps it, but `validate()` raises by default. Pass
`clamp_windows=True` to make `validate()` clamp each overrunning
`window.stop` to the run-time value instead of raising, or call
`edit.repair(source_metadata)` to get back a corrected plan plus the list
of clamps applied — no extra LLM round-trip:

```python
predicted = edit.validate(clamp_windows=True)        # don't reject clampable overruns
fixed_edit, clamps = edit.repair(source_metadata)    # or get a repaired plan back
```

`repair()` clamps `window.stop` only — it is not a full validator (a
`window.start` overrun, concat mismatch, or bad source still stands), so
`validate()` the returned plan before running it. For most flows
`validate(clamp_windows=True)` is the simpler path.

## Context Data

Operations that need side-channel data declare it via
`requires: ClassVar[tuple[str, ...]]`. The runner pulls matching keys
out of the `context` dict and threads them into the op:

```python
# silence_removal and add_subtitles both need a transcription
edit = VideoEdit.from_dict(plan)
video = edit.run(context={"transcription": transcription})
```

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
- **Always validate.** Call `edit.validate()` before `edit.run()`.
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

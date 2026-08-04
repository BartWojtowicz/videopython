# Author edit plans with your own LLM

Use this when your model, in your harness, should author the edits. videopython supplies
the tool schema and a refine loop; you supply the model.

If you would rather videopython run the model, see [Let a local LLM edit for
you](auto-editing.md). If your agent should call the tools itself, see [Drive editing
from an MCP agent](mcp-server.md). The reasoning behind all three is in [LLM-first
design](../explanation/llm-first-design.md).

## The loop in three calls

```python
from videopython.editing import VideoEdit

schema = VideoEdit.json_schema()
plan = call_your_llm(schema=schema, prompt="Create a 15s highlight reel from input.mp4")

edit = VideoEdit.from_dict(plan)
edit.validate()                 # dry run over metadata; no frames touched
edit.run_to_file("output.mp4")
```

## Pass the schema as a tool

`VideoEdit.json_schema()` returns a Draft-07-compatible schema covering segments,
`post_operations`, the matching flags, and a discriminated union over every LLM-exposed
operation. Server-only ops (those needing a server-resolved path, like `image_overlay`)
are excluded so the model cannot emit a plan it is unable to fill in.

=== "Anthropic"

    ```python
    import anthropic
    from videopython.editing import VideoEdit

    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        tools=[{
            "name": "create_video_edit",
            "description": "Create a video editing plan",
            "input_schema": VideoEdit.json_schema(),
        }],
        messages=[{"role": "user", "content":
                   "Cut input.mp4 to the first 10 seconds, resize to 1080x1920, fade in."}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    edit = VideoEdit.from_dict(tool_block.input)
    edit.validate()
    edit.run_to_file("output.mp4")
    ```

=== "OpenAI"

    ```python
    import json
    from openai import OpenAI
    from videopython.editing import VideoEdit

    client = OpenAI()

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
                "parameters": VideoEdit.json_schema(),
            },
        }],
    )

    plan = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    edit = VideoEdit.from_dict(plan)
    edit.validate()
    edit.run_to_file("output.mp4")
    ```

### Strict / grammar-constrained decoding

For providers with a strict structured-output grammar, pass `strict=True`:

```python
VideoEdit.json_schema(strict=True)
Operation.json_schema(strict=True)
```

That emits a submittable closed grammar — every object `additionalProperties: false`,
every property `required`, the op union as an `anyOf` of closed variants with no
`discriminator`, and the union's `$defs` hoisted to the document root so every `$ref`
resolves. Optionality follows the Pydantic type, so a grammar-valid response always
parses back.

Constraining the decode makes a whole class of violations (`window.start >= 0`, enums,
required fields) impossible up front. Cross-field constraints — `timestamp < duration`,
segment dimension equality — cannot live in a grammar and stay with the refine loop
below.

## Refine a plan the model got wrong

Four methods, all taking `source_metadata` first:

```python
edit = VideoEdit.from_dict(plan)                          # permissive parse
edit, repairs = edit.repair(source_metadata)              # clamp the mechanical faults
edit, dim_repairs = edit.normalize_dimensions(source_metadata, "largest")
errors = edit.check(source_metadata)                      # whatever is left, all at once
if errors:
    ...  # re-prompt with the previous plan + the full structured error list
```

| Method | Does | Raises |
|---|---|---|
| `check(meta)` | Collects **every** `PlanError` in one pass; `[]` means valid | never |
| `repair(meta)` | Clamps unambiguous violations, returns `(edit, changelog)` | only on a segment `end` past the source, unless `clamp_segment_end=True` |
| `normalize_dimensions(meta, target)` | Appends a per-segment `resize` so concat geometry matches | never |
| `validate()` / `validate_with_metadata(meta)` | Dry run, first error wins | `PlanValidationError` |

Surface what was changed:

```python
for err in errors:
    print(f"{err.code} at {err.location}: {err.field}={err.value} (limit {err.limit})")

for r in repairs:
    print(f"{r.code}: {r.location}.{r.field} {r.old} -> {r.new}")
```

Branch on `err.code` — a small enum — not on prose. What each method will and will not
touch, and why parsing is deliberately permissive about numbers, is explained in
[the plan lifecycle](../explanation/plan-lifecycle.md).

## Let the model discover the operations

Instead of hardcoding an op list in your prompt:

```python
from videopython.editing import Operation, OpCategory

for op_id, cls in Operation.llm_registry().items():
    doc = (cls.__doc__ or "").splitlines()[0].strip()
    print(f"{op_id}: {doc}")

transforms = {k: v for k, v in Operation.llm_registry().items()
              if v.category is OpCategory.TRANSFORM}
```

`llm_registry()` is the LLM-safe subset; `registry()` is everything, for the worker that
executes a stored plan. For per-op tool definitions:

```python
tools = [{
    "name": f"transform_{op_id}",
    "description": (cls.__doc__ or "").splitlines()[0],
    "input_schema": cls.llm_json_schema(),   # drops llm_hidden advanced fields
} for op_id, cls in Operation.llm_registry().items()
  if cls.category is OpCategory.TRANSFORM]
```

Use `cls.llm_json_schema()` rather than `cls.model_json_schema()` for anything the model
sees — it strips advanced fields such as raw font paths, whose LLM-facing counterpart is
the `font` name enum.

## Include AI operations

AI ops register only once `videopython.ai` is imported. Import it before generating the
schema if your plans may use them:

```python
import videopython.ai                 # registers face_crop, object_detection_overlay

from videopython.editing import VideoEdit
schema = VideoEdit.json_schema()      # now includes them
```

## Supply data the plan cannot carry

Operations that need bulky side-channel input declare it via
`requires: ClassVar[tuple[str, ...]]`; the runner pulls the matching keys out of
`context` and re-bases time-based values onto each segment's local timeline.

```python
edit.run_to_file("out.mp4", context={"transcription": transcription})

needs_transcript = [op_id for op_id, cls in Operation.registry().items()
                    if "transcription" in cls.requires]
```

## Notes

- **Lead with the schema.** It encodes the structural rules, so the model needs no
  few-shot examples.
- **Give the model source metadata** — duration, dimensions, fps — or it will invent time
  ranges that do not exist.
- **Always `validate()` before `run_to_file()`.** It is cheap and catches almost
  everything.
- **Re-prompt with the whole error list** from `check()`, not one error at a time.

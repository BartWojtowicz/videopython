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
discriminated union over every registered `Operation`. AI ops appear in
the union only after `import videopython.ai`.

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
from videopython.base import Operation, OpCategory

# All registered ops
for op_id, cls in Operation.registry().items():
    doc = (cls.__doc__ or "").splitlines()[0].strip()
    print(f"{op_id}: {doc}")

# By category
transforms = {k: v for k, v in Operation.registry().items()
              if v.category is OpCategory.TRANSFORM}

# Per-op JSON Schema (standard Pydantic)
Operation.get("color_adjust").model_json_schema()
```

For per-op tool definitions:

```python
tools = []
for op_id, cls in Operation.registry().items():
    if cls.category is not OpCategory.TRANSFORM:
        continue
    tools.append({
        "name": f"transform_{op_id}",
        "description": (cls.__doc__ or "").splitlines()[0],
        "input_schema": cls.model_json_schema(),
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

```python
edit = VideoEdit.from_dict(plan)
try:
    predicted = edit.validate()
    print(f"Output: {predicted.width}x{predicted.height}, "
          f"{predicted.total_seconds:.1f}s")
except ValueError as e:
    # Feed `e` back to the LLM to retry
    print(f"Invalid plan: {e}")
```

This makes it cheap to let an LLM retry: validate, return the error,
ask the LLM to fix it.

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
  `Operation.registry()` instead of hardcoding the op list.

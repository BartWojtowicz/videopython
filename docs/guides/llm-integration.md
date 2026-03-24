# LLM & AI Agent Integration

videopython is designed to be controlled by LLMs. Every video operation exposes a machine-readable spec with descriptions, parameter types, and value constraints - all available as JSON Schema at runtime. This means an LLM can discover available operations, generate valid edit plans, and validate them before any video frames are loaded.

## How It Works

The typical LLM-driven video editing workflow:

1. **Discover** - query the operation registry to find available operations and their schemas
2. **Generate** - pass `VideoEdit.json_schema()` to an LLM as a tool/function schema or structured-output format
3. **Validate** - call `edit.validate()` to dry-run the plan using metadata only (fast, no frame loading)
4. **Execute** - call `edit.run()` to produce the final video

```python
from videopython.editing import VideoEdit

# 1. Get the schema to pass to your LLM
schema = VideoEdit.json_schema()

# 2. The LLM generates a plan (e.g. via tool call or structured output)
plan = call_your_llm(schema=schema, prompt="Create a 15s highlight reel from input.mp4")

# 3. Parse and validate (no frames loaded yet)
edit = VideoEdit.from_dict(plan)
predicted_metadata = edit.validate()
print(predicted_metadata)  # check output dimensions, duration, fps

# 4. Execute
video = edit.run()
video.save("output.mp4")
```

## Passing the Schema to an LLM

`VideoEdit.json_schema()` returns a standard JSON Schema (Draft-07 compatible) that describes the full structure of a valid editing plan. You can pass it directly to any LLM API that supports structured output or tool/function calling.

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
        {"role": "system", "content": "You are a video editor. Generate editing plans."},
        {"role": "user", "content": "Cut input.mp4 to the first 10 seconds, resize to vertical 1080x1920, and add a fade-in."},
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

tool_call = response.choices[0].message.tool_calls[0]
plan = json.loads(tool_call.function.arguments)

edit = VideoEdit.from_dict(plan)
edit.validate()
video = edit.run()
video.save("output.mp4")
```

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
    messages=[
        {"role": "user", "content": "Cut input.mp4 to the first 10 seconds, resize to vertical 1080x1920, and add a fade-in."},
    ],
)

tool_block = next(b for b in response.content if b.type == "tool_use")
plan = tool_block.input

edit = VideoEdit.from_dict(plan)
edit.validate()
video = edit.run()
video.save("output.mp4")
```

## Operation Discovery

The operation registry lets your code (or an LLM via a tool) discover all available operations at runtime. This is useful for building agents that need to understand what video operations are possible.

### Listing all operations

```python
from videopython.base import get_operation_specs

for op_id, spec in get_operation_specs().items():
    print(f"{op_id}: {spec.description}")
```

Output (abbreviated):

```
cut: Extracts a time range from the video. ...
resize: Scales the frame to new dimensions. ...
crop: Crops the frame to a smaller region. ...
speed_change: Speeds up or slows down video playback. ...
color_adjust: Adjusts brightness, contrast, and saturation. ...
zoom_effect: Progressively zooms into or out of the frame center. ...
...
```

### Filtering by category or capability

```python
from videopython.base import get_specs_by_category, get_specs_by_tag, OperationCategory

# Only transforms
transforms = get_specs_by_category(OperationCategory.TRANSFORMATION)

# Only effects
effects = get_specs_by_category(OperationCategory.EFFECT)

# Operations that change video duration
duration_changers = get_specs_by_tag("changes_duration")

# Operations that need a transcription
transcript_ops = get_specs_by_tag("requires_transcript")
```

### Inspecting a single operation

```python
from videopython.base import get_operation_spec

spec = get_operation_spec("color_adjust")

print(spec.id)            # "color_adjust"
print(spec.description)   # "Adjusts brightness, contrast, and saturation. ..."
print(spec.tags)          # set()
print(spec.category)      # OperationCategory.EFFECT

# JSON Schema for constructor args
print(spec.to_json_schema())
# {
#     "type": "object",
#     "properties": {
#         "brightness": {"type": "number", "minimum": -1, "maximum": 1, "description": "..."},
#         "contrast": {"type": "number", "minimum": 0.5, "maximum": 2.0, "description": "..."},
#         "saturation": {"type": "number", "minimum": 0, "maximum": 2.0, "description": "..."}
#     },
#     "required": [],
#     "additionalProperties": false
# }

# JSON Schema for apply args (effects have start/stop)
print(spec.to_apply_json_schema())
```

### Building a dynamic tool list

You can expose each operation as a separate LLM tool for fine-grained control:

```python
from videopython.base import get_specs_by_category, OperationCategory

tools = []
for op_id, spec in get_specs_by_category(OperationCategory.TRANSFORMATION).items():
    tools.append({
        "name": f"transform_{op_id}",
        "description": spec.description,
        "parameters": spec.to_json_schema(),
    })

# Pass `tools` to your LLM API
```

## Rich Schema Constraints

Every operation parameter has constraints that help LLMs generate valid values on the first try:

| Constraint | Example |
|---|---|
| `minimum` / `maximum` | brightness: -1 to 1, contrast: 0.5 to 2.0 |
| `exclusive_minimum` | zoom_factor > 1, resize width > 0 |
| `enum` | speed_change ramp: `["linear", "ease_in", "ease_out"]` |
| `nullable` | resize width/height (at least one must be non-null) |

These constraints map directly to JSON Schema keywords, so LLMs that support constrained generation (structured output) will respect them automatically.

## Validation Before Execution

`VideoEdit.validate()` performs a fast dry run using `VideoMetadata` - no video frames are loaded. This catches common errors in LLM-generated plans:

- Invalid time ranges (start >= end, end > source duration)
- Missing required arguments
- Incompatible segment dimensions/FPS for concatenation
- Unknown operation IDs
- Invalid parameter values

```python
edit = VideoEdit.from_dict(plan)

try:
    predicted = edit.validate()
    print(f"Output: {predicted.width}x{predicted.height}, {predicted.duration_seconds:.1f}s")
except ValueError as e:
    print(f"Invalid plan: {e}")
    # Send the error back to the LLM for correction
```

This makes it cheap to let an LLM retry if its first plan is invalid - validate, return the error, and ask the LLM to fix it.

## Context Data for Special Operations

Some operations need data that shouldn't be part of the JSON plan itself. For example, `silence_removal` needs a transcription object. Pass these via `context`:

```python
from videopython.editing import VideoEdit

# Assume `transcription` was produced by AudioToText or provided externally
edit = VideoEdit.from_dict(plan)
video = edit.run(context={"transcription": transcription})
```

Operations tagged with `requires_transcript` in the registry automatically receive `context["transcription"]`. Use `get_specs_by_tag("requires_transcript")` to discover which operations need it.

## AI Operations

AI-powered operations (face tracking crop, split screen composite) are registered only when `videopython.ai` is imported. If your LLM-generated plans use AI operations, import AI first:

```python
import videopython.ai  # registers AI operation specs
from videopython.editing import VideoEdit

schema = VideoEdit.json_schema()  # now includes AI ops
```

## Multicam Editing Plans

For podcast-style multicam editing, use `MultiCamEdit.json_schema()` the same way:

```python
from videopython.editing import MultiCamEdit

schema = MultiCamEdit.json_schema()

# LLM generates a multicam plan
plan = call_your_llm(
    schema=schema,
    prompt="Edit this podcast with 3 cameras. Switch every 15-30 seconds with fade transitions.",
)

edit = MultiCamEdit.from_dict(plan)
predicted = edit.validate()  # fast metadata-only check
video = edit.run()
video.save("podcast.mp4")
```

The schema includes transition types (instant, fade, blur) with their parameters and constraints. The LLM decides which camera to use at each cut point and which transitions to apply.

## Tips for Building LLM Video Agents

- **Start with the schema.** Pass `VideoEdit.json_schema()` as the tool schema - it encodes all structural rules so the LLM doesn't need to learn them from examples.
- **Always validate.** Call `edit.validate()` before `edit.run()`. Validation is fast and catches most errors.
- **Use the error loop.** If validation fails, pass the error message back to the LLM and ask it to fix the plan. Most issues are corrected in one retry.
- **Provide source metadata.** Tell the LLM the source video's duration, dimensions, and FPS so it can generate sensible time ranges and resize targets.
- **Expose the registry.** For agentic workflows, let the LLM query operation specs to discover what's available rather than hardcoding a list.
- **Alias normalization.** The parser accepts common aliases (e.g. `blur` for `blur_effect`), but `json_schema()` and `to_dict()` use canonical IDs only.

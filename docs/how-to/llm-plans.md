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

## Pass the schema to your model

`VideoEdit.json_schema()` describes the plan and its LLM-exposed operations. Give
that schema, the source paths, and source metadata to your integration. The
`call_your_llm` function above is a placeholder for your own model call; videopython
does not install a hosted-provider SDK.

If your decoder accepts a closed structured-output schema, use:

```python
from videopython.editing import Operation, VideoEdit

plan_schema = VideoEdit.json_schema(strict=True)
operation_schema = Operation.json_schema(strict=True)
```

See the [schema reference](../reference/video-edit.md#json-schema) for the generated
structure. Decoder support varies. Parse and validate the returned plan even when
the model accepts the schema: numeric bounds and cross-field rules still need checks.

## Refine a plan the model got wrong

Build metadata keyed by the exact source paths, then repair and check the plan:

```python
from videopython.base import VideoMetadata

edit = VideoEdit.from_dict(plan)
source_metadata = {
    str(segment.source): VideoMetadata.from_path(segment.source)
    for segment in edit.segments
}
edit, repairs = edit.repair(source_metadata, clamp_segment_end=True)
edit, dim_repairs = edit.normalize_dimensions(source_metadata, "largest")
errors = edit.check(source_metadata)
if errors:
    ...  # re-prompt with the previous plan and the structured errors
else:
    edit.run_to_file("output.mp4")
```

Clamping segment ends can shorten the requested edit. Omit `clamp_segment_end=True`
if the caller must decide how to handle an overrun. Signatures, failure behavior,
and metadata requirements are in the
[validation reference](../reference/video-edit.md#validation-repair-normalization).

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

AI ops register when their classes are imported. Import the classes before generating the
schema if your plans may use them:

```python
from videopython.ai import FaceTrackingCrop, ObjectDetectionOverlay

from videopython.editing import VideoEdit
schema = VideoEdit.json_schema()      # now includes them
```

## Supply context data

Operations that need bulky side-channel input declare it via
`requires: ClassVar[tuple[str, ...]]`; the runner pulls the matching keys out of
`context` and re-bases time-based values onto each segment's local timeline.

```python
edit.run_to_file("out.mp4", context={"transcription": transcription})

needs_transcript = [op_id for op_id, cls in Operation.registry().items()
                    if "transcription" in cls.requires]
```

## Notes

- **Lead with the schema.** It describes the available operations and structural rules.
- **Give the model source metadata** — duration, dimensions, fps — or it will invent time
  ranges that do not exist.
- **Always `validate()` before `run_to_file()`.** Inspect structured errors before spending time on rendering.
- **Re-prompt with the whole error list** from `check()`, not one error at a time.

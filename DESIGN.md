# Operation Unification

## Problem

One operation lives in 5 places: the class, `base/registry.py` ParamSpec entries,
`VideoMetadata` fluent prediction methods, `Video` fluent transform methods, and
`editing/video_edit.py`'s hand-rolled parser + JSON-schema emitter. Adding an op
means touching all five; drift is constant.

## Target

`Operation` is a `pydantic.BaseModel` subclass — the single source of truth.
Fields ARE the JSON wire format. Schema, validation, serialisation are free.
Signature is committed to single-input `apply(video: Video) -> Video`.

```python
class Resize(Operation):
    op: Literal["resize"] = "resize"          # wire discriminator + registry key
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM
    streamable: ClassVar[bool] = True

    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)
    round_to_even: bool = True

    def apply(self, video: Video) -> Video: ...
    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata: ...
    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None: ...  # None = eager
```

`op` is a one-value `Literal` Pydantic field, not a `ClassVar`. It flows into
the JSON wire and acts as the discriminator natively; Pydantic's
`Discriminator("op")` machinery gives us the union schema for free. The base
exposes `Operation.op_id` as a read-only property mirroring `self.op`.

`Effect(Operation)` adds a `window: TimeRange | None` field, the
shape-and-frame-count-preserving invariant, and the streaming hooks
(`streaming_init`, `process_frame`).

Context-dependent ops (silence removal, subtitles) declare
`requires: ClassVar[tuple[str, ...]] = ("transcription",)`; the runner injects
matching keys into `apply()`. Replaces the `requires_transcript` tag.

Multi-source ops (`PictureInPicture`, `SplitScreenComposite`) take a
`source: Path` field and load just-in-time.

Transitions and `MultiCamEdit` are deleted — they have no live consumers and
their current 2→1 shape pre-dates the streaming pipeline. A clean redesign is
easier once `Operation` is in trunk; tracked as a follow-up, not in scope here.

## Registry

Auto-built via `__pydantic_init_subclass__`: the hook reads the subclass's
`op` field, extracts its `Literal[str]` value, and stores it in
`Operation._registry`. `Operation.registry()` returns a snapshot;
`Operation.get(op_id)` looks up a class. `Operation.json_schema()` returns
the discriminated-union schema (oneOf over all subclasses, tag = `op`) — this
is what we hand to the LLM. The same hook copies Google-style `Args:`
descriptions from the docstring onto `model_fields[name].description`, so
schemas keep their per-parameter documentation without duplication. AI ops
auto-register on `import videopython.ai`.

## VideoEdit

Becomes a thin Pydantic model. Wire format flattens: `op` inline, fields hoisted.

```json
{"segments": [{"source": "a.mp4", "start": 0, "end": 5,
  "operations": [{"op": "resize", "width": 1280},
                 {"op": "blur", "mode": "constant", "iterations": 10,
                  "window": {"start": 1, "stop": 3}}]}],
 "post_operations": [...]}
```

`VideoEdit.model_validate(plan)` parses + validates in one step (Pydantic does
it). `VideoEdit.model_json_schema()` is the LLM-facing schema (Pydantic does
it). Streaming machinery in `base/streaming.py` stays; `run_to_file` inspects
each op's `to_ffmpeg_filter` / `streamable` to plan eager vs streamed.

See `TODO.md` for the migration plan.

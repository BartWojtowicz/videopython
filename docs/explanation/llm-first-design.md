# LLM-first design

"LLM-friendly" is usually a wrapper: a library, plus a hand-written layer that translates
between JSON and the library's real API. That layer drifts from the API it wraps, and
every new feature has to be added twice.

videopython removes the layer. Every editing primitive is a Pydantic `BaseModel` whose
fields **are** the JSON wire format. There is no translation step to drift, because the
schema is generated from the same class the engine executes.

## The schema is the API

```python
class Resize(Operation):
    """Resize the video."""

    op: Literal["resize"] = "resize"          # discriminator + registry key
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    width: int | None = Field(None, gt=0)
    height: int | None = Field(None, gt=0)
```

Field types, defaults, and value constraints surface as standard JSON Schema. `op` is a
one-value `Literal` **field**, not a `ClassVar`, so it travels on the wire as the
discriminator and doubles as the registry key.

Subclasses auto-register through `__pydantic_init_subclass__`, so importing
`videopython.editing` populates the registry, and `VideoEdit.json_schema()` builds a
discriminated union over everything in it. Adding an operation adds it to the LLM's
vocabulary — no second place to update.

## Two visibility switches

Not everything the engine can execute is something a model should emit.

**`llm_exposed: ClassVar[bool]`** — set `False` for operations needing a server-resolved
path (`image_overlay`, `full_image_overlay`). A model asked for a watermark would have to
invent a file path it has no way to know. So `Operation.llm_registry()` and the default
`json_schema()` omit them, while `Operation.registry()` and `from_dict` still see
everything — a stored plan that legitimately contains one keeps executing.

**`llm_hidden` on a field** — `Field(json_schema_extra={"llm_hidden": True})` keeps a
field on the wire (it parses, it runs) but drops it from the LLM-facing schema. The raw
`font_filename` path on `text_overlay`/`add_subtitles` is the canonical case: its
LLM-facing counterpart is the `font` name enum, which is a fixed set the model can
actually choose from and a stored plan can round-trip on.

`cls.model_json_schema()` keeps hidden fields; `cls.llm_json_schema()` and the default
`Operation.json_schema()` strip them.

## Make the invalid unrepresentable, then repair the rest

Two mechanisms, at different points in the pipeline.

**At decode time**, `json_schema(strict=True)` emits a closed grammar — every object
`additionalProperties: false`, every property `required`, the union as an `anyOf` of
closed variants with no `discriminator`, `$defs` hoisted to the root. Optionality follows
the Pydantic type. A decoder that enforces these features can prevent unknown fields
and invalid enum values. Structural validators, numeric bounds, and cross-field
rules still require parsing and validation.

**After decode**, cross-field constraints take over — `timestamp < duration`, segment
dimension equality. No grammar can express those, so they live in
`check()` / `repair()` / `normalize_dimensions()`. See [the plan
lifecycle](plan-lifecycle.md).

The schema describes the model input. Validation checks whether that input makes
sense for the selected media. Mechanical repairs handle only unambiguous changes.

## Selection by id

The auto-editor exists because a vision model is good at one thing and bad at another. It
judges shots well. It authors timestamps badly.

So `AutoEditor` never asks for a timestamp. Analysis produces a catalog of candidate
scenes, each with a stable `scene_id`, exact bounds from scene detection, a caption, a
transcript, and a keyframe. The model authors a plan referencing scenes **by id** and adds
operations; `resolve_plan` maps ids back to the detected bounds.

The model's temporal imprecision therefore cannot reach the render. Precision comes from
the detector, judgment from the model, and each does only what it is good at.

The same idea drives the MCP server's payload discipline: the catalog *text* is always
complete so the agent can shortlist from captions, while keyframes — the payload that
grows with footage — are downscaled and capped, fetched on demand by id.

## Three ways to put a model in the loop

They differ in one thing: who owns the model.

| Mode | Planner | Guide |
|---|---|---|
| Bring your own LLM | Yours, in your harness | [Author edit plans with your own LLM](../how-to/llm-plans.md) |
| `AutoEditor` | An Ollama vision model | [Let a local LLM edit for you](../how-to/auto-editing.md) |
| MCP server | The connecting agent's own model | [Drive editing from an MCP agent](../how-to/mcp-server.md) |

All three sit on the same registry, the same schema, and the same validate/repair loop.
The auto-editor and the MCP server are thin — most of what makes them work is the
primitives described above.

## AI operations are registered lazily

AI operation classes register when their defining modules load. Import
`FaceTrackingCrop` and `ObjectDetectionOverlay` from `videopython.ai` before
schema generation or parsing a plan that uses them. Importing `videopython.ai`
alone leaves its lazy exports unloaded. See the
[LLM guide](../how-to/llm-plans.md#include-ai-operations) for the import example.

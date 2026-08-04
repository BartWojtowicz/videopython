# The plan lifecycle

A plan goes through up to five stages between "some JSON arrived" and "frames are being
written". Each stage owns a different class of problem. Knowing which stage owns what is
the difference between a refine loop that converges in one round and one that plays
whack-a-mole.

```
from_dict()  →  repair()  →  normalize_dimensions()  →  check()  →  run_to_file()
   shape        mechanics        concat geometry        everything      execution
```

## Parse owns shape, validation owns numbers

`VideoEdit.from_dict()` enforces the **shape**: field types, required fields, unknown
ops, extra fields, and op-local structural rules (`resize` needs at least one dimension).
Anything wrong there is a Pydantic `ValidationError`.

The **numeric bounds** of the plan skeleton — segment `start`/`end`, effect `window`
ranges — are deliberately *not* enforced at parse. A negative `window.start` parses fine.
A segment with `start >= end` parses fine. Validation reports them instead.

That looks backwards until you consider who writes plans. If bounds raised at parse, a
model that emitted `window.start: -0.5` would produce an exception with one message,
while a model that emitted a segment ending past the source would produce a structured
`PlanError` — two different error channels for the same class of mistake, and only one of
them repairable. Deferring numbers to validation gives every numeric violation the same
treatment: structured, collectable, and often auto-fixable.

## The four validation entry points

| Call | Reports | Raises | Use for |
|---|---|---|---|
| `validate()` / `validate_with_metadata(meta)` | First failure | `PlanValidationError` | Scripts, a final gate |
| `check(meta)` | **Every** error, as a list | never | Refine loops |
| `repair(meta)` | `(edit, changelog)` | only on a segment `end` past the source | Fixing mechanical faults |
| `normalize_dimensions(meta, target)` | `(edit, changelog)` | never | Making concat geometry hold |

All of them chain each operation's `predict_metadata` across the plan and check segment
bounds, effect windows, and concatenation compatibility. None of them decode a frame;
`validate_with_metadata` does not even open the file.

`PlanValidationError` subclasses `ValueError`, so `except ValueError` keeps working, and
carries structured `PlanError`s: `code` (a small enum), `location` (e.g.
`"segments[1].operations[0]"`), `field`, `value`, `limit`. **Branch on `code`, never on
the message text.**

## What `repair()` will and will not do

It clamps only what has one obvious correct answer, and records every change in a
`PlanRepair` changelog:

- effect `window.start`/`window.stop` into `[0, duration]` — negatives to `0`, overruns to
  the duration — in segments and in `post_operations`;
- time-valued op parameters past the clip end (`freeze_frame.timestamp` and friends),
  generically, via each op's declared time fields;
- a negative segment `start` to `0`;
- with `clamp_segment_end=True`, a segment `end` past the source to the source end. Off by
  default, because shortening a segment changes editorial intent.

It **never invents intent**. A concat dimension mismatch or an `end <= start` range is
left exactly as it was, for `check()` to report and for you to re-prompt about. It also
never raises on an operation it cannot repair — it leaves it for `check()`. So always
`check()` the returned plan before running it.

## Why `normalize_dimensions()` exists separately

`CONCAT_MISMATCH` is the one error class you cannot cleanly fix in your own layer.
Detecting it needs each segment's *predicted post-operation* dimensions; fixing it needs a
per-segment `resize` inserted *before* the concat. Both require the engine's own
prediction machinery.

So it is a first-class method. Given a target — an explicit `(width, height)`, `"first"`,
`"largest"`, or `"match"` (the lowest common resolution, the same policy
`match_to_lowest_resolution` applies in-stream) — it appends a `resize` to every segment
whose predicted output differs, and returns the usual changelog. The "all segments share
dimensions" invariant becomes satisfiable by construction.

Like `repair()` and `check()`, it is best-effort and non-raising: a segment it cannot
predict yet is left untouched and deferred.

## The one thing the runner tolerates

A duration-shrinking operation (`speed_change`, `freeze_frame`) ordered *before* a
windowed effect leaves that window's `stop` past the now-shorter clip. This is common,
harmless, and unambiguous, so `run_to_file()` clamps it rather than failing.

To keep the reports consistent with that, `validate(clamp_windows=True)` and
`check(..., clamp_windows=True)` do not report it either, and `repair()` clamps it in the
returned plan.

## The whole loop

```python
edit = VideoEdit.from_dict(plan)                       # shape enforced
edit, repairs = edit.repair(source_metadata)           # mechanics clamped
edit, dim_repairs = edit.normalize_dimensions(source_metadata, "largest")
errors = edit.check(source_metadata)                   # whatever is left, all at once
if errors:
    ...  # re-prompt with the previous plan + the full structured list
edit.run_to_file("out.mp4")
```

`source_metadata` leads every signature in the family, so the calls read the same way. The
practical version of this loop, with provider code around it, is in [Author edit plans
with your own LLM](../how-to/llm-plans.md).

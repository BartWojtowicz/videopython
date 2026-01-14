# Verified Video Editing Composition

Experiments for validating video editing pipelines before execution.

## Goal

Design a system that catches compatibility errors **before execution** rather than at runtime.

## Current State

- Validation happens lazily at execution time
- `can_be_merged_with()` exists but only checked during `Transition.apply()`
- No dry-run or spec prediction capabilities
- Operations have deterministic metadata transformations

---

## Approach Options

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **A: VideoSpec** | Each operation declares metadata transformation | Simple, matches existing patterns | Metadata-only validation |
| **B: ValidatedPipeline** | Build ops without executing, then validate/execute separately | Clean API, familiar pattern | New abstraction needed |
| **C: Graph-based DAG** | Build dependency graph, compile validates, execute | Complex multi-branch workflows | Overkill for simple cases |

**Decision:** Start with Option A (VideoSpec) as foundation, then build B on top if needed.

---

## Experiment 1: VideoSpec Prediction

### Hypothesis

We can accurately predict output metadata for all transform types given input metadata and transform parameters.

### Implementation

File: `experiments/spec_prediction.py`

### Results

**PASSED: 7/7 tests**

| Transform | Predicted | Actual | Match |
|-----------|-----------|--------|-------|
| CutSeconds | 1920x1080@23.98fps, 1.00s | 1920x1080@23.98fps, 1.00s | YES |
| CutFrames | 1920x1080@23.98fps, 1.25s | 1920x1080@23.98fps, 1.25s | YES |
| Resize (both) | 640x480@23.98fps, 2.00s | 640x480@23.98fps, 2.00s | YES |
| Resize (width) | 640x360@23.98fps, 2.00s | 640x360@23.98fps, 2.00s | YES |
| Resize (height) | 853x480@23.98fps, 2.00s | 853x480@23.98fps, 2.00s | YES |
| ResampleFPS | 1920x1080@15.00fps, 2.00s | 1920x1080@15.00fps, 2.00s | YES |
| Crop | 320x240@23.98fps, 2.00s | 320x240@23.98fps, 2.00s | YES |

**Conclusion:** VideoSpec prediction works accurately for all transform types.

---

## Experiment 2: Transform Output Prediction

### Hypothesis

Adding `predict_output(spec) -> spec` to transforms is straightforward and matches existing patterns.

### Results

Based on Experiment 1 findings, prediction functions are simple:

```python
# CutSeconds: duration changes
predict_cut_seconds(spec, start, end) -> VideoSpec(h, w, fps, end-start)

# Resize: dimensions change (with aspect ratio logic)
predict_resize(spec, width, height) -> VideoSpec(new_h, new_w, fps, duration)

# ResampleFPS: fps changes, duration stays same
predict_resample_fps(spec, target_fps) -> VideoSpec(h, w, target_fps, duration)

# Crop: dimensions change
predict_crop(spec, width, height) -> VideoSpec(height, width, fps, duration)
```

**Key insight:** All transforms have deterministic output based solely on input spec + params.

---

## Learnings

### Confirmed

1. **Metadata prediction is accurate** - All 7 transform types can be predicted without execution
2. **Simple dataclass suffices** - VideoSpec only needs height, width, fps, duration_seconds
3. **Equality needs tolerance** - Duration can vary by ~0.1s due to frame/time rounding
4. **FPS comparison needs rounding** - `round(fps)` for compatibility checks (handles 23.976 vs 24)

### Design Decisions

1. **VideoSpec vs VideoMetadata** - VideoSpec is simpler (4 fields vs 5), focused on validation
2. **Prediction as standalone functions** - Started with functions, can move to transform methods later
3. **Duration-based spec** - Used duration_seconds instead of frame_count (simpler, sufficient)

### Transition Prediction (Added)

Transitions also work with VideoSpec:

```python
def predict_transition(spec1, spec2, effect_time_seconds=0.0):
    if not spec1.can_merge_with(spec2):
        return "Incompatible specs error"
    return VideoSpec(spec1.height, spec1.width, spec1.fps,
                     spec1.duration + spec2.duration - effect_time_seconds)
```

| Test | Result |
|------|--------|
| Transition (compatible) | PASS |
| Transition (incompatible detection) | PASS |

**Total: 9/9 tests passing**

### Next Steps

1. ~~Add prediction for Transitions~~ DONE
2. Add prediction for Effects (should be identity - no metadata change)
3. ~~Build ValidatedPipeline using VideoSpec predictions~~ DONE

---

## Experiment 3: ValidatedPipeline

### Implementation

File: `experiments/validated_pipeline.py`

```python
pipeline = (
    ValidatedPipeline()
    .add_source("v1", "clip1.mp4")
    .add_source("v2", "clip2.mp4")
    .add_transform("v1", "v1_resized", Resize(640, 480))
    .add_transform("v2", "v2_resized", Resize(1280, 720))  # Different size!
    .add_transition("v1_resized", "v2_resized", "merged", FadeTransition(0.5))
)

result = pipeline.validate()
# Invalid: 1 error(s)
#   - [step2:transition] Incompatible specs for merge: 640x480@24fps vs 1280x720@24fps
```

### Results

**PASSED: 5/5 tests**

| Test | Description | Result |
|------|-------------|--------|
| Simple transforms | CutSeconds -> Resize chain | PASS |
| Compatible transition | Same video merged | PASS |
| Incompatible transition | Different sizes detected | PASS |
| Missing input | Bad reference caught | PASS |
| Execute pipeline | Prediction matches actual | PASS |

### Key Features

1. **Lazy source loading** - Only reads metadata via ffprobe, doesn't load frames
2. **Error accumulation** - Collects all errors before reporting
3. **Named nodes** - Clear step identification in errors
4. **Fluent API** - Method chaining for pipeline construction

---

## Open Questions

1. Should `predict_output()` be required or optional on transforms?
2. How to handle runtime-dependent values (e.g., cut to half duration)?
3. Should we auto-suggest fixes (like StackVideos does)?
4. Is the graph-based approach needed for complex workflows?

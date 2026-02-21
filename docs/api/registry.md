# Operation Registry

The operation registry provides machine-readable metadata for all video operations in videopython. It allows downstream tools to discover available operations, their parameters, and capabilities without importing internal modules or parsing docstrings.

## Quick Start

```python
from videopython.base import get_operation_specs, get_operation_spec, OperationCategory

# List all registered operations
specs = get_operation_specs()
for op_id, spec in specs.items():
    print(f"{op_id}: {spec.description}")

# Look up a specific operation (by ID or alias)
spec = get_operation_spec("cut")
print(spec.to_json_schema())

# AI operations are registered when videopython.ai is imported
import videopython.ai
specs = get_operation_specs()
assert "face_crop" in specs
```

## Operation Categories

Each operation belongs to one of four categories, defined by `OperationCategory`:

| Category | Apply signature | Description |
|---|---|---|
| `TRANSFORMATION` | `apply(video) -> Video` | Modifies video structure (cut, resize, crop) |
| `EFFECT` | `apply(video, start?, stop?) -> Video` | Applies visual effects over a time range |
| `TRANSITION` | `apply((video, video)) -> Video` | Combines two videos with a transition |
| `SPECIAL` | Non-standard | Operations with unique signatures |

## Capability Tags

Operations are tagged with capability metadata used for filtering:

- `changes_duration` - Operation may change video duration
- `changes_dimensions` - Operation may change video dimensions
- `changes_fps` - Operation may change frame rate
- `multi_source` - Operation accepts additional video inputs
- `multi_source_only` - Operation requires multiple video inputs
- `requires_transcript` - Operation requires transcription data
- `requires_faces` - Operation requires face detection (AI)

```python
from videopython.base import get_specs_by_tag, get_specs_by_category, OperationCategory

# Get all operations that change dimensions
dimension_ops = get_specs_by_tag("changes_dimensions")

# Get all effects
effects = get_specs_by_category(OperationCategory.EFFECT)
```

## Registering Custom Operations

Use `spec_from_class` to register your own operations:

```python
from videopython.base import Video, register, spec_from_class, OperationCategory

class MyCustomEffect:
    """Apply a custom visual effect."""
    def __init__(self, intensity: float = 0.5):
        self.intensity = intensity

    def apply(self, video: Video, start: float | None = None, stop: float | None = None) -> Video:
        ...

spec = spec_from_class(
    MyCustomEffect,
    op_id="my_custom_effect",
    category=OperationCategory.EFFECT,
    tags={"custom"},
)
register(spec)
```

`spec_from_class` introspects both the constructor and `apply` method signatures to build parameter schemas automatically. The `video`, `videos`, and `transcription` parameters are excluded from the apply schema by default. Use `exclude_params` / `exclude_apply_params` for non-JSON-serializable arguments and `param_overrides` / `apply_param_overrides` to add constraints like `minimum`/`maximum`.

## JSON Schema Generation

Every `OperationSpec` can generate JSON Schemas for both constructor arguments and apply method arguments:

```python
spec = get_operation_spec("cut")

# Constructor args schema
schema = spec.to_json_schema()
# {
#     "type": "object",
#     "properties": {
#         "start": {"type": "number", "description": "..."},
#         "end": {"type": "number", "description": "..."}
#     },
#     "required": ["start", "end"],
#     "additionalProperties": false
# }

# Apply args schema (e.g. for effects with start/stop)
spec = get_operation_spec("blur_effect")
apply_schema = spec.to_apply_json_schema()
# {
#     "type": "object",
#     "properties": {
#         "start": {"type": "number", ...},
#         "stop": {"type": "number", ...}
#     },
#     "required": [],
#     "additionalProperties": false
# }
```

Transformations and transitions typically have an empty apply schema (no extra arguments beyond the video input), while effects expose `start`/`stop` parameters.

## Registered Operations

### Base Operations

| ID | Class | Category | Tags |
|---|---|---|---|
| `cut_frames` | `CutFrames` | transformation | `changes_duration` |
| `cut` | `CutSeconds` | transformation | `changes_duration` |
| `resize` | `Resize` | transformation | `changes_dimensions` |
| `resample_fps` | `ResampleFPS` | transformation | `changes_fps` |
| `crop` | `Crop` | transformation | `changes_dimensions` |
| `speed_change` | `SpeedChange` | transformation | `changes_duration` |
| `picture_in_picture` | `PictureInPicture` | transformation | `multi_source` |
| `blur_effect` | `Blur` | effect | -- |
| `zoom_effect` | `Zoom` | effect | -- |
| `color_adjust` | `ColorGrading` | effect | -- |
| `vignette` | `Vignette` | effect | -- |
| `ken_burns` | `KenBurns` | effect | -- |
| `full_image_overlay` | `FullImageOverlay` | effect | -- |
| `instant_transition` | `InstantTransition` | transition | `multi_source_only` |
| `fade_transition` | `FadeTransition` | transition | `changes_duration`, `multi_source_only` |
| `blur_transition` | `BlurTransition` | transition | `changes_duration`, `multi_source_only` |
| `stack_videos` | `StackVideos` | special | `multi_source_only`, `changes_dimensions` |
| `add_subtitles` | `TranscriptionOverlay` | special | `requires_transcript` |

### AI Operations (require `import videopython.ai`)

| ID | Class | Category | Tags |
|---|---|---|---|
| `face_crop` | `FaceTrackingCrop` | transformation | `requires_faces`, `changes_dimensions` |
| `auto_framing` | `AutoFramingCrop` | transformation | `requires_faces`, `changes_dimensions` |
| `split_screen` | `SplitScreenComposite` | transformation | `requires_faces`, `multi_source`, `changes_dimensions` |

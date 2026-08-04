# AI operations

Editing operations that run a model. They live in `videopython.ai` rather than
`videopython.editing` so the core editing layer keeps no AI dependency
([why](../../explanation/architecture.md#why-ai-effects-live-in-ai-not-editing)), but they
are ordinary registry entries: put them in a segment's `operations` list like any other.

They register only after `import videopython.ai`.

## FaceTrackingCrop

The `face_crop` transform. Reframes around the tracked face, with framing rules (headroom
/ thirds / center) and a bounded camera speed. It constructs a
[`FaceSmoothingTracker`](understanding.md#face-tracking) internally.

```python
from videopython.ai import FaceTrackingCrop
from videopython.editing import VideoEdit, SegmentConfig

# Horizontal source to vertical, following the subject
edit = VideoEdit(segments=[SegmentConfig(source="input.mp4", start=0, end=5, operations=[
    FaceTrackingCrop(target_aspect=(9, 16)),
])])
edit.run_to_file("vertical.mp4")

# Headroom framing with a bounded camera speed
FaceTrackingCrop(framing_rule="headroom", max_speed=0.1)
```

::: videopython.ai.FaceTrackingCrop

## ObjectDetectionOverlay

The `object_detection_overlay` effect. Detects objects with a D-FINE COCO model and
composites colour-coded boxes with class labels. The detector
([`ObjectDetector`](understanding.md#objectdetector)) is constructed internally; the
drawing is done by the AI-free [renderer](#renderer).

```python
from videopython.ai import ObjectDetectionOverlay
from videopython.editing import VideoEdit, SegmentConfig

# Defaults: per-class colours, confidence shown, detection every 2nd frame
edit = VideoEdit(segments=[SegmentConfig(source="street.mp4", start=0, end=5, operations=[
    ObjectDetectionOverlay(),
])])
edit.run_to_file("annotated.mp4")

ObjectDetectionOverlay(class_filter=["person", "car"], detection_interval=1, model_size="s")
```

In a JSON plan (it is LLM-exposed):

```json
{"op": "object_detection_overlay", "class_filter": ["person", "car", "dog"],
 "confidence_threshold": 0.4, "detection_interval": 2,
 "window": {"start": 0, "stop": 5}}
```

### Cost

Memory stays bounded on long clips — it streams — but compute does not: a D-FINE forward
pass runs per sampled frame. To cap it:

| Knob | Effect |
|---|---|
| `window` | Restricts the overlay, and therefore detection, to a time range |
| `detection_interval` | Detect every Nth frame, hold boxes in between (default `2`). Higher is faster; fast motion shows more lag |
| `class_filter` | Fewer classes to draw |
| `model_size` | `"n"` (nano, default, fastest) → `"s"` → `"m"` (most accurate) |

::: videopython.ai.ObjectDetectionOverlay

## Renderer

Pure and AI-free, reusable with any list of
[`DetectedObject`](understanding.md#videopython.base.DetectedObject). Colours are deterministic per class,
so a class keeps its colour across frames and across runs.

```python
from videopython.base import DetectionStyle, class_color, draw_detections

frame = draw_detections(frame, detections, DetectionStyle(show_confidence=False))
```

::: videopython.base.draw_detections.draw_detections

::: videopython.base.draw_detections.DetectionStyle

::: videopython.base.draw_detections.class_color

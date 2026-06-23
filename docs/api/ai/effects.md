# AI Effects

AI-powered, shape-preserving [effects](../effects.md). Unlike plain effects,
these run a model per frame, so they physically live in `videopython.ai` and the
core editing layer keeps no AI dependency.

## ObjectDetectionOverlay

Detects objects in every frame with a D-FINE COCO model and composites tidy,
colour-coded bounding boxes with class labels (and optional confidence). The
detector ([`ObjectDetector`](understanding.md#objectdetector)) is constructed
internally; the box/label drawing is done by the AI-free renderer
[`videopython.base.draw_detections`](#renderer).

```python
from videopython.ai import ObjectDetectionOverlay
from videopython.editing import VideoEdit, SegmentConfig

# Default: per-class colours, confidence shown, detect every 2nd frame.
edit = VideoEdit(segments=[SegmentConfig(source="street.mp4", start=0, end=5, operations=[
    ObjectDetectionOverlay(),
])])
edit.run_to_file("annotated.mp4")

# Only people and cars, detect every frame, larger model for accuracy.
edit = VideoEdit(segments=[SegmentConfig(source="street.mp4", start=0, end=5, operations=[
    ObjectDetectionOverlay(
        class_filter=["person", "car"],
        detection_interval=1,
        model_size="s",
    ),
])])
edit.run_to_file("annotated.mp4")
```

In a JSON editing plan (it is exposed in the LLM-facing schema):

```json
{
  "op": "object_detection_overlay",
  "class_filter": ["person", "car", "dog"],
  "confidence_threshold": 0.4,
  "detection_interval": 2,
  "window": {"start": 0, "stop": 5}
}
```

### Performance

`object_detection_overlay` is **streamable** — memory stays bounded on long
clips — but detection is **compute**-bound: a D-FINE forward pass runs per
sampled frame. To cap cost:

- **`window`** — restrict the overlay (and therefore detection) to a time range.
- **`detection_interval`** — run detection every Nth frame and hold the boxes in
  between (default `2`). Higher is faster; fast-moving objects show more lag.
- **`class_filter`** — fewer classes to draw.
- **`model_size`** — `"n"` (nano, default, fastest) → `"s"` → `"m"` (most accurate).

::: videopython.ai.ObjectDetectionOverlay

## Renderer

The drawing is a pure, AI-free function reusable with any list of
[`DetectedObject`](understanding.md#detectedobject). Colours are deterministic
per class, so a class is the same colour in every frame and across runs.

```python
from videopython.base import DetectionStyle, class_color, draw_detections

frame = draw_detections(frame, detections, DetectionStyle(show_confidence=False))
```

::: videopython.base.draw_detections.draw_detections

::: videopython.base.draw_detections.DetectionStyle

::: videopython.base.draw_detections.class_color

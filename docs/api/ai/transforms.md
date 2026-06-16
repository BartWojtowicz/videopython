# AI Transforms

AI-powered video transforms. Framing-oriented behavior (headroom / thirds /
speed clamp) is implemented on `FaceTrackingCrop`.

The underlying `FaceTracker` lives in
[`videopython.ai.understanding.faces`](understanding.md#facetracker);
`FaceTrackingCrop` constructs one internally.

## Usage

```python
from videopython.ai import FaceTrackingCrop
from videopython.editing import VideoEdit, SegmentConfig

# Create vertical content from horizontal by tracking faces
edit = VideoEdit(segments=[SegmentConfig(source="input.mp4", start=0, end=5, operations=[
    FaceTrackingCrop(target_aspect=(9, 16)),
])])
edit.run_to_file("vertical.mp4")

# Headroom framing + bounded camera speed
edit = VideoEdit(segments=[SegmentConfig(source="input.mp4", start=0, end=5, operations=[
    FaceTrackingCrop(framing_rule="headroom", max_speed=0.1),
])])
edit.run_to_file("framed.mp4")
```

## FaceTrackingCrop

::: videopython.ai.FaceTrackingCrop

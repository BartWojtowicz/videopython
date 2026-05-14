# AI Transforms

AI-powered video transforms. Framing-oriented behavior (headroom / thirds /
speed clamp) is implemented on `FaceTrackingCrop`.

The underlying `FaceTracker` lives in
[`videopython.ai.understanding.faces`](understanding.md#facetracker);
`FaceTrackingCrop` constructs one internally.

## Usage

```python
from videopython.ai import FaceTrackingCrop
from videopython.base import Video

video = Video.from_path("input.mp4")

# Create vertical content from horizontal by tracking faces
vertical = FaceTrackingCrop(target_aspect=(9, 16)).apply(video)

# Headroom framing + bounded camera speed
framed = FaceTrackingCrop(framing_rule="headroom", max_speed=0.1).apply(video)
```

## FaceTrackingCrop

::: videopython.ai.FaceTrackingCrop

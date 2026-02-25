# AI Transforms

AI-powered video transforms that use face detection for intelligent cropping and tracking.
Framing-oriented behavior (headroom / thirds / speed clamp) is implemented on
`FaceTrackingCrop`.

## Usage

```python
from videopython.ai import FaceTrackingCrop, SplitScreenComposite
from videopython.base import Video

video = Video.from_path("input.mp4")
video2 = Video.from_path("input_2.mp4")

# Create vertical content from horizontal by tracking faces
crop = FaceTrackingCrop(target_aspect=(9, 16))
vertical_video = crop.apply(video)

# Face-tracking crop with headroom framing and limited camera speed
framing = FaceTrackingCrop(framing_rule="headroom", max_speed=0.1)
framed_video = framing.apply(video)

# Create split-screen with face tracking
composite = SplitScreenComposite(layout="2x1")
split_video = composite.apply(video, video2)
```

## FaceTracker

::: videopython.ai.FaceTracker

## FaceTrackingCrop

::: videopython.ai.FaceTrackingCrop

## SplitScreenComposite

::: videopython.ai.SplitScreenComposite

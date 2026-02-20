# AI Transforms

AI-powered video transforms that use face detection for intelligent cropping and tracking.

## Usage

```python
from videopython.ai import FaceTrackingCrop, SplitScreenComposite, AutoFramingCrop
from videopython.base import Video

video = Video.from_path("input.mp4")
video2 = Video.from_path("input_2.mp4")

# Create vertical content from horizontal by tracking faces
crop = FaceTrackingCrop(target_aspect=(9, 16))
vertical_video = crop.apply(video)

# Create split-screen with face tracking
composite = SplitScreenComposite(layout="2x1")
split_video = composite.apply(video, video2)

# Apply cinematographic auto-framing
framing = AutoFramingCrop(framing_rule="headroom")
framed_video = framing.apply(video)
```

## FaceTracker

::: videopython.ai.FaceTracker

## FaceTrackingCrop

::: videopython.ai.FaceTrackingCrop

## SplitScreenComposite

::: videopython.ai.SplitScreenComposite

## AutoFramingCrop

::: videopython.ai.AutoFramingCrop

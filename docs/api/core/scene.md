# Scene Detection

Lightweight scene detection using histogram comparison (no AI/ML dependencies).

## SceneDetector

Detects scene changes in videos by comparing color histograms of consecutive frames. When the histogram difference exceeds a threshold, a scene boundary is detected.

```python
from videopython.base import Video, SceneDetector

video = Video.from_path("video.mp4")

# Create detector with custom settings
detector = SceneDetector(
    threshold=0.3,        # 0.0-1.0, lower = more sensitive
    min_scene_length=0.5  # minimum scene duration in seconds
)

# Detect scenes
scenes = detector.detect(video)

for scene in scenes:
    print(f"Scene: {scene.start:.2f}s - {scene.end:.2f}s ({scene.duration:.2f}s)")
    print(f"  Frames: {scene.start_frame} - {scene.end_frame}")
```

### How It Works

1. Converts each frame to HSV color space
2. Calculates normalized histograms for Hue, Saturation, and Value channels
3. Compares consecutive frames using histogram correlation
4. Marks boundaries where difference exceeds threshold
5. Merges scenes shorter than `min_scene_length`

### Parameters

- `threshold` (float, default=0.3): Sensitivity for scene change detection. Range 0.0 to 1.0. Lower values detect more scene changes.
- `min_scene_length` (float, default=0.5): Minimum scene duration in seconds. Scenes shorter than this are merged with adjacent scenes.

::: videopython.base.scene.SceneDetector

## SceneDescription

Returned by `SceneDetector.detect()`. Contains timing and frame information for each detected scene.

::: videopython.base.description.SceneDescription

# Scene Detection

Lightweight scene detection using histogram comparison (no AI/ML dependencies).

## SceneDetector

Detects scene changes in videos by comparing color histograms of consecutive frames. When the histogram difference exceeds a threshold, a scene boundary is detected.

### Basic Usage (In-Memory)

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

### Parallel Processing (Recommended for Long Videos)

For long videos, use `detect_parallel()` which processes the video using multiple CPU cores. This provides ~3.5x speedup on 8-core machines.

```python
from videopython.base import SceneDetector

detector = SceneDetector(threshold=0.3, min_scene_length=0.5)

# Process video file directly with parallel workers
scenes = detector.detect_parallel("long_video.mp4", num_workers=8)

# Or let it auto-detect CPU count
scenes = detector.detect_parallel("long_video.mp4")
```

### Memory-Efficient Streaming

For memory-constrained environments, use `detect_streaming()` which processes frames one at a time with O(1) memory usage.

```python
from videopython.base import SceneDetector

detector = SceneDetector(threshold=0.3, min_scene_length=0.5)

# Stream frames from file - only 2 frames in memory at any time
scenes = detector.detect_streaming("very_long_video.mp4")
```

### Method Comparison

| Method | Memory | Speed | Best For |
|--------|--------|-------|----------|
| `detect(video)` | O(all frames) | Fast | Short videos already in memory |
| `detect_parallel(path)` | O(workers) | **Fastest** | Long videos, multi-core systems |
| `detect_streaming(path)` | O(1) | Slower | Memory-constrained environments |

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

::: videopython.base.SceneDescription

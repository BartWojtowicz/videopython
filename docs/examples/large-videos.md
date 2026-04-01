# Processing Large Videos

Process videos that are too large to fit in memory using streaming APIs.

## Goal

Analyze or process long videos (hours of footage) without running out of RAM.

## The Problem

Loading a full video into memory can be expensive:

```python
# This loads ALL frames into RAM - problematic for long videos
video = Video.from_path("2_hour_movie.mp4")  # Could use 50GB+ RAM
```

A 2-hour video at 1080p30 has ~216,000 frames. At ~6MB per frame (uncompressed RGB), that's over 1TB of data.

## Solution: Streaming Editing Pipeline

For editing workflows, `VideoEdit.run_to_file()` streams frames one at a time from
ffmpeg decode through per-frame effect processing to ffmpeg encode. Memory usage is
constant (~250 MB) regardless of video length.

```python
from videopython.editing import VideoEdit

plan = {
    "segments": [
        {
            "source": "2_hour_movie.mp4",
            "start": 0,
            "end": 7200,
            "transforms": [
                {"op": "resize", "args": {"width": 1920, "height": 1080}},
            ],
            "effects": [
                {"op": "color_adjust", "args": {"saturation": 0, "contrast": 1.15}},
                {"op": "fade", "args": {"mode": "in_out", "duration": 1.0}},
                {"op": "volume_adjust", "args": {"volume": 1.5}},
            ],
        }
    ],
}

edit = VideoEdit.from_dict(plan)
edit.run_to_file("output.mp4", crf=20, preset="medium")
# Peak memory: ~250 MB regardless of video length
```

When all operations are streamable, frames are never loaded into memory. If any operation
is not streamable (e.g. `reverse`, `speed_change`), the pipeline falls back to eager
mode automatically.

Check `VideoEdit.json_schema()` for `x-streamable: true` on each operation to see which
ones support streaming.

## Solution: FrameIterator

`FrameIterator` streams frames one at a time with O(1) memory usage:

```python
from videopython.base import FrameIterator

# Process frames without loading entire video
with FrameIterator("long_video.mp4") as frames:
    for frame_idx, frame in frames:
        # frame is a numpy array (H, W, 3) in RGB
        # Only one frame in memory at a time
        process_frame(frame)
```

## Full Example: Extract Thumbnails

Extract one thumbnail per minute from a long video:

```python
from videopython.base import FrameIterator, VideoMetadata
from PIL import Image
import os

def extract_thumbnails(video_path: str, output_dir: str, interval_seconds: float = 60.0):
    """Extract thumbnails at regular intervals from a video."""
    os.makedirs(output_dir, exist_ok=True)

    # Get metadata without loading video
    metadata = VideoMetadata.from_path(video_path)
    fps = metadata.fps
    interval_frames = int(interval_seconds * fps)

    with FrameIterator(video_path) as frames:
        for frame_idx, frame in frames:
            if frame_idx % interval_frames == 0:
                # Save thumbnail
                timestamp = frame_idx / fps
                img = Image.fromarray(frame)
                img.thumbnail((320, 180))  # Resize to thumbnail
                img.save(f"{output_dir}/thumb_{timestamp:.0f}s.jpg")
                print(f"Saved thumbnail at {timestamp:.0f}s")

# Extract one thumbnail per minute
extract_thumbnails("2_hour_movie.mp4", "thumbnails/", interval_seconds=60)
```

## Scene Detection on Large Videos

Use streaming scene detection for memory-efficient processing:

```python
from videopython.base import SceneDetector

detector = SceneDetector(threshold=0.3, min_scene_length=1.0)

# Streaming: O(1) memory, processes frames one at a time
scenes = detector.detect_streaming("long_video.mp4")

# Or parallel: Faster on multi-core systems
scenes = detector.detect_parallel("long_video.mp4", num_workers=8)

for scene in scenes:
    print(f"Scene: {scene.start:.1f}s - {scene.end:.1f}s")
```

## AI Video Analysis (Scene-First)

`VideoAnalyzer.analyze_path()` returns scene-centered outputs:

```python
from videopython.ai import VideoAnalyzer, VideoAnalysisConfig

config = VideoAnalysisConfig(
    enabled_analyzers={"audio_to_text", "semantic_scene_detector", "scene_vlm"},
)

analysis = VideoAnalyzer(config=config).analyze_path("long_video.mp4")
for scene in (analysis.scenes.samples if analysis.scenes else []):
    print(scene.scene_index, scene.start_second, scene.end_second)
    for chunk in scene.visual_segments:
        print("  ", chunk.start_second, chunk.end_second, chunk.caption)
```

## Processing a Segment

Process only a portion of a large video:

```python
from videopython.base import FrameIterator

# Only iterate frames from 1:00:00 to 1:10:00
with FrameIterator("movie.mp4", start_second=3600, end_second=4200) as frames:
    for frame_idx, frame in frames:
        process_frame(frame)
```

## Method Comparison

| Approach | Memory | Speed | Use Case |
|----------|--------|-------|----------|
| `VideoEdit.run_to_file()` | O(1) | Fast | Editing long videos with effects/transforms |
| `Video.from_path()` | O(all frames) | Fast access | Short videos, need random access |
| `FrameIterator` | O(1) | Sequential | Long videos, single pass analysis |
| `SceneDetector.detect_streaming()` | O(1) | Slower | Memory-constrained |
| `SceneDetector.detect_parallel()` | O(workers) | Fastest | Multi-core systems |

## Tips

- **Check metadata first**: Use `VideoMetadata.from_path()` to check video size before deciding how to process it.
- **Sample wisely**: For AI analysis, 0.1-0.5 FPS is usually sufficient. Higher rates waste compute.
- **Use parallel detection**: `detect_parallel()` is 3-4x faster than streaming on multi-core machines.
- **Process segments**: If you only need part of a video, use `start_second`/`end_second` parameters.

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

## Video Analysis on Large Files

Use `VideoAnalyzer.analyze_path()` instead of loading the video first:

```python
from videopython.ai import VideoAnalyzer

analyzer = VideoAnalyzer()

# Memory-efficient: streams frames, only loads sampled frames for AI analysis
result = analyzer.analyze_path(
    "long_documentary.mp4",
    frames_per_second=0.1,  # Sample 1 frame every 10 seconds
    transcribe=True,
)

print(f"Detected {len(result.scene_descriptions)} scenes")
for scene in result.scene_descriptions:
    print(f"{scene.start:.0f}s - {scene.end:.0f}s: {scene.frame_descriptions[0].description if scene.frame_descriptions else 'No description'}")
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
| `Video.from_path()` | O(all frames) | Fast access | Short videos, need random access |
| `FrameIterator` | O(1) | Sequential | Long videos, single pass |
| `SceneDetector.detect_streaming()` | O(1) | Slower | Memory-constrained |
| `SceneDetector.detect_parallel()` | O(workers) | Fastest | Multi-core systems |
| `VideoAnalyzer.analyze_path()` | O(sampled frames) | Depends on AI | Long videos with AI analysis |

## Tips

- **Check metadata first**: Use `VideoMetadata.from_path()` to check video size before deciding how to process it.
- **Sample wisely**: For AI analysis, 0.1-0.5 FPS is usually sufficient. Higher rates waste compute.
- **Use parallel detection**: `detect_parallel()` is 3-4x faster than streaming on multi-core machines.
- **Process segments**: If you only need part of a video, use `start_second`/`end_second` parameters.

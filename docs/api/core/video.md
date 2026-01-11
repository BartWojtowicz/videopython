# Video

The `Video` class is the core data structure in videopython.

## Video

::: videopython.base.Video
    options:
      members:
        - __init__
        - from_path
        - from_frames
        - from_image
        - save
        - copy
        - split
        - add_audio
        - add_audio_from_file
        - is_loaded
        - video_shape
        - frame_shape
        - total_seconds
        - metadata

## VideoMetadata

Get video metadata without loading frames into memory:

```python
from videopython.base import VideoMetadata

metadata = VideoMetadata.from_path("video.mp4")
print(f"Duration: {metadata.total_seconds}s")
print(f"Resolution: {metadata.width}x{metadata.height}")
print(f"FPS: {metadata.fps}")
print(f"Total frames: {metadata.frame_count}")
```

::: videopython.base.VideoMetadata

## FrameIterator

Memory-efficient frame iterator for streaming video frames without loading the entire video into memory. Useful for processing very long videos.

```python
from videopython.base import FrameIterator

# Stream frames one at a time - O(1) memory usage
with FrameIterator("long_video.mp4") as frames:
    for frame_idx, frame in frames:
        # frame is a numpy array (H, W, 3) in RGB format
        process_frame(frame)

# With time bounds
with FrameIterator("video.mp4", start_second=10.0, end_second=60.0) as frames:
    for frame_idx, frame in frames:
        process_frame(frame)
```

::: videopython.base.FrameIterator

# AI Object Swapping

Replace, remove, or modify objects in videos using AI-powered segmentation and inpainting.

## Local Pipeline

Object swapping uses local SAM2 + GroundingDINO segmentation with SDXL inpainting/compositing.

## ObjectSwapper

Main class for object manipulation in videos.

### Swap Object with Generated Content

Replace an object with AI-generated content from a text prompt:

```python
from videopython.base import Video
from videopython.ai import ObjectSwapper

video = Video.from_path("street.mp4")
swapper = ObjectSwapper()

# Replace red car with a blue motorcycle
result = swapper.swap(
    video=video,
    source_object="red car",
    target_object="blue motorcycle",
)

# Create video from swapped frames
swapped_video = Video.from_frames(result.swapped_frames, video.fps)
swapped_video.save("swapped.mp4")
```

### Swap Object with Image

Replace an object with a provided image:

```python
result = swapper.swap_with_image(
    video=video,
    source_object="red car",
    replacement_image="motorcycle.png",
)
```

### Remove Object

Remove an object and fill with background:

```python
result = swapper.remove_object(
    video=video,
    object_prompt="red car",
)
```

### Segment Only

Get object masks without modifying the video:

```python
track = swapper.segment_only(
    video=video,
    object_prompt="person",
)

print(f"Tracked {len(track.masks)} frames")
for mask in track.masks:
    print(f"Frame {mask.frame_index}: confidence {mask.confidence:.2f}")
```

### Visualize Tracking

Debug visualization of tracked object:

```python
debug_frames = swapper.visualize_track(video, track)
debug_video = Video.from_frames(debug_frames, video.fps)
debug_video.save("debug_tracking.mp4")
```

### Progress Tracking

```python
def on_progress(stage: str, progress: float) -> None:
    print(f"[{progress*100:5.1f}%] {stage}")

result = swapper.swap(
    video=video,
    source_object="red car",
    target_object="blue motorcycle",
    progress_callback=on_progress,
)
```

::: videopython.ai.ObjectSwapper

## SwapResult

Result of a swap or remove operation.

```python
result = swapper.swap(video, "car", "truck")

print(f"Processed {len(result.swapped_frames)} frames")
print(f"Object tracked: {result.source_object}")
print(f"Track confidence: {result.track.masks[0].confidence:.2f}")
```

::: videopython.ai.swapping.SwapResult

## ObjectTrack

Tracked object across multiple frames.

::: videopython.ai.swapping.ObjectTrack

## ObjectMask

Single-frame object mask with confidence and bounding box.

::: videopython.ai.swapping.ObjectMask

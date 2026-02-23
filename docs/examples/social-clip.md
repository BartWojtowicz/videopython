# Social Media Clip

Create a vertical video clip optimized for TikTok, Instagram Reels, or YouTube Shorts.

## Goal

Take a landscape video, extract a segment, convert to vertical 9:16 format, add an intro transition, and mix background audio.

## Full Example

```python
from videopython.base import Video, Crop, FadeTransition
import numpy as np

# Load source video
video = Video.from_path("raw_footage.mp4")

# Extract a 15-second segment and standardize using fluent API
video = video.cut(30, 45).resample_fps(30)

# Convert landscape to vertical (9:16)
# First resize height, then crop width to center
height, width = video.frame_shape[:2]
target_height = 1920
target_width = 1080

# Scale to target height, then center-crop width
scale_factor = target_height / height
new_width = int(width * scale_factor)
video = video.resize(width=new_width, height=target_height)

# Center crop to 9:16
crop_x = (new_width - target_width) // 2
video = Crop(x=crop_x, y=0, width=target_width, height=target_height).apply(video)

# Create a title card from solid color
title_frame = np.full((target_height, target_width, 3), [20, 20, 20], dtype=np.uint8)
title_card = Video.from_image(title_frame, fps=30, length_seconds=2.0)

# Combine with fade transition
final = title_card.transition_to(video, FadeTransition(effect_time_seconds=0.5))

# Add background music
final = final.add_audio_from_file("upbeat_music.mp3")

# Save
final.save("social_clip.mp4")
```

## Step-by-Step Breakdown

### 1. Extract Segment

```python
# Chain transforms with fluent API
video = video.cut(30, 45).resample_fps(30)

# Validate the operations first (optional)
meta = video.metadata.cut(30, 45).resample_fps(30)
print(f"Output: {meta}")  # Check duration, fps
```

### 2. Convert to Vertical

The key is to scale the video so the height matches your target, then crop the width:

```python
# Original: 1920x1080 (16:9)
# Target: 1080x1920 (9:16)

scale_factor = 1920 / 1080  # ~1.78
new_width = int(1920 * 1.78)  # ~3413

# After resize: 3413x1920
# After crop: 1080x1920 (centered)
```

### 3. Add Intro Card

```python
title_frame = np.full((1920, 1080, 3), [20, 20, 20], dtype=np.uint8)
title_card = Video.from_image(title_frame, fps=30, length_seconds=2.0)
```

### 4. Smooth Transition

```python
final = title_card.transition_to(video, FadeTransition(effect_time_seconds=0.5))
```

## Alternative: JSON Editing Plan (`VideoEdit`)

For LLM-generated or UI-generated edits, use `VideoEdit` instead of manual orchestration:

```python
from videopython.base import VideoEdit

plan = {
    "segments": [
        {"source": "raw_footage.mp4", "start": 30.0, "end": 45.0}
    ],
    "post_transforms": [
        {"op": "resize", "args": {"height": 1920}}
    ],
    "post_effects": [
        {"op": "color_adjust", "args": {"brightness": 0.05}}
    ],
}

edit = VideoEdit.from_dict(plan)
edit.validate()  # dry run using VideoMetadata
final = edit.run()
final.save("social_clip.mp4")
```

If you need a parser-aligned JSON Schema for plan generation/validation:

```python
schema = VideoEdit.json_schema()
```

## Tips

- **Aspect Ratios**: TikTok/Reels use 9:16 (1080x1920). YouTube Shorts accepts 9:16 up to 1080x1920.
- **Duration**: Check each platform's latest limits before publishing, and trim your edit accordingly.
- **Audio**: Always add music or narration. Silent videos perform poorly on social platforms.

# Social Media Clip

Create a vertical video clip optimized for TikTok, Instagram Reels, or YouTube Shorts.

## Goal

Take a landscape video, extract a segment, convert to vertical 9:16 format, add a transition intro, and overlay text.

## Full Example

```python
from videopython.base import (
    Video,
    CutSeconds,
    Resize,
    Crop,
    ResampleFPS,
    TransformationPipeline,
    FadeTransition,
)
from videopython.base.effects import FullImageOverlay
from videopython.base.text import ImageText
import numpy as np

# Load source video
video = Video.from_path("raw_footage.mp4")

# Extract a 15-second segment and standardize
pipeline = TransformationPipeline([
    CutSeconds(start=30, end=45),
    ResampleFPS(fps=30),
])
video = pipeline.run(video)

# Convert landscape to vertical (9:16)
# First resize height, then crop width to center
height, width = video.frame_shape[:2]
target_height = 1920
target_width = 1080

# Scale to target height, then center-crop width
scale_factor = target_height / height
new_width = int(width * scale_factor)
video = Resize(width=new_width, height=target_height).apply(video)

# Center crop to 9:16
crop_x = (new_width - target_width) // 2
video = Crop(x=crop_x, y=0, width=target_width, height=target_height).apply(video)

# Create a title card from solid color
title_frame = np.full((target_height, target_width, 3), [20, 20, 20], dtype=np.uint8)
title_card = Video.from_image(title_frame, fps=30, length_seconds=2.0)

# Combine with fade transition
fade = FadeTransition(effect_time_seconds=0.5)
final = fade.apply((title_card, video))

# Add background music
final = final.add_audio_from_file("upbeat_music.mp3")

# Save
final.save("social_clip.mp4")
```

## Step-by-Step Breakdown

### 1. Extract Segment

```python
pipeline = TransformationPipeline([
    CutSeconds(start=30, end=45),  # 15-second clip
    ResampleFPS(fps=30),           # Consistent framerate
])
video = pipeline.run(video)
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
fade = FadeTransition(effect_time_seconds=0.5)
final = fade.apply((title_card, video))
```

## Tips

- **Aspect Ratios**: TikTok/Reels use 9:16 (1080x1920). YouTube Shorts accepts 9:16 up to 1080x1920.
- **Duration**: Keep clips under 60 seconds for Reels, under 3 minutes for TikTok.
- **Audio**: Always add music or narration. Silent videos perform poorly on social platforms.

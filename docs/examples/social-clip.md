# Social Media Clip

Create a vertical video clip optimized for TikTok, Instagram Reels, or YouTube Shorts.

## Goal

Take a landscape video, extract a segment, convert to vertical 9:16 format, add a fade-in, and mix background audio.

## Full Example

```python
from videopython.base import Video
from videopython.editing import CutSeconds, Resize, Crop, ResampleFPS, Fade
from videopython.editing.operation import TimeRange
import numpy as np

# Load source video, then extract and standardize a 15s segment
video = Video.from_path("raw_footage.mp4")
video = CutSeconds(start=30, end=45).apply(video)
video = ResampleFPS(fps=30).apply(video)

# Convert landscape to vertical (9:16):
# scale to target height, then center-crop width
height, width = video.frame_shape[:2]
target_height = 1920
target_width = 1080
new_width = int(width * (target_height / height))
video = Resize(width=new_width, height=target_height).apply(video)
video = Crop(width=target_width, height=target_height).apply(video)  # center mode

# Fade in over the first 0.5s
video = Fade(mode="in", duration=0.5, window=TimeRange(stop=0.5)).apply(video)

# Mix background music
video = video.add_audio_from_file("upbeat_music.mp3")
video.save("social_clip.mp4")
```

## Same Plan via `VideoEdit`

For LLM-generated or UI-generated edits, use `VideoEdit`:

```python
from videopython.editing import VideoEdit

plan = {
    "segments": [{
        "source": "raw_footage.mp4",
        "start": 30.0,
        "end": 45.0,
        "operations": [
            {"op": "resample_fps", "fps": 30},
            {"op": "resize", "height": 1920},
            {"op": "crop", "width": 1080, "height": 1920, "mode": "center"},
            {"op": "fade", "mode": "in", "duration": 0.5,
             "window": {"stop": 0.5}},
        ],
    }],
}

edit = VideoEdit.from_dict(plan)
edit.validate()      # dry run using VideoMetadata
final = edit.run().add_audio_from_file("upbeat_music.mp3")
final.save("social_clip.mp4")
```

If you need a JSON Schema for plan generation/validation:

```python
schema = VideoEdit.json_schema()
```

## Tips

- **Aspect Ratios**: TikTok/Reels use 9:16 (1080x1920). YouTube Shorts accepts 9:16 up to 1080x1920.
- **Duration**: Check each platform's latest limits before publishing, and trim your edit accordingly.
- **Audio**: Always add music or narration. Silent videos perform poorly on social platforms.

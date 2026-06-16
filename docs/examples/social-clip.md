# Social Media Clip

Create a vertical video clip optimized for TikTok, Instagram Reels, or YouTube Shorts.

## Goal

Take a landscape video, extract a segment, convert to vertical 9:16 format, add a fade-in, and mix background audio.

## Full Example

Operations run only through the streaming engine, so the whole edit is a
single `VideoEdit` plan executed with `run_to_file`. The 15s cut is the
segment's `start`/`end`; standardizing, the landscape-to-vertical conversion,
and the fade-in are the segment's `operations`:

```python
from videopython.editing import VideoEdit, SegmentConfig
from videopython.editing.transforms import Resize, Crop, ResampleFPS
from videopython.editing.effects import Fade

# Extract a 15s segment (cut via start/end) and turn it into a vertical 9:16
# clip: standardize fps, scale to the target height, center-crop the width,
# then fade in over the first 0.5s.
edit = VideoEdit(segments=[SegmentConfig(
    source="raw_footage.mp4",
    start=30.0,
    end=45.0,
    operations=[
        ResampleFPS(fps=30),
        Resize(height=1920),                      # scale to height, keep aspect
        Crop(width=1080, height=1920, mode="center"),
        Fade(mode="in", duration=0.5),
    ],
)])

edit.validate()                       # dry run using VideoMetadata
edit.run_to_file("social_clip.mp4")   # streams to disk

# To lay a music bed over the rendered clip, load it back and mix audio:
# from videopython.base import Video
# Video.from_path("social_clip.mp4").add_audio_from_file("upbeat_music.mp3").save("social_clip.mp4")
```

## Same Plan as a Dict

For LLM-generated or UI-generated edits, build the same plan from a dict (the
JSON wire format) instead of operation objects:

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
            {"op": "fade", "mode": "in", "duration": 0.5},
        ],
    }],
}

edit = VideoEdit.from_dict(plan)
edit.validate()      # dry run using VideoMetadata
edit.run_to_file("social_clip.mp4")   # streams to disk
```

If you need a JSON Schema for plan generation/validation:

```python
schema = VideoEdit.json_schema()
```

## Tips

- **Aspect Ratios**: TikTok/Reels use 9:16 (1080x1920). YouTube Shorts accepts 9:16 up to 1080x1920.
- **Duration**: Check each platform's latest limits before publishing, and trim your edit accordingly.
- **Audio**: Always add music or narration. Silent videos perform poorly on social platforms.

# Build a vertical social clip

Turn a landscape source into a 9:16 clip for TikTok, Reels, or Shorts: pick a 15-second
range, standardize the frame rate, convert to vertical, fade in, and lay music over it.

## The plan

The cut is the segment's `start`/`end`; everything else is an operation. Scale to the
target height first, then center-crop the width — that fills the frame instead of
letterboxing it.

```python
from videopython.editing import VideoEdit, SegmentConfig, Resize, Crop, ResampleFPS, Fade

edit = VideoEdit(segments=[SegmentConfig(
    source="raw_footage.mp4",
    start=30.0,
    end=45.0,
    operations=[
        ResampleFPS(fps=30),
        Resize(height=1920),                       # scale to height, keep aspect
        Crop(width=1080, height=1920, mode="center"),
        Fade(mode="in", duration=0.5),
    ],
)])

edit.validate()
edit.run_to_file("social_clip.mp4")
```

## Add a music bed

Audio mixing happens on a `Video`, not in the plan, so render first and mix after:

```python
from videopython.base import Video

(Video.from_path("social_clip.mp4")
      .add_audio_from_file("upbeat_music.mp3")     # overlay=False replaces instead
      .save("social_clip.mp4"))
```

To duck or mute the source audio inside the plan instead, add `VolumeAdjust` — it is an
audio-only effect and takes a `window`:

```python
from videopython.editing import VolumeAdjust, TimeRange

VolumeAdjust(volume=0.2)                                   # whole segment
VolumeAdjust(volume=0.0, window=TimeRange(stop=2.0))       # mute the first 2s
```

## The same plan as data

For a UI or an LLM that stores plans, use the dict (JSON) form — same models, same
validation:

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
edit.validate()
edit.run_to_file("social_clip.mp4")
```

## Reframe around a speaker instead of center-cropping

A center crop cuts the subject in half when they stand off-axis. With the `[ai]` extra,
`FaceTrackingCrop` follows them:

```python
from videopython.ai import FaceTrackingCrop

operations = [
    FaceTrackingCrop(target_aspect=(9, 16), framing_rule="headroom", max_speed=0.1),
    Fade(mode="in", duration=0.5),
]
```

`max_speed` bounds how fast the virtual camera may move, which keeps the result from
jittering. See [AI operations](../reference/ai/operations.md).

## Notes

- **Aspect ratio** — 1080×1920 (9:16) is the safe target for all three platforms.
- **Order matters** — resize before crop, and put `fade` last so it applies to the final
  framing.
- **Silent clips underperform.** Add music or narration.
- **Check each platform's current duration limits** before publishing and trim the
  segment accordingly.

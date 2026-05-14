# Quick Start

This guide covers the essential operations in videopython.

!!! tip "Follow Along"
    All examples assume you have videos to work with. You can use any `.mp4` file, or download sample videos from sites like [Pexels](https://www.pexels.com/videos/).

## Loading Videos

```python
from videopython.base import Video

# Load from file
video = Video.from_path("input.mp4")

# Load a specific segment (more efficient for long videos)
video = Video.from_path("input.mp4", start_second=10, end_second=20)

# Create from a static image
import numpy as np
image = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Black frame
video = Video.from_image(image, fps=24, length_seconds=3.0)

# Check video properties
print(video.metadata)  # 1920x1080 @ 30fps, 10.5 seconds
print(video.total_seconds)
print(video.frame_shape)  # (height, width, channels)
```

## Basic Transformations

Every editing primitive is an `Operation`. Apply one to a `Video`:

```python
from videopython.base import Video, CutSeconds, Resize, ResampleFPS

video = Video.from_path("input.mp4")
video = CutSeconds(start=0, end=10).apply(video)
video = Resize(width=1280, height=720).apply(video)
video = ResampleFPS(fps=30).apply(video)

# Resize preserves aspect ratio when only one dimension is set:
video = Resize(width=1280).apply(video)
```

For multi-step plans use `VideoEdit`, which also gives you a dry-run
via `.validate()`:

```python
from videopython.editing import VideoEdit

edit = VideoEdit.from_dict({
    "segments": [{
        "source": "input.mp4",
        "start": 0,
        "end": 10,
        "operations": [
            {"op": "resize", "width": 1280, "height": 720},
            {"op": "resample_fps", "fps": 30},
        ],
    }]
})
print(edit.validate())   # predicted VideoMetadata, no frames loaded
video = edit.run()
```

## Combining Videos

Concatenate two videos with `+`. They must share fps and dimensions —
align them with `Resize` / `ResampleFPS` first.

```python
from videopython.base import Video

video1 = Video.from_path("clip1.mp4")
video2 = Video.from_path("clip2.mp4")
combined = video1 + video2
```

For multi-segment edits with auto-matching of fps/resolution, use
`VideoEdit` (sets `match_to_lowest_fps` / `match_to_lowest_resolution`
to `true` by default).

## Working with Audio

```python
from videopython.base import Video

video = Video.from_path("input.mp4")

# Add audio from file (overlays on existing audio)
video = video.add_audio_from_file("music.mp3")

# Add audio without overlay (replaces existing)
video = video.add_audio_from_file("narration.mp3", overlay=False)

# Save with audio
video.save("output.mp4")
```

## Saving Videos

```python
video.save("output.mp4")  # Default MP4
video.save("output.webm", format="webm")  # WebM format
video.save("output.mov", format="mov")  # QuickTime

# Supported formats: mp4, avi, mov, mkv, webm

# Control encoding quality and speed
video.save("output.mp4", preset="slow", crf=18)  # Higher quality, slower encoding
video.save("output.mp4", preset="ultrafast", crf=28)  # Faster encoding, lower quality
```

!!! tip "Encoding Options"
    - `preset`: Speed/compression tradeoff. Options: ultrafast, superfast, veryfast, faster, fast, medium (default), slow, slower, veryslow. Slower presets produce smaller files.
    - `crf`: Quality level (0-51). Default is 23. Lower values = better quality, larger files. 18 is visually lossless.

## AI Features (Quick Preview)

!!! note "Local AI"
    AI features run locally and may download model weights on first use.

```python
from videopython.ai import TextToImage, TextToSpeech

# Generate an image
generator = TextToImage()
image = generator.generate_image("A sunset over mountains")

# Generate speech
tts = TextToSpeech()
audio = tts.generate_audio("Welcome to videopython!")
```

See the [API Reference](../api/index.md) for complete documentation.

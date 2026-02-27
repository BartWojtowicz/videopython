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

```python
from videopython.base import Video

video = Video.from_path("input.mp4")

# Chain transformations with fluent API
video = video.cut(0, 10).resize(1280, 720).resample_fps(30)

# Or apply transforms one at a time
video = Video.from_path("input.mp4")
video = video.cut(1.5, 6.5)
video = video.resize(width=1280)  # Height calculated to preserve aspect ratio

# Validate operations before executing (fast, metadata only)
meta = Video.from_path("input.mp4").metadata
output_meta = meta.cut(0, 10).resize(1280, 720)
print(f"Output will be: {output_meta}")  # Check dimensions, duration, fps
```

## Combining Videos

!!! warning "Matching Dimensions"
    Videos must have the same dimensions and FPS to be combined. Use `.resize()` and `.resample_fps()` first if needed.

```python
from videopython.base import Video, FadeTransition, BlurTransition

video1 = Video.from_path("clip1.mp4")
video2 = Video.from_path("clip2.mp4")

# Simple concatenation (videos must have same dimensions and FPS)
combined = video1 + video2

# With fade transition (fluent API)
combined = video1.transition_to(video2, FadeTransition(effect_time_seconds=1.5))

# With blur transition
combined = video1.transition_to(video2, BlurTransition(effect_time_seconds=1.0))

# Validate transition compatibility first
meta1 = video1.metadata
meta2 = video2.metadata
combined_meta = meta1.transition_to(meta2, effect_time=1.5)  # Raises if incompatible
```

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

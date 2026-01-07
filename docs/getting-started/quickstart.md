# Quick Start

This guide covers the essential operations in VideoPython.

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
from videopython.base import (
    Video,
    CutSeconds,
    Resize,
    ResampleFPS,
    TransformationPipeline,
)

video = Video.from_path("input.mp4")

# Apply individual transforms
cut = CutSeconds(start=1.5, end=6.5)
video = cut.apply(video)

# Or use a pipeline for multiple transforms
pipeline = TransformationPipeline([
    CutSeconds(start=0, end=10),
    ResampleFPS(fps=30),
    Resize(width=1280, height=720),
])
video = pipeline.run(video)
```

## Combining Videos

```python
from videopython.base import Video, FadeTransition, BlurTransition

video1 = Video.from_path("clip1.mp4")
video2 = Video.from_path("clip2.mp4")

# Simple concatenation (videos must have same dimensions and FPS)
combined = video1 + video2

# With fade transition
fade = FadeTransition(effect_time_seconds=1.5)
combined = fade.apply((video1, video2))

# With blur transition
blur = BlurTransition(effect_time_seconds=1.0)
combined = blur.apply((video1, video2))
```

## Working with Audio

```python
from videopython.base import Video

video = Video.from_path("input.mp4")

# Add audio from file (overlays on existing audio)
video.add_audio_from_file("music.mp3")

# Add audio without overlay (replaces existing)
video.add_audio_from_file("narration.mp3", overlay=False)

# Save with audio
video.save("output.mp4")
```

## Saving Videos

```python
video.save("output.mp4")  # Default MP4
video.save("output.webm", format="webm")  # WebM format
video.save("output.mov", format="mov")  # QuickTime

# Supported formats: mp4, avi, mov, mkv, webm
```

## AI Features (Quick Preview)

```python
import asyncio
from videopython.ai import TextToImage, TextToSpeech

# Generate an image with DALL-E
async def generate():
    generator = TextToImage(backend="openai")
    image = await generator.generate_image("A sunset over mountains")
    return image

image = asyncio.run(generate())

# Generate speech with OpenAI TTS
async def speak():
    tts = TextToSpeech(backend="openai")
    audio = await tts.generate_audio("Welcome to VideoPython!")
    return audio

audio = asyncio.run(speak())
```

See the [API Reference](../api/index.md) for complete documentation.

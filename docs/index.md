# VideoPython

Minimal video generation and processing library designed with short-form videos in mind, with focus on simplicity and ease of use for both humans and AI agents.

## Features

- **Simple video editing**: Cut, resize, resample FPS, and combine videos
- **Transitions**: Fade, blur, and instant transitions between clips
- **Audio support**: Add music, speech, or overlay audio tracks
- **Text overlays**: Render subtitles with word-level highlighting
- **AI-powered generation**: Generate videos, images, music, and speech from text
- **AI understanding**: Transcribe audio, describe frames, detect scenes

## Quick Example

```python
from videopython.base import Video, CutSeconds, FadeTransition

# Load and trim videos
video1 = Video.from_path("intro.mp4")
video2 = Video.from_path("main.mp4")

# Apply transformations
cut = CutSeconds(start=0, end=5)
video1 = cut.apply(video1)

# Combine with fade transition
fade = FadeTransition(effect_time_seconds=1.0)
final = fade.apply((video1, video2))

# Add audio and save
final.add_audio_from_file("background.mp3")
final.save("output.mp4")
```

## Installation

```bash
pip install videopython

# With AI features
pip install "videopython[ai]"
```

See the [Installation Guide](getting-started/installation.md) for more details.

# videopython

A minimal Python library for video editing, processing, and AI workflows, built for short-form content.

## Quick Example

```python
from videopython.base import Video
from videopython.ai import TextToImage, ImageToVideo, TextToSpeech

def create_video():
    # Generate an image and animate it
    image = TextToImage(backend="openai").generate_image(
        "A cozy coffee shop on a rainy evening, warm lighting"
    )
    video = ImageToVideo().generate_video(image=image, fps=24)
    video = video.resize(1080, 1920)  # Vertical format

    # Add narration
    audio = TextToSpeech(backend="openai").generate_audio(
        "Sometimes the best ideas come with a cup of coffee and the sound of rain."
    )
    video = video.add_audio(audio)
    video.save("coffee_shop.mp4")

create_video()
```

## What You Can Do

- **Edit videos** - Cut, resize, crop, resample FPS, combine clips with effects
- **Add transitions** - Fade, blur, or instant transitions between segments
- **Generate content** - Create images, videos, speech, and music from text prompts
- **Transcribe and subtitle** - Auto-generate word-level subtitles from speech
- **Analyze video** - Detect objects, faces, text, scenes, actions, and motion
- **Dub and revoice** - Translate speech to 50+ languages with voice cloning
- **Swap objects** - Replace or remove objects using AI segmentation

## Installation

```bash
# Base features
pip install videopython

# AI features
pip install "videopython[ai]"
```

See the [Installation Guide](getting-started/installation.md) for FFmpeg setup and configuration.

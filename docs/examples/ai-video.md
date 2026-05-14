# AI-Generated Video

Create a video entirely from AI-generated content: images, animation, and narration.

## Goal

Generate images from text prompts, animate them into video segments, add AI-generated speech, and combine everything with transitions.

## Full Example

```python
from videopython.base import Resize, Fade
from videopython.base.operation import TimeRange
from videopython.ai import TextToImage, ImageToVideo, TextToSpeech


def create_ai_video():
    scenes = [
        {"image_prompt": "A serene mountain landscape at sunrise, photorealistic",
         "narration": "In the mountains, every sunrise brings new possibilities."},
        {"image_prompt": "A flowing river through a forest, cinematic lighting",
         "narration": "Nature flows with endless energy and grace."},
        {"image_prompt": "A starry night sky over a calm lake, dramatic",
         "narration": "And when night falls, the universe reveals its wonders."},
    ]

    image_gen = TextToImage()
    video_gen = ImageToVideo()
    speech_gen = TextToSpeech()

    videos = []
    for scene in scenes:
        image = image_gen.generate_image(scene["image_prompt"])
        video = video_gen.generate_video(image=image, fps=24)
        video = Resize(width=1920, height=1080).apply(video)
        audio = speech_gen.generate_audio(scene["narration"])
        videos.append(video.add_audio(audio))

    # Concatenate, with a 1s fade-in on each follow-on segment
    final = videos[0]
    for next_video in videos[1:]:
        next_video = Fade(mode="in", duration=1.0,
                          window=TimeRange(stop=1.0)).apply(next_video)
        final = final + next_video
    return final


video = create_ai_video()
video.save("ai_generated.mp4")
```

## Step-by-Step Breakdown

### 1. Generate Images

```python
image_gen = TextToImage()  # Uses local SDXL pipeline
image = image_gen.generate_image("A serene mountain landscape at sunrise")
```

### 2. Animate to Video

```python
video_gen = ImageToVideo()  # Uses local CogVideoX1.5-5B-I2V
video = video_gen.generate_video(image=image, fps=24)
```

!!! note "Local Models"
    `ImageToVideo` and `TextToVideo` require significant GPU memory (CUDA). An NVIDIA A40 or better is recommended for video generation.

### 3. Generate Speech

```python
speech_gen = TextToSpeech()  # Uses local Chatterbox Multilingual TTS
audio = speech_gen.generate_audio("Your narration text here")
```

### 4. Combine Segments

```python
from videopython.base import Fade
from videopython.base.operation import TimeRange

next_video = Fade(mode="in", duration=1.0, window=TimeRange(stop=1.0)).apply(next_video)
final = final + next_video
```

## Tips

- **Consistency**: Use similar prompt styles across scenes for visual coherence.
- **Timing**: Match narration length to video segment duration.
- **Performance**: Local generation quality and speed depend heavily on your GPU and model choice.

# AI-Generated Video

Create a video entirely from AI-generated content: images, animation, and narration.

## Goal

Generate images from text prompts, animate them into video segments, add AI-generated speech, and combine everything with transitions.

## Full Example

```python
from videopython.base import Video, FadeTransition
from videopython.ai import TextToImage, ImageToVideo, TextToSpeech

def create_ai_video():
    # Define scenes with prompts
    scenes = [
        {
            "image_prompt": "A serene mountain landscape at sunrise, photorealistic",
            "narration": "In the mountains, every sunrise brings new possibilities.",
        },
        {
            "image_prompt": "A flowing river through a forest, cinematic lighting",
            "narration": "Nature flows with endless energy and grace.",
        },
        {
            "image_prompt": "A starry night sky over a calm lake, dramatic",
            "narration": "And when night falls, the universe reveals its wonders.",
        },
    ]

    # Initialize AI generators
    image_gen = TextToImage(backend="openai")
    video_gen = ImageToVideo(backend="local")
    speech_gen = TextToSpeech(backend="openai")

    videos = []
    for scene in scenes:
        # Generate image
        image = image_gen.generate_image(scene["image_prompt"])

        # Animate image to video (4 seconds)
        video = video_gen.generate_video(image=image, fps=24)

        # Resize to consistent dimensions using fluent API
        video = video.resize(1920, 1080)

        # Generate narration audio
        audio = speech_gen.generate_audio(scene["narration"])

        # Add audio to video segment
        video = video.add_audio(audio)
        videos.append(video)

    # Combine all segments with fade transitions
    final = videos[0]
    for next_video in videos[1:]:
        final = final.transition_to(next_video, FadeTransition(effect_time_seconds=1.0))

    return final

# Run and save
video = create_ai_video()
video.save("ai_generated.mp4")
```

## Step-by-Step Breakdown

### 1. Generate Images

```python
image_gen = TextToImage(backend="openai")  # Uses DALL-E 3
image = image_gen.generate_image("A serene mountain landscape at sunrise")
```

Available backends:

| Backend | Model |
|---------|-------|
| `openai` | DALL-E 3 |
| `local` | Stable Diffusion XL |

### 2. Animate to Video

```python
video_gen = ImageToVideo(backend="local")  # Uses Stable Video Diffusion
video = video_gen.generate_video(image=image, fps=24)
```

!!! note "Local Models"
    Local backends (`ImageToVideo`, `TextToVideo`) require significant GPU memory. Expect 8GB+ VRAM for smooth operation.

### 3. Generate Speech

```python
speech_gen = TextToSpeech(backend="openai")  # Uses OpenAI TTS
audio = speech_gen.generate_audio("Your narration text here")
```

Available backends:

| Backend | Model |
|---------|-------|
| `openai` | OpenAI TTS |
| `elevenlabs` | Multilingual v2 |
| `local` | Bark |

### 4. Combine Segments

```python
final = video1.transition_to(video2, FadeTransition(effect_time_seconds=1.0))
```

## Tips

- **Consistency**: Use similar prompt styles across scenes for visual coherence.
- **Timing**: Match narration length to video segment duration.
- **Quality**: OpenAI backends generally produce higher quality but cost money. Local backends are free but require GPU.

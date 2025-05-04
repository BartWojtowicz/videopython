# About

Minimal video generation and processing library.

## Setup 

### Install ffmpeg
```bash
# Install with brew for MacOS:
brew install ffmpeg
# Install with apt-get for Ubuntu:
sudo apt-get install ffmpeg
```

### Install with pip
```bash
pip install videopython[ai]
```
> You can install without `[ai]` dependencies for basic video handling and processing. 
> The funcionalities found in `videopython.ai` won't work.

## Basic Usage

### Video handling

```python
from videopython.base.video import Video

# Load videos and print metadata
video1 = Video.from_path("tests/test_data/small_video.mp4")
print(video1)

video2 = Video.from_path("tests/test_data/big_video.mp4")
print(video2)

# Define the transformations
from videopython.base.transforms import CutSeconds, ResampleFPS, Resize, TransformationPipeline

pipeline = TransformationPipeline(
    [CutSeconds(start=1.5, end=6.5), ResampleFPS(fps=30), Resize(width=1000, height=1000)]
)
video1 = pipeline.run(video1)
video2 = pipeline.run(video2)

# Combine videos, add audio and save
from videopython.base.transitions import FadeTransition

fade = FadeTransition(effect_time_seconds=3.0)
video = fade.apply(videos=(video1, video2))
video.add_audio_from_file("tests/test_data/test_audio.mp3")

savepath = video.save()
```

### Video Generation

> Using Nvidia A40 or better is recommended for the `videopython.ai` module.
```python
# Generate image and animate it
from videopython.ai.generation import ImageToVideo
from videopython.ai.generation import TextToImage
from videopython.ai.generation import TextToMusic

image = TextToImage().generate_image(prompt="Golden Retriever playing in the park")
video = ImageToVideo().generate_video(image=image, fps=24)

# Video generation directly from prompt
from videopython.ai.generation import TextToVideo
video_gen = TextToVideo()
video = video_gen.generate_video("Dogs playing in the snow")
for _ in range(10):
    video += video_gen.generate_video("Dogs playing in the snow")

# Cut the first 2 seconds
from videopython.base.transforms import CutSeconds
transformed_video = CutSeconds(start_second=0, end_second=2).apply(video.copy())

# Upsample to 30 FPS
from videopython.base.transforms import ResampleFPS
transformed_video = ResampleFPS(new_fps=30).apply(transformed_video)

# Resize to 1000x1000
from videopython.base.transforms import Resize
transformed_video = Resize(width=1000, height=1000).apply(transformed_video)

# Add generated music
# MusicGen cannot generate more than 1503 tokens (~30seconds of audio)
text_to_music = TextToMusic()
audio = text_to_music.generate_audio("Happy dogs playing together in a park", max_new_tokens=256)
transformed_video.add_audio(audio=audio)

filepath = transformed_video.save()
```

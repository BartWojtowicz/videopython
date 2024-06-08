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
pip install videopython[generation]
```
> You can install without `[generation]` dependencies for basic video handling and processing. 
> The funcionalities found in `videopython.generation` won't work.

## Basic Usage
> Using Nvidia A40 or better is recommended for the `videopython.generation` module.

```python
# Generate image and animate it
from videopython.generation import ImageToVideo
from videopython.generation import TextToImage

image = TextToImage().generate_image(prompt="Golden Retriever playing in the park")
video = ImageToVideo().generate_video(image=image, fps=24)

# Video generation directly from prompt
from videopython.generation import TextToVideo
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

filepath = transformed_video.save()
```

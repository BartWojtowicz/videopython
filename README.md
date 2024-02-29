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
pip install videopython
```

## Basic Usage

```python
# Generate image and animate it (currently need stabilit API for image generation )
from videopython.generation.video import ImageToVideo
from videopython.generation.image import TextToImage

image = TextToImage(stability_key=<YOUR_KEY>).generate_image(prompt="Space image")
video = ImageToVideo().generate_video(image=image, fps=24)

# Video generation with Zeroscope model
from videopython.generation.video import TextToVideo
video_gen = TextToVideo(gpu_optimized=True)
video = video_gen.generate_video("Dogs playing in the snow")
for _ in range(10):
    video += video_gen.generate_video("The same happy dog all over again")

# Cut the first 2 seconds
from videopython.base.transforms import CutSeconds
transformed_video = CutSeconds(start_second=0, end_second=2).apply(video.copy())

# Upsample to 30 FPS
from videopython.base.transforms import ResampleFPS
transformed_video = ResampleFPS(new_fps=30).apply(transformed_video)

# Resize to 1000x1000
from videopython.base.transforms import Resize
transformed_video = Resize(new_width=1000, new_height=1000)).apply(transformed_video)

filepath = transformed_video.save()
```

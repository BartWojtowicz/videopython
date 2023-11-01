# About

Minimal video generation and processing library written on top of `numpy`, `opencv` and `ffmpeg`.

## Setup 

```bash
# Install ffmpeg with brew for MacOS:
brew install ffmpeg
# Install with apt-get for Ubuntu:
sudo apt-get install ffmpeg

# Install python dependencies
pip3 install -r requirements.txt
```

## Basic Usage

```python
from videopython.base import Video
from videopython.base.transitions import FadeTransition

# Load video
video = Video.from_path("tests/test_data/fast_benchmark.mp4")
print(video.metadata)
print(video.frames.shape) # Video is based on numpy representation of frames

transformed_video = video + video
print(transformed_video.metadata)

fade = FadeTransition(2.0) # 2s effect time
transformed_video = fade.apply(videos=(video, transformed_video))
print(transformed_video.metadata)

transformed_video.save("./data/exported/")
```

### Running Unit Tests
```bash
PYTHONPATH=. pytest videopython
```

### How to download stock data?
[Read here.](./scripts/README.md)

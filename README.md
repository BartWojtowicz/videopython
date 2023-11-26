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
from videopython.base.video import Video
from videopython.base.transitions import FadeTransition

# Load video
video = Video.from_path("tests/test_data/fast_benchmark.mp4")
print(video.metadata)
print(video.frames.shape) # Video is based on numpy representation of frames

# Generate videos
video1 = Video.from_prompt("Dogs playing in the snow.")
video2 = Video.from_prompt("Dogs going back home.")

# Add videos
combined_video = video1 + video2
print(combined_video.metadata)

# Apply fade transition between videos
fade = FadeTransition(0.5) # 0.5s effect time
faded_video = fade.apply(videos=(video1, video2))
print(faded_video.metadata)

# Add audio from file
faded_video.add_audio_from_file("tests/test_data/test_audio.mp3")

# Save to a file
faded_video.save("my_video.mp4")
```

### Running Unit Tests
```bash
PYTHONPATH=./src/ pytest
```
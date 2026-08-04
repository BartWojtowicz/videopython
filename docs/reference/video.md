# Video

`videopython.base` — the core data containers and I/O primitives. No AI dependencies.

## Video

Holds decoded frames in memory as a numpy array. Use it for short clips, random access,
and audio mixing. For anything long, prefer [`VideoEdit`](video-edit.md) or
[`FrameIterator`](#frameiterator) — see
[Process hour-long videos](../how-to/long-videos.md).

```python
from videopython.base import Video

video = Video.from_path("input.mp4")
video = Video.from_path("input.mp4", start_second=10, end_second=20)   # bounded load

import numpy as np
video = Video.from_image(np.zeros((1080, 1920, 3), dtype=np.uint8), fps=24, length_seconds=3.0)

combined = video_a + video_b        # concat; requires matching fps and dimensions

video = video.add_audio_from_file("music.mp3")               # overlays existing audio
video = video.add_audio_from_file("narration.mp3", overlay=False)   # replaces it

video.save("output.mp4")
video.save("output.webm", format="webm")     # mp4 | avi | mov | mkv | webm
video.save("output.mp4", preset="slow", crf=18)
```

`preset` is the FFmpeg speed/compression trade-off (`ultrafast`, `superfast`, `veryfast`,
`faster`, `fast`, `medium` (default), `slow`, `slower`, `veryslow`); slower presets
produce smaller files. `crf` is quality, 0–51, default 23, lower is better; 18 is
visually lossless.

::: videopython.base.Video
    options:
      members:
        - __init__
        - from_path
        - from_frames
        - from_image
        - save
        - copy
        - split
        - add_audio
        - add_audio_from_file
        - is_loaded
        - video_shape
        - frame_shape
        - total_seconds
        - metadata

## VideoMetadata

Reads the container header only — no frames are decoded.

```python
from videopython.base import VideoMetadata

meta = VideoMetadata.from_path("video.mp4")
meta.width, meta.height, meta.fps, meta.total_seconds, meta.frame_count
```

::: videopython.base.VideoMetadata

## FrameIterator

Streams frames one at a time, O(1) memory.

```python
from videopython.base import FrameIterator

with FrameIterator("long_video.mp4") as frames:
    for frame_idx, frame in frames:
        ...          # (H, W, 3) uint8, RGB

with FrameIterator("video.mp4", start_second=10.0, end_second=60.0) as frames:
    for frame_idx, frame in frames:
        ...
```

::: videopython.base.FrameIterator

## Exceptions

All videopython errors derive from `VideoPythonError`.

::: videopython.base.VideoPythonError

::: videopython.base.VideoError

::: videopython.base.VideoLoadError

::: videopython.base.VideoMetadataError

::: videopython.base.TransformError

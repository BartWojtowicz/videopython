# Process hour-long videos

`Video.from_path()` loads every frame into RAM. A 2-hour 1080p30 source is ~216,000
frames — over a terabyte uncompressed — so for long sources you want the APIs that never
hold more than one frame at a time.

## Pick the right entry point

| Approach | Memory | Use it for |
|---|---|---|
| `VideoEdit.run_to_file()` | O(1) (~250 MB) | Editing with transforms and effects |
| `FrameIterator` | O(1) | Single-pass analysis over frames |
| `VideoDubber.dub_file()` | O(audio + weights) | Dubbing without touching frames |
| `Video.from_path()` | O(all frames) | Short clips that need random access |

## Edit without loading frames

`run_to_file()` streams FFmpeg decode → per-frame effects → FFmpeg encode. Memory is flat
regardless of duration — this is the normal path, not a special mode.

```python
from videopython.editing import VideoEdit

edit = VideoEdit.from_dict({
    "segments": [{
        "source": "2_hour_movie.mp4",
        "start": 0,
        "end": 7200,
        "operations": [
            {"op": "resize", "width": 1920, "height": 1080},
            {"op": "color_adjust", "saturation": 0, "contrast": 1.15},
            {"op": "fade", "mode": "in_out", "duration": 1.0},
            {"op": "volume_adjust", "volume": 1.5},
        ],
    }],
})
edit.run_to_file("output.mp4", crf=20, preset="medium")
```

Operations that need context stream too — pass it to the runner:

```python
edit.run_to_file("output.mp4", context={"transcription": transcription})
```

## Check a plan will stream before you download the source

Every registered operation streams, but a few plan *shapes* have no streaming strategy
(for example a per-frame effect ordered after burned-in subtitles). Those are rejected
with structured `STREAMING_UNSUPPORTED` errors before any decode.

`streamability()` answers the question without touching the disk, which makes it usable
as a job-admission gate:

```python
report = edit.streamability()
report.streamable      # bool — will the plan run?
report.unstreamable    # the offending ops, each with a reason and a reorder hint
report.errors()        # the same, as structured PlanErrors
```

`edit.check(meta)` reports the same errors alongside the ordinary validity errors. The
rules and how to fix each shape are in
[the streaming engine](../explanation/streaming-engine.md#shapes-that-cannot-stream).

## Iterate frames yourself

```python
from videopython.base import FrameIterator

with FrameIterator("long_video.mp4") as frames:
    for frame_idx, frame in frames:
        ...          # frame is an (H, W, 3) uint8 RGB array; one at a time

# Bounded to a time range
with FrameIterator("movie.mp4", start_second=3600, end_second=4200) as frames:
    for frame_idx, frame in frames:
        ...
```

Worked example — one thumbnail per minute:

```python
import os

from PIL import Image

from videopython.base import FrameIterator, VideoMetadata


def extract_thumbnails(video_path: str, output_dir: str, interval_seconds: float = 60.0):
    os.makedirs(output_dir, exist_ok=True)
    fps = VideoMetadata.from_path(video_path).fps
    interval_frames = int(interval_seconds * fps)

    with FrameIterator(video_path) as frames:
        for frame_idx, frame in frames:
            if frame_idx % interval_frames == 0:
                img = Image.fromarray(frame)
                img.thumbnail((320, 180))
                img.save(f"{output_dir}/thumb_{frame_idx / fps:.0f}s.jpg")
```

## Keep AI analysis affordable

`VideoAnalyzer` runs a per-scene VLM pass, so cost scales with scene count. Use
`sampling="low"` (8-frame budget per scene, 20-second adjacent-scene merge) and enable
only the analyzers you need:

```python
from videopython.ai import VideoAnalyzer, VideoAnalysisConfig

config = VideoAnalysisConfig(
    enabled_analyzers={"audio_to_text", "semantic_scene_detector", "scene_vlm"},
)
analysis = VideoAnalyzer(config=config, sampling="low").analyze_path("long_video.mp4")

for scene in (analysis.scenes.samples if analysis.scenes else []):
    print(scene.scene_index, scene.start_second, scene.end_second)
    if scene.scene_description:
        print("  ", scene.scene_description.caption)
```

## Dub without loading frames

`dub_and_replace()` goes through `Video.from_path()` and is impractical on long sources.
`dub_file()` works on paths: it extracts the audio with FFmpeg, dubs the audio only, and
muxes it back with a video **stream copy** — no re-encode. Add `low_memory=True` to
unload each stage's model (Whisper, Demucs, translator, Chatterbox) after it runs.

```python
from videopython.ai.dubbing import VideoDubber

dubber = VideoDubber(low_memory=True)
result = dubber.dub_file(
    input_path="2_hour_movie.mp4",
    output_path="dubbed.mp4",
    target_lang="es",
    voice_clone=True,
    preserve_background=True,
)
```

Peak memory is model weights plus the audio track — independent of resolution and
duration. See [Dub a video](dubbing.md).

## Notes

- **Read metadata first.** `VideoMetadata.from_path()` tells you what you are dealing
  with for free.
- **Bound the work.** `start_second`/`end_second` on `FrameIterator`, `window` on
  effects, and a segment's `start`/`end` all keep you from paying for frames you discard.
- **Sample sparsely for analysis.** 0.1–0.5 fps is plenty for most understanding tasks.

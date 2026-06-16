# AI-Generated Video

Create a video entirely from AI-generated content: images, animation, and narration.

## Goal

Generate images from text prompts, animate them into video segments, add AI-generated speech, and combine everything with transitions.

## Full Example

AI generation produces in-memory `Video` objects; narration is attached with
`Video.add_audio`. Editing operations run only through the streaming engine, so
each generated scene is saved to disk, then all scenes are assembled in a single
`VideoEdit` plan (one `SegmentConfig` per scene) executed with `run_to_file`.
The per-scene `operations` standardize resolution, and a `transition_in`
crossfades each follow-on scene into the previous one:

```python
from pathlib import Path

from videopython.editing import VideoEdit, SegmentConfig, TransitionSpec
from videopython.editing.transforms import Resize
from videopython.ai import TextToImage, ImageToVideo, TextToSpeech
from videopython.base.video import VideoMetadata


def create_ai_video(output_path: str, workdir: str = "scenes"):
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

    # Generate each scene with narration and save it to disk.
    Path(workdir).mkdir(parents=True, exist_ok=True)
    scene_paths = []
    for i, scene in enumerate(scenes):
        image = image_gen.generate_image(scene["image_prompt"])
        video = video_gen.generate_video(image=image)
        audio = speech_gen.generate_audio(scene["narration"])
        path = f"{workdir}/scene_{i}.mp4"
        video.add_audio(audio).save(path)
        scene_paths.append(path)

    # Assemble every scene into one streaming plan. Resize standardizes each
    # scene to 1080p; a 1s dissolve crossfades each follow-on scene in (the
    # first scene has no predecessor, so it carries no transition_in).
    segments = []
    for i, path in enumerate(scene_paths):
        meta = VideoMetadata.from_path(path)
        segments.append(SegmentConfig(
            source=path,
            start=0,
            end=meta.total_seconds,
            operations=[Resize(width=1920, height=1080)],
            transition_in=None if i == 0 else TransitionSpec(type="dissolve", duration=1.0),
        ))

    edit = VideoEdit(segments=segments)
    edit.run_to_file(output_path)


create_ai_video("ai_generated.mp4")
```

## Step-by-Step Breakdown

### 1. Generate Images

```python
image_gen = TextToImage()  # Uses local SDXL pipeline
image = image_gen.generate_image("A serene mountain landscape at sunrise")
```

### 2. Animate to Video

```python
video_gen = ImageToVideo()  # Uses local CogVideoX1.5-5B-I2V (outputs at 16 fps)
video = video_gen.generate_video(image=image)
```

!!! note "Local Models"
    `ImageToVideo` and `TextToVideo` require significant GPU memory (CUDA). An NVIDIA A40 or better is recommended for video generation.

### 3. Generate Speech

```python
speech_gen = TextToSpeech()  # Uses local Chatterbox Multilingual TTS
audio = speech_gen.generate_audio("Your narration text here")
```

### 4. Combine Segments

Saved scenes are assembled in a single streaming plan. Each scene is one
`SegmentConfig`; a `transition_in` crossfades it into the previous scene (so the
first scene carries none):

```python
from videopython.editing import VideoEdit, SegmentConfig, TransitionSpec
from videopython.base.video import VideoMetadata

segments = []
for i, path in enumerate(scene_paths):
    meta = VideoMetadata.from_path(path)
    segments.append(SegmentConfig(
        source=path,
        start=0,
        end=meta.total_seconds,
        transition_in=None if i == 0 else TransitionSpec(type="dissolve", duration=1.0),
    ))

VideoEdit(segments=segments).run_to_file("ai_generated.mp4")
```

### 5. Reframe for Vertical (optional)

To turn a horizontal scene into a vertical 9:16 clip, add the AI `face_crop`
operation (the `FaceTrackingCrop` transform), which tracks the speaker's face
and crops around it. It is just another entry in a segment's `operations` list:

```python
from videopython.editing import VideoEdit, SegmentConfig
from videopython.ai.transforms import FaceTrackingCrop

edit = VideoEdit(segments=[SegmentConfig(
    source="scenes/scene_0.mp4",
    start=0,
    end=5,
    operations=[FaceTrackingCrop(target_aspect=(9, 16), framing_rule="center")],
)])
edit.run_to_file("scene_0_vertical.mp4")
```

## Tips

- **Consistency**: Use similar prompt styles across scenes for visual coherence.
- **Timing**: Match narration length to video segment duration.
- **Performance**: Local generation quality and speed depend heavily on your GPU and model choice.

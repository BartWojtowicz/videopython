# Assemble a video from AI-generated media

Generate images from prompts, animate them, add narration, and cut the result together
with crossfades. Everything runs locally; image and video generation need a CUDA GPU
(see [Install](../install.md#hardware)).

## The shape of the job

Generation produces in-memory `Video` and `Audio` objects. Editing operations run only
through the streaming engine, which reads from files — so save each generated scene, then
assemble the saved files in one plan.

```python
from pathlib import Path

from videopython.ai import TextToImage, ImageToVideo, TextToSpeech
from videopython.base.video import VideoMetadata
from videopython.editing import VideoEdit, SegmentConfig, TransitionSpec, Resize


def create_ai_video(output_path: str, workdir: str = "scenes") -> None:
    scenes = [
        {"image_prompt": "A serene mountain landscape at sunrise, photorealistic",
         "narration": "In the mountains, every sunrise brings new possibilities."},
        {"image_prompt": "A flowing river through a forest, cinematic lighting",
         "narration": "Nature flows with endless energy and grace."},
        {"image_prompt": "A starry night sky over a calm lake, dramatic",
         "narration": "And when night falls, the universe reveals its wonders."},
    ]

    image_gen, video_gen, speech_gen = TextToImage(), ImageToVideo(), TextToSpeech()

    Path(workdir).mkdir(parents=True, exist_ok=True)
    scene_paths = []
    for i, scene in enumerate(scenes):
        image = image_gen.generate_image(scene["image_prompt"])
        video = video_gen.generate_video(image=image)
        audio = speech_gen.generate_audio(scene["narration"])
        path = f"{workdir}/scene_{i}.mp4"
        video.add_audio(audio).save(path)
        scene_paths.append(path)

    # One segment per scene. Resize standardizes to 1080p; a 1s dissolve crossfades
    # each follow-on scene in (the first has no predecessor, so it carries none).
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

    VideoEdit(segments=segments).run_to_file(output_path)


create_ai_video("ai_generated.mp4")
```

## The generators

| Step | Class | Local model |
|---|---|---|
| Prompt → still | `TextToImage` | Qwen-Image |
| Still → motion | `ImageToVideo` | Wan2.2-I2V-A14B (16 fps output) |
| Prompt → motion | `TextToVideo` | Wan2.2-T2V-A14B |
| Text → narration | `TextToSpeech` | Chatterbox Multilingual |
| Text → music bed | `TextToMusic` | MusicGen |

Full signatures in [AI generation](../reference/ai/generation.md).

## Shape the narration delivery

`generate_audio` forwards three Chatterbox knobs. Each defaults to `None`, meaning "let
Chatterbox decide":

```python
dramatic = speech_gen.generate_audio(
    "We made it.",
    exaggeration=0.85,   # more expressive
    cfg_weight=0.35,     # slower pacing
)
```

## Reframe a generated scene for vertical

`face_crop` is just another operation:

```python
from videopython.ai import FaceTrackingCrop
from videopython.editing import VideoEdit, SegmentConfig

edit = VideoEdit(segments=[SegmentConfig(
    source="scenes/scene_0.mp4",
    start=0,
    end=5,
    operations=[FaceTrackingCrop(target_aspect=(9, 16), framing_rule="center")],
)])
edit.run_to_file("scene_0_vertical.mp4")
```

## Notes

- **Keep prompts stylistically consistent** across scenes or the cut will not feel like
  one piece.
- **Match narration length to scene length** — `ImageToVideo` clips are short, so long
  narration will outrun the picture.
- **Generation dominates the wall clock.** Save intermediate scenes (as above) so a
  failed assembly does not cost you another generation pass.

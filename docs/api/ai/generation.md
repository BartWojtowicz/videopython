# AI Generation

Generate videos, images, audio, and music from text prompts.

## Local Model Support

| Class | Local Model Family |
|-------|--------------------|
| TextToVideo | CogVideoX1.5-5B |
| ImageToVideo | CogVideoX1.5-5B-I2V |
| TextToSpeech | Chatterbox Multilingual |
| TextToMusic | MusicGen |
| TextToImage | SDXL |

## TextToVideo

::: videopython.ai.TextToVideo

## ImageToVideo

::: videopython.ai.ImageToVideo

## TextToImage

::: videopython.ai.TextToImage

## TextToSpeech

`generate_audio` accepts three optional Chatterbox `generate()` knobs —
`exaggeration`, `cfg_weight`, and `temperature` — for callers who want to
shape per-utterance prosody. Each defaults to `None`, which means "don't pass
the kwarg, let Chatterbox use its default". The dubbing pipeline derives them
per-segment from source vocals RMS via the
[`Expressiveness`](dubbing.md#expressiveness) dataclass.

```python
from videopython.ai import TextToSpeech

tts = TextToSpeech()

# Chatterbox defaults.
audio = tts.generate_audio("Welcome to videopython.")

# Dramatic delivery (higher exaggeration, lower cfg_weight slows pacing).
dramatic = tts.generate_audio(
    "We made it.",
    exaggeration=0.85,
    cfg_weight=0.35,
)
```

::: videopython.ai.TextToSpeech

## TextToMusic

::: videopython.ai.TextToMusic

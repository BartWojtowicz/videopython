# AI generation

Generate video, images, speech and music from text prompts. All local — see
[Local-only AI](../../explanation/local-ai.md) for the hardware each one needs.

| Class | Local model family |
|---|---|
| `TextToVideo` | Wan2.2-T2V-A14B |
| `ImageToVideo` | Wan2.2-I2V-A14B |
| `TextToImage` | Qwen-Image |
| `TextToSpeech` | Chatterbox Multilingual |
| `TextToMusic` | MusicGen |

A worked pipeline: [Assemble a video from AI-generated
media](../../how-to/ai-generated-video.md).

::: videopython.ai.TextToVideo

::: videopython.ai.ImageToVideo

::: videopython.ai.TextToImage

## TextToSpeech

`generate_audio` accepts three optional Chatterbox `generate()` knobs — `exaggeration`,
`cfg_weight` and `temperature`. Each defaults to `None`, which means "do not pass the
kwarg; let Chatterbox use its default". The dubbing pipeline derives them per segment from
source vocals RMS via [`Expressiveness`](dubbing.md#expressiveness).

```python
from videopython.ai import TextToSpeech

tts = TextToSpeech()
audio = tts.generate_audio("Welcome to videopython.")

dramatic = tts.generate_audio("We made it.", exaggeration=0.85, cfg_weight=0.35)
```

::: videopython.ai.TextToSpeech

::: videopython.ai.TextToMusic

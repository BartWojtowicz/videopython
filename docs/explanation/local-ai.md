# Local-only AI

`videopython.ai` has no cloud backend, no API-key configuration, and no hosted fallback.
Every model runs on your machine.

## The trade

What you get: no per-minute billing on a workload that is inherently long-running, no
media leaving your infrastructure, reproducible output that does not change when a
provider retires a model, and the ability to run the whole pipeline offline once weights
are cached.

What you pay: model weights download on first use and take real disk space; image and
video generation need a CUDA GPU and *raise* rather than falling back to CPU; and you
operate an [Ollama](https://ollama.com) server yourself for the LLM-backed features.

That trade only makes sense because of what videopython is for. Dubbing a two-hour
source or captioning a hundred scenes are long, bulk, repeatable jobs — exactly the shape
where per-call API pricing hurts most and where a local GPU amortizes well.

## What each capability runs

| Capability | Model family | Hardware |
|---|---|---|
| `TextToImage` | Qwen-Image | CUDA required |
| `TextToVideo` | Wan2.2-T2V-A14B | CUDA required |
| `ImageToVideo` | Wan2.2-I2V-A14B | CUDA required |
| `TextToSpeech` | Chatterbox Multilingual | CUDA or CPU |
| `TextToMusic` | MusicGen | CUDA, MPS, or CPU |
| `AudioToText` | Whisper (+ pyannote for diarization) | CPU, GPU optional |
| `AudioClassifier` | AST | CPU, GPU optional |
| `SemanticSceneDetector` | TransNetV2 | CPU, GPU optional |
| Face tracking | OpenCV YuNet | CPU |
| `ObjectDetector` | D-FINE (COCO) | CPU, GPU optional |
| `SceneVLM`, translation, planning | Any Ollama vision model | Ollama server |

## Why Ollama, and where it is required

Three features need a general-purpose LLM rather than a task-specific model: scene
captioning (`SceneVLM`, and therefore `VideoAnalyzer`), dubbing translation, and edit
planning (`AutoEditor` and the MCP server's captioning step).

Rather than bundle a particular LLM runtime and its weights, videopython talks to a local
Ollama server. You choose the model and the hardware it runs on; the library only needs
two guarantees from it: **vision capability** where keyframes are involved, and support
for Ollama's structured-output `format`, which is what makes the model return schema-valid
JSON instead of prose.

That second requirement is the one that bites. Some builds — certain MLX vision models,
for example — accept images but ignore `format`. They fail with prose where JSON was
expected. If a planner or captioner returns prose, change the model tag.

`qwen3.6:27b` is the default across all three features: Apache-2.0, vision-capable, and
honors `format`.

There is deliberately **no in-process fallback**. A silent degradation to a weaker path
would produce plausible-looking captions and translations that quietly got worse, which is
the worst failure mode for a pipeline whose output you are going to publish.

## Licensing is a selection criterion

Default model choices are permissively licensed (Apache-2.0 or equivalent) rather than
best-on-benchmark. A library that shipped an AGPL or research-only default would make
every commercial user audit their dependency graph before shipping. Where you want a
different trade, every model is swappable through a constructor argument.

## Managing memory

Pipelines that chain several models — dubbing runs Whisper, Demucs, a translator, and
Chatterbox — keep them all resident by default, which is fastest but expensive.
`low_memory=True` releases each stage's model after it runs, trading some latency for a
much lower ceiling. `SceneVLM.unload()` exists for the same parity on the Ollama side.

For long sources, combine that with the path-based APIs that never load frames — see
[Process hour-long videos](../how-to/long-videos.md).

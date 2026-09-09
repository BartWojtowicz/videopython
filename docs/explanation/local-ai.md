# Local AI

`videopython.ai` has no hosted inference backend, no API-key configuration, and no
hosted fallback. Task-specific models run in the videopython process. LLM-backed
features use an Ollama service that you operate.

## The trade

What you get: no per-minute inference billing on a workload that is inherently
long-running, control over media processing and model versions, and offline execution
once weights are cached.

What you pay: model weights download on first use and take real disk space; image and
video generation need a CUDA GPU and *raise* rather than falling back to CPU; and you
operate an [Ollama](https://ollama.com) server yourself for the LLM-backed features.

That trade only makes sense because of what videopython is for. Dubbing a two-hour
source or captioning a hundred scenes are long, bulk, repeatable jobs — exactly the shape
where per-call API pricing hurts most and where a local GPU amortizes well.

Model families are listed in [AI generation](../reference/ai/generation.md) and
[AI understanding](../reference/ai/understanding.md). Hardware and installation
requirements are in [Install](../install.md#hardware).

## Why Ollama, and where it is required

Three features need a general-purpose LLM rather than a task-specific model: scene
captioning (`SceneVLM`, and therefore `VideoAnalyzer`), dubbing translation, and edit
planning (`AutoEditor` and the MCP server's captioning step).

Rather than bundle a particular LLM runtime and its weights, videopython talks to the
configured Ollama server. You choose the model, host, and hardware; the library only
needs two guarantees from it: **vision capability** where keyframes are involved, and
support for Ollama's structured-output `format`, which makes the model return
schema-valid JSON instead of prose.

The normal setup keeps Ollama on the same machine. If `OLLAMA_HOST` points to another
machine, relevant prompts and images are sent to that host.

That second requirement is the one that bites. Some builds — certain MLX vision models,
for example — accept images but ignore `format`. They fail with prose where JSON was
expected. If a planner or captioner returns prose, change the model tag.

The default model tag and setup commands are in [Install](../install.md#3-ollama-only-for-llm-backed-features).

There is deliberately **no in-process fallback**. A silent degradation to a weaker path
would produce plausible-looking captions and translations that quietly got worse, which is
the worst failure mode for a pipeline whose output you are going to publish.

## Managing memory

Pipelines that chain several models — dubbing runs Whisper, Demucs, a translator, and
Chatterbox — keep them all resident by default, which is fastest but expensive.
`low_memory=True` releases each dubbing stage's model after use, including an explicit
release request for the translator on Ollama. `SceneVLM.unload()` clears its local
client; caption-model residency follows the Ollama server policy.

For long sources, combine that with the path-based APIs that never load frames — see
[Process hour-long videos](../how-to/long-videos.md).

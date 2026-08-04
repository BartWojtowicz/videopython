# Install

## 1. FFmpeg

videopython shells out to FFmpeg for every decode and encode, so install it first:

```bash
brew install ffmpeg                 # macOS
sudo apt-get install ffmpeg         # Ubuntu / Debian
choco install ffmpeg                # Windows (Chocolatey)
```

Burned-in subtitles (`add_subtitles`) need an FFmpeg built with libass — the packages
above all include it.

## 2. The package

```bash
pip install videopython             # core editing, no ML dependencies
pip install "videopython[ai]"       # + every AI capability
pip install "videopython[ai,mcp]"   # + the videopython-mcp server

uv add videopython                  # or with uv
uv add videopython --extra ai
```

Python `>=3.11, <3.14`.

`[ai]` is the single AI extra: transcription, diarization, detection, scene and VLM
understanding, source separation, translation, TTS, media generation, dubbing, and the
LLM auto-editing planner. The heavy ML dependencies load lazily at first use, so
`import videopython` stays fast even with `[ai]` installed.

`[mcp]` adds the `videopython-mcp` console script, a stdio
[Model Context Protocol](https://modelcontextprotocol.io) server. It needs `[ai]` too —
see [Drive editing from an MCP agent](how-to/mcp-server.md).

## 3. Ollama (only for LLM-backed features)

Scene captioning (`SceneVLM`, and therefore `VideoAnalyzer`), dubbing translation, and
the `AutoEditor` / MCP planner all call a local [Ollama](https://ollama.com) server.
There is no in-process fallback.

```bash
ollama serve                # start the local daemon
ollama pull qwen3.6:27b     # the default vision / translation model
```

The model must be vision-capable and must support Ollama's structured-output `format`.
The default `qwen3.6:27b` is Apache-2.0. Generation, transcription, detection, and audio
classification do **not** need Ollama.

## Hardware

| Capability | Requirement |
|---|---|
| Core editing | CPU only |
| `TextToImage`, `TextToVideo`, `ImageToVideo` | **NVIDIA CUDA GPU** — these ~20–28B models raise on CPU/MPS rather than falling back. A40 or better recommended for video |
| `TextToMusic` | CUDA, Apple MPS, or CPU |
| `TextToSpeech`, dubbing | CUDA or CPU |
| Transcription, detection, scene understanding | CPU (GPU optional) |

Model weights download on first use. Where a class accepts `device=`, use it to force
placement:

```python
from videopython.ai import TextToSpeech

tts = TextToSpeech(device="cuda")
```

For long or memory-constrained runs, see
[Process hour-long videos](how-to/long-videos.md).

## Notes on two dependencies

!!! note "TTS comes from a fork"
    `[ai]` installs
    [`videopython-chatterbox`](https://pypi.org/project/videopython-chatterbox/) rather
    than `chatterbox-tts`. Upstream pins `torch==2.6.0`, `diffusers==0.29.0` and
    `transformers==5.2.0` with `==`, which cannot be satisfied alongside the rest of
    `[ai]` (`pyannote-audio` alone requires `torch>=2.8`). The fork is upstream's source
    with corrected metadata; the import name is still `chatterbox`, so nothing in your
    code changes.

    Both distributions install a top-level `chatterbox` package — never install
    `chatterbox-tts` alongside `[ai]`.

!!! note "Dubbing TTS is pluggable"
    The dubbing pipeline synthesizes with the local Chatterbox `TextToSpeech` by
    default. Inject your own `SpeechBackend` into `VideoDubber` to run synthesis out of
    process and keep chatterbox out of your environment entirely — see
    [Dub a video](how-to/dubbing.md#swap-the-tts-backend).

## Verify the install

```python
from videopython.base import Video, VideoMetadata

print(VideoMetadata.from_path("some_video.mp4"))
```

If that prints resolution, fps and duration, FFmpeg and videopython are wired up
correctly. Continue with [Tutorial 1: your first edit](tutorials/first-edit.md).

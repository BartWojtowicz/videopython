# Install

## 1. FFmpeg

videopython shells out to FFmpeg for every decode and encode, so install it first:

```bash
brew install ffmpeg-full            # macOS
sudo apt-get install ffmpeg         # Ubuntu / Debian
choco install ffmpeg                # Windows (Chocolatey)
```

Homebrew installs `ffmpeg-full` as keg-only. Add its binaries to `PATH`:

```bash
export PATH="$(brew --prefix ffmpeg-full)/bin:$PATH"
```

Burned-in subtitles (`add_subtitles`) need an FFmpeg built with libass. The regular
Homebrew `ffmpeg` formula does not include libass; `ffmpeg-full` does.

## 2. The package

```bash
pip install videopython             # core editing, no ML dependencies
pip install "videopython[ai]"       # + every AI capability
pip install "videopython[mcp]"      # + the focused videopython-mcp stack

uv add videopython                  # or with uv
uv add videopython --extra ai
uv add videopython --extra mcp
```

## Supported environments

Videopython supports CPython 3.11, 3.12, 3.13, and 3.14 on Ubuntu, macOS, and Windows.
The full test suite runs on Ubuntu for every supported Python version. Clean-wheel
rendering and MCP handshake checks run on all three operating systems.

`ffmpeg` and `ffprobe` must be available on `PATH`. FFmpeg must provide the `libx264`
and AAC encoders plus the `xfade` and `acrossfade` filters. Burned-in subtitles also
need the `subtitles` filter from a build with libass.

`[ai]` is the single AI extra: transcription, diarization, detection, scene and VLM
understanding, source separation, translation, TTS, media generation, dubbing, and the
LLM auto-editing planner. The heavy ML dependencies load lazily at first use, so
`import videopython` stays fast even with `[ai]` installed.

`[mcp]` installs the `videopython-mcp` stdio server and the focused local stack it uses:
transcription, scene detection and captioning, face and object operations, and Ollama.
It does not install generation, dubbing, diarization, VAD, source separation, or TTS.
Install `[ai,mcp]` only when the same environment also needs those capabilities. See
[Drive editing from an MCP agent](how-to/mcp-server.md). Before connecting an agent,
review the [MCP security boundary](explanation/mcp-security.md).

## 3. Ollama (only for LLM-backed features)

Scene captioning (`SceneVLM`, enabled by default in `VideoAnalyzer`), dubbing
translation, and the `AutoEditor` planner call a configured [Ollama](https://ollama.com)
server. MCP uses Ollama for captioning; the connected agent supplies its own planner.
There is no in-process fallback.

```bash
ollama serve                # start the local daemon
ollama pull qwen3.6:27b     # the default vision / translation model
```

Captioning and planning require a vision model with structured-output `format` support.
Translation needs text and structured output only. The default is `qwen3.6:27b`.
It is a large model: the MCP workflow is not a
lightweight install, and the Ollama host must have enough memory to run it. Generation,
transcription, detection, and audio classification do **not** need Ollama.

## Hardware

| Capability | Requirement |
|---|---|
| Core editing | CPU only |
| MCP agent editing | CPU for media analysis; Ollama host capable of running `qwen3.6:27b` |
| `TextToImage`, `TextToVideo`, `ImageToVideo` | NVIDIA CUDA GPU; these models reject CPU/MPS generation. See the [tested environment](reference/verification.md#earlier-full-ai-verification) |
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

## Speech synthesis dependencies

!!! note "TTS comes from a fork"
    `[ai]` installs
    [`videopython-chatterbox`](https://pypi.org/project/videopython-chatterbox/) rather
    than `chatterbox-tts`. The fork has corrected dependency metadata and a
    short-text alignment fix. The import name remains `chatterbox`.

    Both distributions install a top-level `chatterbox` package — never install
    `chatterbox-tts` alongside `[ai]`.

!!! note "Dubbing TTS is pluggable"
    The dubbing pipeline synthesizes with the local Chatterbox `TextToSpeech` by
    default. Inject a backend object into `VideoDubber` to run synthesis out of
    process without loading Chatterbox in the videopython process — see
    [Dub a video](how-to/dubbing.md#swap-the-tts-backend).

## Verify the install

```python
from videopython.base import VideoMetadata

print(VideoMetadata.from_path("some_video.mp4"))
```

If that prints resolution, fps and duration, FFmpeg and videopython are wired up
correctly. Continue with [Tutorial 1: your first edit](tutorials/first-edit.md).

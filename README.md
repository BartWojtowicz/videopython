# videopython

[![PyPI](https://img.shields.io/pypi/v/videopython)](https://pypi.org/project/videopython/)
[![Python](https://img.shields.io/pypi/pyversions/videopython)](https://pypi.org/project/videopython/)
[![License](https://img.shields.io/github/license/BartWojtowicz/videopython)](LICENSE)
[![CI](https://github.com/BartWojtowicz/videopython/actions/workflows/ci.yml/badge.svg)](https://github.com/BartWojtowicz/videopython/actions/workflows/ci.yml)

Structured, local-first video editing for Python and AI agents.

Videopython represents an edit as a validated Python model or JSON plan. Whether the
plan comes from your code, an LLM, or an MCP client, it renders through the same
bounded-memory streaming engine.

[Documentation](https://videopython.com) ·
[First edit](https://videopython.com/tutorials/first-edit/) ·
[API reference](https://videopython.com/reference/) ·
[Roadmap](ROADMAP.md)

## Why videopython?

- **Structured edits** — segments and operations are Pydantic models with a generated
  JSON Schema.
- **Predictable rendering** — validate dimensions, timing, and operation constraints
  before decoding frames.
- **Bounded memory** — stream decode, effects, and encode without loading the full
  source into memory.
- **Local AI** — add transcription, scene understanding, generation, dubbing, and
  automatic editing without cloud inference APIs.
- **Agent-ready tools** — expose analysis, planning, validation, and rendering through
  the included MCP server.

## Installation

Install [FFmpeg](https://ffmpeg.org/download.html), then choose the package extras you
need:

```bash
pip install videopython              # core video and audio editing
pip install "videopython[ai]"        # all local AI features
pip install "videopython[mcp]"       # MCP and its focused analysis stack
```

Videopython supports Python `>=3.11, <3.15`. The `ai` and `mcp` extras are independent;
install `videopython[ai,mcp]` if you need both. See the
[installation guide](https://videopython.com/install/) for FFmpeg features, model
downloads, Ollama setup, and hardware requirements.

## Quick start

Describe the edit, validate it without loading frames, then render it:

```python
from videopython.editing import VideoEdit

edit = VideoEdit.from_dict({
    "segments": [
        {
            "source": "input.mp4",
            "start": 10.0,
            "end": 20.0,
            "operations": [
                {"op": "resize", "width": 1080, "height": 1920},
                {"op": "color_adjust", "saturation": 1.15, "contrast": 1.05},
                {"op": "fade", "mode": "in", "duration": 0.5},
            ],
        }
    ]
})

edit.validate()
edit.run_to_file("output.mp4")
```

`run_to_file()` streams the source through FFmpeg and the operation pipeline, so memory
use stays bounded for long videos. Continue with
[Your first edit](https://videopython.com/tutorials/first-edit/).

## What you can build

| Area | Capabilities | Start here |
|---|---|---|
| Editing | Cuts, transforms, effects, overlays, subtitles, audio, and multi-segment plans | [Editing guides](https://videopython.com/how-to/) |
| AI workflows | Transcription, detection, scene understanding, generation, dubbing, and automatic editing | [Local AI](https://videopython.com/explanation/local-ai/) |
| LLM integrations | Generated schemas, structured validation, repair, and dimension normalization | [LLM plan guide](https://videopython.com/how-to/llm-plans/) |
| MCP agents | Local tools for media analysis, planning, validation, and rendering | [MCP guide](https://videopython.com/how-to/mcp-server/) |

Core editing does not install PyTorch or other model runtimes. AI dependencies load only
when you use an AI feature.

## Documentation

The documentation follows [Diataxis](https://diataxis.fr/):

- [Tutorials](https://videopython.com/tutorials/) teach the library through complete
  examples.
- [How-to guides](https://videopython.com/how-to/) cover specific editing and AI tasks.
- [Reference](https://videopython.com/reference/) documents the API, operations, and
  JSON wire format.
- [Explanation](https://videopython.com/explanation/) covers the streaming engine,
  plan lifecycle, architecture, and LLM-first design.

## Project status

Videopython is pre-1.0, so public interfaces can still change. See the
[roadmap](ROADMAP.md) for the stability criteria and [release notes](RELEASE_NOTES.md)
for changes between versions.

For local setup, tests, documentation builds, and releases, see
[DEVELOPMENT.md](DEVELOPMENT.md).

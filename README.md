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
[API reference](https://videopython.com/reference/)

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

Install FFmpeg, then run `uv add videopython` (or `pip install videopython`).
The optional `ai` and `mcp` extras add model runtimes and the MCP server.
See the [installation guide](https://videopython.com/install/) for supported
Python versions, extras, FFmpeg features, and model setup.

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

## Scope

Videopython is a library for programmatic editing. It does not provide a hosted
inference service, an interactive editing application, universal support for model
runtimes and FFmpeg builds, or a second in-memory operation engine.

## Project status

Videopython is pre-1.0, so public interfaces can still change. See the
[compatibility policy](docs/reference/compatibility.md) for public contracts and
versioning rules, and [release notes](RELEASE_NOTES.md)
for changes between versions. Report vulnerabilities through the [security
policy](SECURITY.md).

For local setup, tests, documentation builds, and releases, see
[DEVELOPMENT.md](DEVELOPMENT.md).

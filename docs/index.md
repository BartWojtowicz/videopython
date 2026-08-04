---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# videopython

Minimal, LLM-friendly Python library for programmatic video editing, processing, and AI workflows.

<div class="hero-buttons">
  <a href="install/" class="btn-primary">Install</a>
  <a href="tutorials/first-edit/" class="btn-secondary">First edit in 10 minutes</a>
</div>

</div>

```python
from videopython.editing import VideoEdit

edit = VideoEdit.from_dict({
    "segments": [
        {"source": "intro.mp4", "start": 0, "end": 3,
         "operations": [{"op": "resize", "width": 1080, "height": 1920}]},
        {"source": "raw.mp4", "start": 10, "end": 25,
         "operations": [
             {"op": "resize", "width": 1080, "height": 1920},
             {"op": "resample_fps", "fps": 30},
             {"op": "fade", "mode": "in", "duration": 0.5},
         ]},
    ],
})
edit.run_to_file("output.mp4")
```

An edit is a plain data structure — a dict, or the JSON an LLM emits — validated before
any frame is touched and rendered by a streaming engine whose memory stays flat
regardless of source length.

## Where to go next

The documentation is split by what you are trying to do.

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### [Tutorials](tutorials/index.md)

Learn by doing. Start here if you are new: two short, guaranteed-to-work lessons that
take you from an installed package to a rendered video.

</div>

<div class="feature-card" markdown>

### [How-to guides](how-to/index.md)

Recipes for a specific goal — a vertical social clip, an hour-long source, a dubbed
track, an LLM or agent driving the edit.

</div>

<div class="feature-card" markdown>

### [Reference](reference/index.md)

The API surface: classes, operations, parameters, and the JSON wire format. Look things
up here, don't read it front to back.

</div>

<div class="feature-card" markdown>

### [Explanation](explanation/index.md)

Why the library is shaped the way it is — the streaming engine, the plan lifecycle, the
LLM-first design, and the local-only AI stack.

</div>

</div>

## What is in the box

- **Editing** — multi-segment plans with cuts, resize, crop, speed, freeze, silence
  removal, and ~25 effects (blur, color grading, Ken Burns, fades, overlays, subtitles).
- **Local AI** — generate images, video, speech and music; transcribe with diarization;
  detect scenes, faces and objects; caption shots with a vision model. No cloud API keys.
- **LLM control** — every operation is a Pydantic model, so the JSON Schema *is* the
  tool schema. Plans validate, self-repair, and normalize before they render.
- **Agent control** — `videopython-mcp` exposes the whole auto-edit pipeline as
  [Model Context Protocol](https://modelcontextprotocol.io) tools.

```bash
pip install videopython              # core editing
pip install "videopython[ai]"        # + all local AI features (GPU recommended)
pip install "videopython[ai,mcp]"    # + the MCP server
```

Python `>=3.11, <3.14`. See [Install](install.md) for FFmpeg, Ollama, and hardware notes.

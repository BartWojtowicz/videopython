---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# videopython

Minimal, LLM-friendly Python library for programmatic video editing, processing, and AI workflows.

<div class="hero-buttons">
  <a href="getting-started/quickstart/" class="btn-primary">Get Started</a>
  <a href="api/" class="btn-secondary">API Reference</a>
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

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### Core Editing

Cut, resize, crop, change speed, freeze frames, silence removal. Multi-segment editing plans with automatic fps/resolution matching for concatenation.

</div>

<div class="feature-card" markdown>

### Effects & Audio

Blur, zoom, color grading, vignette, Ken Burns, text and image overlay. Load, normalize, time-stretch, and mix audio tracks.

</div>

<div class="feature-card" markdown>

### LLM-Driven Editing

JSON editing plans with full JSON Schema generation, dry-run validation, and an operation registry with rich constraints. [Learn more →](guides/llm-integration.md)

</div>

<div class="feature-card" markdown>

### AI Generation

Generate images, video, speech, and music from text prompts. SDXL, CogVideoX, Chatterbox Multilingual TTS, MusicGen — all local, no API keys. [Learn more →](api/ai/generation.md)

</div>

<div class="feature-card" markdown>

### AI Video Analysis

Transcribe with speaker diarization, classify ambient audio, detect scene boundaries, and describe scenes with a vision-language model. One `VideoAnalyzer` runs the full pipeline and returns a serializable `VideoAnalysis`. [Learn more →](api/ai/video_analysis.md)

</div>

<div class="feature-card" markdown>

### AI Dubbing

Translate speech, clone the original voice, and re-time the dub onto the source — all in one pipeline. Whisper + MarianMT/Qwen3 + Chatterbox + Demucs. Source-prosody-conditioned expressiveness and a transcript-quality gate that rejects garbage input before paying for translation and TTS. [Learn more →](api/ai/dubbing.md)

</div>

</div>

## Installation

```bash
pip install videopython          # core editing
pip install "videopython[ai]"    # + ALL local AI features (GPU recommended)
```

`[ai]` installs everything; granular extras (`[asr]`, `[vision]`, `[separation]`, `[translation]`, `[tts]`, `[generation]`, `[dub]`) install only one capability each.

Python `>=3.11, <3.14`. AI features run locally -- no cloud API keys required.

See the [Installation Guide](getting-started/installation.md) for FFmpeg setup and details.

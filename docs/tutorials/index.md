# Tutorials

Two lessons for someone who has never used videopython. They are meant to be typed out
in order, top to bottom — every step is safe to run and every step produces something
you can play back.

Tutorials teach; they do not try to cover the whole API. When you want to accomplish a
particular task, switch to the [how-to guides](../how-to/index.md); when you want the
exact parameters of something, look it up in the [reference](../reference/index.md).

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### [1. Your first edit](first-edit.md)

Cut a clip, resize it, fade it in, and render it. Then add a second segment and let the
plan concatenate them. Core install only — no AI, no GPU.

</div>

<div class="feature-card" markdown>

### [2. Subtitle a video with AI](subtitles.md)

Transcribe speech with a local Whisper model and burn word-level subtitles onto the
video. Needs `pip install "videopython[ai]"`; runs on CPU.

</div>

</div>

## What you need

- videopython installed, with FFmpeg on your PATH — see [Install](../install.md).
- One `.mp4` file to work with. Any video will do; free clips are available from
  [Pexels](https://www.pexels.com/videos/) if you don't have one. Tutorial 2 needs a
  video with speech in it.

# How-to guides

Task-oriented recipes. Each one assumes you already know the basics from the
[tutorials](../tutorials/index.md) and gets straight to the goal.

## Editing

- [Build styled excerpts and summaries](social-clip.md) — selected footage with
  captions, a logo and title, or a ducked music bed.
- [Process hour-long videos](long-videos.md) — bounded frame buffers, frame
  iteration, and the memory-efficient dubbing path.

## AI

- [Assemble a video from AI-generated media](ai-generated-video.md) — text to image to
  video, with narration and crossfades.
- [Dub a video into another language](dubbing.md) — translate, clone the voice, and
  re-time it onto the source.
- [Update dubbing timing consumers](update-dubbing.md) — migrate timing calls and saved results to 0.61.2.

Subtitling is covered end to end in
[Tutorial 2](../tutorials/subtitles.md).

## Putting an LLM in the loop

There are three ways, and they differ in *who owns the model*:

| You want… | Use |
|---|---|
| videopython to edit for you, fully local | [Let a local LLM edit for you](auto-editing.md) |
| your own agent/harness to drive the tools | [Drive editing from an MCP agent](mcp-server.md) |
| to author and validate plans from your own LLM integration | [Author edit plans with your own LLM](llm-plans.md) |

To inspect and replay a complete JSON-plan example without a model, see
[Reproduce the agent-authored edit demo](agent-edit-demo.md).

The design behind all three is described in
[LLM-first design](../explanation/llm-first-design.md).

## Verification

- [Verify local AI models](verify-models.md) — run the repository harnesses and record model checks.

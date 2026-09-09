# Reproduce the agent-authored edit demo

This demo turns a landscape clip using the dog demo's edit plan into an eight-second vertical cut.
It is a deterministic replay of the JSON plan an LLM or MCP agent can author, so it
does not need an API key, a model, or the `[ai]` extra.

<video controls muted loop playsinline width="512">
  <source src="../../assets/agent-edit-demo.mp4" type="video/mp4">
</video>

## Run it

Clone the repository, install FFmpeg, and provide a landscape video at least ten
seconds long and at least 400×500 pixels. Run:

```bash
uv sync
uv run python examples/agent_edit_demo.py --source input.mp4 --output demo.mp4
```

The original dog source is a local test fixture and is not included in a clean
checkout. `--source` replaces that default; `--output` selects the rendered file.

## Give an agent the same brief

> Turn the dog clip into a vertical social shot. Use the expressive middle section,
> increase the color slightly, add a restrained punch-in, and fade at both ends.

The replay uses this plan:

```json
{
  "segments": [
    {
      "source": "src/tests/test_data/big_video.mp4",
      "start": 2.0,
      "end": 10.0,
      "operations": [
        {"op": "crop", "width": 400, "height": 500},
        {"op": "resize", "width": 512, "height": 640},
        {"op": "color_adjust", "contrast": 1.08, "saturation": 1.15},
        {"op": "punch_in", "zoom_factor": 1.04, "attack_frames": 12, "release_frames": 12},
        {"op": "fade", "mode": "in_out", "duration": 0.35}
      ]
    }
  ]
}
```

`VideoEdit.from_dict()` parses the same JSON shape exposed to an LLM. The script calls
`validate()` before rendering, then streams the edit to the output file. To let an agent
select scenes by catalog id instead of receiving exact time bounds, use the
[MCP workflow](mcp-server.md).

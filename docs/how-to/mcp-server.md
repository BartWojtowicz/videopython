# Drive editing from an MCP agent

`videopython-mcp` exposes the auto-editing pipeline as
[Model Context Protocol](https://modelcontextprotocol.io) tools, so an MCP-capable agent
edits with **its own model** as the planner. No in-process LLM.

## Set it up

```bash
pip install "videopython[ai,mcp]"
ollama serve              # scene captioning still uses a local vision model
ollama pull qwen3.6:27b
videopython-mcp           # stdio server
```

Register `videopython-mcp` with your MCP client (Claude Desktop, Claude Code, or any
other) as a **stdio** server. It takes no arguments.

## The flow the agent follows

1. `analyze_video(path)` for each source — scenes, transcript, captions, cached
   server-side.
2. `build_catalog()` — returns every candidate scene as JSON text, plus up to 12
   downscaled keyframes. Any omitted ids are named in a trailing note.
3. `scene_keyframes(scene_ids)` — pull the frames that were capped out, for a shortlist
   the agent picked from the catalog text.
4. Author an `EditPlan` against the `schema://videopython/edit-plan` resource, referencing
   scenes by `id`.
5. `validate_edit(plan)` → `run_edit(plan, output_path)`.

The tool signatures are in the [MCP reference](../reference/mcp.md).

The server is a thin wrapper over the same primitives as the programmatic
[`AutoEditor`](auto-editing.md) — the only difference is who owns the planning model. It
caches analyses and the catalog, so the agent passes small payloads (scene ids), never
whole analysis blobs.

## Keep long footage from flooding the context

Keyframes are the payload that grows with the footage, so the MCP path bounds it:

- every image it returns is downscaled to a longest side of ≤768 px, roughly 10× smaller
  than a full-resolution PNG;
- `build_catalog` inlines at most 12 of them and names the rest;
- the catalog **text** is always complete, so the agent can shortlist from captions and
  transcripts alone, then call `scene_keyframes` for the few frames it actually wants to
  look at.

A ~100-scene library therefore stays workable. Downscaling is scoped to MCP —
`SceneVLM` captioning and the local planner still see full-resolution frames.

## Speed up analysis when captions are all you need

```
analyze_video(path, profile="editing")
```

`profile="editing"` skips audio classification, which the catalog never reads. On long
sources that is a meaningful saving. `profile="full"` (the default) runs everything.

## Notes

- **`repair_edit` returns a concrete `VideoEdit`, not a re-submittable `EditPlan`.** Keep
  refining the by-id plan; use the repaired edit for inspection.
- **`run_edit` normalizes the output suffix to `.mp4`.**
- **`validate_edit` returns every problem at once**, so one round trip is usually enough
  to fix a plan.

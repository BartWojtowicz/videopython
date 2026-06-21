# MCP Server

`videopython` ships a [Model Context Protocol](https://modelcontextprotocol.io)
server that exposes the auto-editing pipeline as tools, so an MCP-capable agent
drives the edit with **its own model as the planner** (no in-process LLM).

## Install & run

```bash
pip install "videopython[ai,mcp]"
ollama serve         # scene captioning uses a local Ollama vision model
ollama pull gemma3:27b   # the default scene-VLM model
videopython-mcp      # stdio server
```

Register `videopython-mcp` with your MCP client (e.g. Claude Desktop) as a stdio
server.

## Tools

- **`analyze_video(path, profile="full")`** — analyze a source (scenes + transcript
  + captions); cached server-side for the catalog. Returns a short summary. Pass
  `profile="editing"` to skip audio classification — faster on long sources, since
  the catalog never reads it.
- **`build_catalog(sources=None)`** — returns the candidate scenes as one JSON text
  block (id, duration, shot_type, caption, transcript per scene — enough to
  shortlist from text alone), followed by up to 12 **downscaled** keyframe images.
  If there are more scenes, a trailing note names the omitted ids so the agent
  never assumes it saw every frame. Author the edit by referencing the returned
  `id` values.
- **`scene_keyframes(scene_ids)`** — fetch downscaled keyframe images for a chosen
  shortlist of scene ids. Use after `build_catalog` to pull the frames that were
  capped out, without re-inlining the whole library.
- **`validate_edit(plan)`** — validate an `EditPlan` (references catalog
  `scene_id`s); returns every problem at once as structured errors.
- **`repair_edit(plan)`** — clamp mechanical issues + normalize dimensions;
  returns the repaired `VideoEdit` + a changelog, for inspection. That `edit` is
  a concrete VideoEdit, not a re-submittable `EditPlan` — keep refining the by-id
  plan.
- **`run_edit(plan, output_path)`** — resolve, repair, validate, then render to an
  MP4 file (the suffix is normalized to `.mp4`), or return the remaining errors.

## Resource

- **`schema://videopython/edit-plan`** — the JSON Schema for the `EditPlan` the
  agent authors.

## Flow

`analyze_video` per source → `build_catalog` (read the scene JSON + the first
keyframes; pull any capped-out frames with `scene_keyframes`) → author an
`EditPlan` (scene ids + operations) against the schema resource → `validate_edit`
→ `run_edit`. The server is a thin wrapper over the same primitives as the
programmatic `AutoEditor`; the planner is the client's model. The server caches
analyses + the catalog so the agent passes small payloads (scene ids), not whole
analysis blobs.

## Scaling to many scenes

Keyframes are the payload that grows with the footage, so the server keeps it
bounded: every image the MCP path returns is downscaled (longest side ≤ 768px,
~10× smaller than a full-res PNG), and `build_catalog` inlines at most 12 of them,
naming the rest. The agent shortlists from the always-complete catalog *text*, then
calls `scene_keyframes(scene_ids)` for just the frames it wants to look at. This
keeps a ~100-scene library from flooding the model's context in one result.
(Downscaling is scoped to MCP — `SceneVLM` captioning and the local planner keep
full-resolution frames.)

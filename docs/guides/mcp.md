# MCP Server

`videopython` ships a [Model Context Protocol](https://modelcontextprotocol.io)
server that exposes the auto-editing pipeline as tools, so an MCP-capable agent
drives the edit with **its own model as the planner** (no in-process LLM).

## Install & run

```bash
pip install "videopython[ai,mcp]"
ollama serve         # scene captioning uses a local Ollama vision model
videopython-mcp      # stdio server
```

Register `videopython-mcp` with your MCP client (e.g. Claude Desktop) as a stdio
server.

## Tools

- **`analyze_video(path)`** — analyze a source (scenes + transcript + captions);
  cached server-side for the catalog. Returns a short summary.
- **`build_catalog(sources=None)`** — returns the candidate scenes (JSON) plus one
  keyframe image per scene, so the model can see the footage. Author the edit by
  referencing the returned `id` values.
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

`analyze_video` per source → `build_catalog` (see scenes + keyframes) → author an
`EditPlan` (scene ids + operations) against the schema resource → `validate_edit`
→ `run_edit`. The server is a thin wrapper over the same primitives as the
programmatic `AutoEditor`; the planner is the client's model. The server caches
analyses + the catalog so the agent passes small payloads (scene ids), not whole
analysis blobs.

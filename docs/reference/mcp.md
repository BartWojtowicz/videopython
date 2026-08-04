# MCP server

`videopython-mcp` — a stdio [Model Context Protocol](https://modelcontextprotocol.io)
server exposing the auto-editing pipeline. Install with the `[ai,mcp]` extras; setup and
the intended flow are in [Drive editing from an MCP agent](../how-to/mcp-server.md).

The server caches analyses and the catalog, so tool payloads stay small — the agent passes
scene ids, never analysis blobs.

## Tools

### `analyze_video(path, profile="full")`

Analyze a source: scenes, transcript, captions. Cached server-side for the catalog.
Returns a short summary. `profile="editing"` skips audio classification, which the
catalog never reads — faster on long sources.

### `build_catalog(sources=None)`

Returns the candidate scenes as one JSON text block — id, duration, shot_type, caption
and transcript per scene, enough to shortlist from text alone — followed by up to **12**
downscaled keyframe images. If more scenes exist, a trailing note names the omitted ids.
Author the edit by referencing the returned `id` values.

### `scene_keyframes(scene_ids)`

Downscaled keyframes for a chosen shortlist of scene ids. Use after `build_catalog` to
pull frames that were capped out, without re-inlining the whole library.

### `validate_edit(plan)`

Validate an `EditPlan` (which references catalog `scene_id`s). Returns every problem at
once as structured errors.

### `repair_edit(plan)`

Clamp mechanical issues and normalize dimensions. Returns the repaired `VideoEdit` plus a
changelog, **for inspection** — that edit is a concrete `VideoEdit`, not a re-submittable
`EditPlan`. Keep refining the by-id plan.

### `run_edit(plan, output_path)`

Resolve, repair, validate, then render to an MP4 (the suffix is normalized to `.mp4`), or
return the remaining errors.

## Resource

### `schema://videopython/edit-plan`

The JSON Schema for the `EditPlan` the agent authors. The plan types themselves are
documented in [AI auto-editing](ai/auto-edit.md).

## Image budget

Every image the MCP path returns is downscaled to a longest side of ≤768 px (~10× smaller
than a full-resolution PNG), and `build_catalog` inlines at most 12. Downscaling is scoped
to MCP — `SceneVLM` captioning and the in-process planner keep full-resolution frames.

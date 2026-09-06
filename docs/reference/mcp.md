# MCP server

`videopython-mcp` — a stdio [Model Context Protocol](https://modelcontextprotocol.io)
server exposing the auto-editing pipeline. Install with the `[mcp]` extra; setup and
the intended flow are in [Drive editing from an MCP agent](../how-to/mcp-server.md).

The server caches analyses and the catalog, so tool payloads stay small — the agent passes
scene ids, never analysis blobs.

## Tools

### `analyze_video(path, profile="editing")`

Analyze a source: scenes, transcript, captions. Cached server-side for the catalog.
Returns a short summary. The default `profile="editing"` skips audio classification,
which the catalog never reads. `profile="full"` runs all analyzers.

Returns `source`, `duration`, `fps`, `width`, `height`, `scenes`, and `analyzers`.
`analyzers` contains one record for each configured analysis stage:

| `status` | `reason` | Meaning |
|---|---|---|
| `completed` | `null` | The analyzer completed. |
| `skipped` | `disabled` | The selected profile disabled the analyzer. |
| `failed` | `initialization_failed` | The requested analyzer could not load. |
| `failed` | `execution_failed` | The requested analyzer loaded but did not complete. |

The remaining analysis is still cached when one analyzer fails. Check these records
before building a plan that depends on a missing transcript, caption, or face result.

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

```json
{"valid": false, "errors": [{"code": "unknown_scene_ids", "value": ["clip#9"], "message": "..."}]}
```

### `repair_edit(plan)`

Clamp mechanical issues and normalize dimensions. Returns the repaired `VideoEdit` plus a
changelog, **for inspection** — that edit is a concrete `VideoEdit`, not a re-submittable
`EditPlan`. Keep refining the by-id plan.

```json
{
  "edit": {
    "segments": [
      {
        "source": "clip.mp4",
        "start": 0.0,
        "end": 8.0,
        "operations": [],
        "transition_in": null
      }
    ],
    "post_operations": [],
    "match_to_lowest_fps": true,
    "match_to_lowest_resolution": true,
    "music_bed": null
  },
  "repairs": [
    {
      "location": "segments[0]",
      "field": "end",
      "old": 9.0,
      "new": 8.0,
      "code": "segment_end_exceeds_source"
    }
  ],
  "errors": []
}
```

`edit` is `null` when resolution fails. Each repair always has `location`, `field`,
`old`, `new`, and `code`.

### `run_edit(plan, output_path)`

Resolve, repair, validate, then render to an MP4 (the suffix is normalized to `.mp4`), or
return the remaining errors.

```json
{"output_path": "output.mp4", "errors": []}
```

`output_path` is `null` when the plan cannot be resolved or validated.

## Error objects

Every error has a stable `code` and a diagnostic `message`. Code-specific fields are:

- plan validation: `location`, `op`, `field`, `value`, `limit`, and `detail`;
- unknown scene ids: `value`, containing the unknown ids;
- invalid plan schema: `detail`, containing Pydantic error records.

Fields that do not apply are `null`. Error messages are for diagnostics; branch on
`code` and structured fields instead.

## Resource

### `schema://videopython/edit-plan`

The JSON Schema for the `EditPlan` the agent authors. The plan types themselves are
documented in [AI auto-editing](ai/auto-edit.md).

## Image budget

Every image the MCP path returns is downscaled to a longest side of ≤768 px (~10× smaller
than a full-resolution PNG), and `build_catalog` inlines at most 12. Downscaling is scoped
to MCP — `SceneVLM` captioning and the in-process planner keep full-resolution frames.

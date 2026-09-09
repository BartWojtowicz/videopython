# MCP server

`videopython-mcp` — a stdio [Model Context Protocol](https://modelcontextprotocol.io)
server exposing the auto-editing pipeline. Install with the `[mcp]` extra; setup and
the intended flow are in [Drive editing from an MCP agent](../how-to/mcp-server.md).

The server caches analyses and the catalog, so tool payloads stay small — the agent passes
scene ids, never analysis blobs. The server's filesystem, process, and network access is
defined in [MCP security boundary](../explanation/mcp-security.md).

## Tools

### `analyze_video(path, profile="editing")`

Analyze a source: scenes, transcript, captions. Cached server-side for the catalog.
Returns a short summary. The default `profile="editing"` skips audio classification,
which the catalog never reads. `profile="full"` runs all analyzers.

Returns `source` as a resolved absolute path, plus `duration`, `fps`, `width`,
`height`, `scenes`, and `analyzers`. A successful analysis clears the current catalog.
`analyzers` contains one record for each configured analysis stage:

| `status` | `reason` | Meaning |
|---|---|---|
| `completed` | `null` | The analyzer completed. |
| `skipped` | `disabled` | The selected profile disabled the analyzer. |
| `failed` | `initialization_failed` | The requested analyzer could not load. |
| `failed` | `execution_failed` | The requested analyzer loaded but did not complete. |

The remaining analysis is still cached when one analyzer fails. Check these records
before building a plan that depends on a missing transcript, caption, or face result.

### `export_analysis(source, output_path)`

Verify the selected source's content digest and write its cached `VideoAnalysis`
as JSON at `output_path`. `source` resolves to an absolute path. No inference runs.

### `import_analysis(path)`

Load saved JSON, reject unsupported formats or changed/unbound sources, then cache
it under its resolved source path. This clears the current catalog. Call
`build_catalog` before using scene IDs. Import does not start analyzers, change the
saved configuration, or fill unknown provenance from current models.

Both tools return `path`, `source`, `config`, `provenance`, and `analyzers`.
`path` and `source` are absolute paths. Failed and skipped stage outcomes are
preserved. Format, identity, and file-access errors are MCP tool errors; they do not
use edit-plan error codes. The [analysis reference](ai/video-analysis.md#saved-identity-and-migration)
defines provenance and migration.

### `build_catalog(sources=None, mode="visual", speech=None)`

Returns the candidate scenes as one JSON text block — id, duration, shot_type, caption
and transcript per scene, enough to shortlist from text alone — followed by up to **12**
downscaled keyframe images. If more scenes exist, a trailing note names the omitted ids.
Author the edit by referencing the returned `id` values.

For spoken passages, use `mode="speech"` and a `speech` object such as
`{"min_duration": 10, "max_duration": 30, "pause_duration": 0.8}`. Visual mode
requires `speech=null`. The [speech-candidate contract](ai/auto-edit.md#speech-candidates)
defines boundaries, missing-alignment behavior, and ID invalidation. If speech mode
finds no passages, the catalog has an empty `scenes` list and a following text block
explains that no complete aligned passages fit. Building any catalog clears the
previous selection and image cache.

### `scene_keyframes(scene_ids)`

Downscaled keyframes for a chosen shortlist of scene ids. Use after `build_catalog` to
pull frames that were capped out, without re-inlining the whole library.

### `scene_transcripts(scene_ids)`

Return a JSON text block mapping requested IDs to full normalized transcript text.
This works for visual scenes and speech passages. Duplicate IDs return one entry.
Unknown IDs return a text block with code `unknown_scene_ids`, as with
`scene_keyframes`. Requires a catalog; it performs no inference or media read.

### `validate_edit(plan)`

Validate an `EditPlan` (which references catalog `scene_id`s). Returns every problem at
once as structured errors.

```json
{
  "valid": false,
  "errors": [{
    "code": "unknown_scene_ids", "message": "Unknown scene ids: ['clip#9']",
    "value": ["clip#9"], "location": null, "op": null, "field": null,
    "limit": null, "detail": null
  }]
}
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

`edit` is `null` when resolution fails. A returned edit has been repaired, but
`errors=[]` here does not establish that it passes a new validation check. Each repair always has `location`, `field`,
`old`, `new`, and `code`.

### `run_edit(plan, output_path)`

Resolve, repair, validate, then render to an MP4 (the suffix is normalized to `.mp4`), or
return the remaining errors.

```json
{"output_path": "output.mp4", "errors": []}
```

`output_path` is `null` when the plan cannot be resolved or validated.

When the request includes `_meta.progressToken`, `run_edit` sends MCP progress
notifications during rendering. The numeric `progress` is a monotonically increasing
notification sequence; `total` is omitted. The `message` is JSON containing the
[`RenderProgress` fields](video-edit.md#render-progress). Counts inside the message
apply to one stage and can reset. Do not display the notification sequence as a
percentage. The tool's final result schema is unchanged.

Rendering runs in a worker thread, leaving the server event loop available to deliver
notifications. Progress uses the MCP transport, not prints to stdout. Clients that do
not request progress still receive the usual final result.

## Error objects

Errors from `validate_edit`, `repair_edit`, and `run_edit` have a stable `code` and a diagnostic `message`. Code-specific fields are:

- plan validation: `location`, `op`, `field`, `value`, `limit`, and `detail`;
- unknown scene ids: `value`, containing the unknown ids;
- invalid plan schema: `detail`, containing Pydantic error records.

Fields that do not apply are `null`. `scene_keyframes` instead returns a text block
with `code`, `value`, and `message` for unknown ids. Missing catalogs and failures
outside plan validation can surface as MCP tool errors. Error messages are for diagnostics; branch on
`code` and structured fields instead.

## Resource

### `schema://videopython/edit-plan`

The JSON Schema for the `EditPlan` the agent authors. The plan types themselves are
documented in [AI auto-editing](ai/auto-edit.md).

## Image budget

Every image the MCP path returns is downscaled to a longest side of 768 px, and
`build_catalog` extracts and inlines at most 12. Catalog text does not require image
extraction. Each image request batches scene midpoints by source and decodes only
through the last requested frame. The server retains at most 12 downscaled images,
evicting the least recently requested image when the cache is full. A cache hit needs
no decode. Building a new catalog clears the image cache. Omitted catalog rows do not
cause image extraction until requested.

`SceneVLM` captioning and the in-process planner keep full-resolution frames.
The [catalog measurements](verification.md#catalog-keyframe-extraction) record the
latency and memory tradeoff.

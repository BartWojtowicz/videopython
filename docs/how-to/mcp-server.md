# Drive editing from an MCP agent

`videopython-mcp` exposes the auto-editing pipeline as
[Model Context Protocol](https://modelcontextprotocol.io) tools, so an MCP-capable agent
edits with **its own model** as the planner. No in-process LLM.

## Set it up

```bash
pip install "videopython[mcp]"
ollama serve              # scene captioning still uses a local vision model
ollama pull qwen3.6:27b
videopython-mcp           # stdio server
```

Register `videopython-mcp` with your MCP client (Claude Desktop, Claude Code, or any
other) as a **stdio** server. It takes no arguments. The server can read and write with
the permissions of that client process. Review the [MCP security
boundary](../explanation/mcp-security.md) before connecting an agent.

## The flow the agent follows

1. `analyze_video(path)` for each source — scenes, transcript, captions, cached
   server-side. Inspect `analyzers` and retry or change the plan if a required analyzer
   failed. See the [status values](../reference/mcp.md#analyze_videopath-profileediting).
2. `build_catalog()` — returns every candidate scene as JSON text, plus a limited set of
   downscaled keyframes. Any omitted ids are named in a trailing note.
3. `scene_keyframes(scene_ids)` — pull the frames that were capped out, for a shortlist
   the agent picked from the catalog text. Use `scene_transcripts(scene_ids)` for full
   text before choosing spoken passages.
4. Author an `EditPlan` against the `schema://videopython/edit-plan` resource, referencing
   scenes by `id`.
5. `validate_edit(plan)` → `run_edit(plan, output_path)`.

For transcript-based cuts, build the catalog with `mode="speech"` and explicit
`speech` duration settings. This can select several passages from a single visual
shot. The tool signatures are in the [MCP reference](../reference/mcp.md).

The server is a thin wrapper over the same primitives as the programmatic
[`AutoEditor`](auto-editing.md) — the only difference is who owns the planning model. It
caches analyses and the catalog, so the agent passes small payloads (scene ids), never
whole analysis blobs.

## Keep long footage from flooding the context

Shortlist scenes from catalog text, then request keyframes for those ids. The server
caps the images included by `build_catalog`; see the
[image budget](../reference/mcp.md#image-budget) for limits. Catalog text is complete
and grows with the number of scenes.

## Run the full analysis profile

```
analyze_video(path, profile="full")
```

The default `profile="editing"` skips audio classification, which the catalog never
reads. Use `profile="full"` only when another consumer needs that result.

## Account for the local model

The MCP extra omits unrelated generation, dubbing, diarization, source separation, and
TTS dependencies. It also disables VAD to keep `torchaudio` out of the agent stack;
Whisper performs its standard language detection instead. Agent editing is still a
substantial local-AI workflow: it runs Whisper, TransNetV2, face analysis, and
`qwen3.6:27b` scene captioning. Confirm that the Ollama host can run the model before
analyzing long footage.

## Notes

- **`repair_edit` returns a concrete `VideoEdit`, not a re-submittable `EditPlan`.** Keep
  refining the by-id plan; use the repaired edit for inspection.
- **`run_edit` normalizes the output suffix to `.mp4`.**
- **`validate_edit` returns every problem at once**, so one round trip is usually enough
  to fix a plan.

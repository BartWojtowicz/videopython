# MCP security boundary

`videopython-mcp` is a local stdio server. It does not listen on a network port and it
does not authenticate its MCP client. The client that starts the process is trusted to
request media analysis and rendering with the permissions of that process.

The server is not a filesystem sandbox. It inherits the operating-system account,
working directory, environment variables, `PATH`, and network access of the MCP client
that starts it. It also starts `ffmpeg` and `ffprobe` subprocesses with those same
permissions.

## File reads and returned data

The server does not restrict paths to the current directory or to a configured root.
Relative paths resolve from the process working directory. Absolute paths and symbolic
links work when the operating-system account can read them.

| Tool or input | File access |
|---|---|
| `analyze_video(path)` | Reads the selected media, including its video, audio, and container metadata. |
| `build_catalog()` | Reads analyzed sources again to extract keyframes. |
| `scene_keyframes(scene_ids)` | Reads analyzed sources again for requested keyframes that are not cached. |
| Plan operations | Path-bearing operations can read assets such as overlay images. Source video paths come from scenes that were already analyzed. |
| `validate_edit()` and `repair_edit()` | Can probe source media and inspect referenced assets while checking a plan. |
| `run_edit()` | Reads the source media and referenced assets, then renders the result. |

Analysis results can contain transcripts, captions, file metadata, and keyframe images.
The connected MCP client receives this data. Do not connect a client that is not allowed
to see the source content.

Analyses and catalog text stay in process memory for the stdio session, along with
a bounded cache of downscaled keyframes. The [MCP reference](../reference/mcp.md#image-budget)
defines its limits. Rendering uses the
operating system's temporary directory for intermediate media. Normal completion removes
owned temporary files, but an abrupt process or machine failure can leave them behind.

## Output writes

`run_edit(plan, output_path)` accepts any path that the process can write. It:

- changes the suffix to `.mp4`;
- creates missing parent directories;
- overwrites an existing output file through FFmpeg;
- can remove an incomplete output if rendering fails.

There is no output-directory allowlist and no confirmation prompt. Treat `output_path`
as a trusted instruction from the connected client.

## Models and network access

The MCP workflow uses local model libraries and Ollama. Model weights can download on
first use as described in [Install](../install.md). These downloads use the process's
network access and write to the configured model caches.

Scene captioning sends prompts and sampled video frames to the configured Ollama host.
The normal setup uses a local Ollama service. If `OLLAMA_HOST` names another machine,
those frames leave the videopython process and are visible to that service.

## Deployment consequences

The practical security boundary is the operating-system account and its mounted files.
For a client or agent that you do not fully control:

- run the MCP server as a dedicated low-privilege user or in a container;
- mount only required source directories, preferably read-only;
- give the process a separate writable output directory;
- keep credentials and unrelated secrets out of its environment;
- keep Ollama on a trusted host;
- do not expose the stdio server through a network gateway unless that gateway adds
  authentication and path restrictions.

These controls belong to the MCP client or deployment environment. Videopython does not
add them inside the server because local clients have different storage and permission
models.

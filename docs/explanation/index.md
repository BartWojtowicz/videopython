# Explanation

Background on why videopython is shaped the way it is. Nothing here is needed to get
work done — it is here so that when the library behaves in a way that surprises you, the
behavior has a reason you can find.

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### [Architecture](architecture.md)

The library layers and MCP server, the dependency layering that keeps AI optional, and why importing
videopython stays fast with `[ai]` installed.

</div>

<div class="feature-card" markdown>

### [The streaming engine](streaming-engine.md)

Why `run_to_file()` is the only execution path, how an operation becomes either an FFmpeg
filter or a per-frame function, and which plan shapes cannot stream.

</div>

<div class="feature-card" markdown>

### [The plan lifecycle](plan-lifecycle.md)

Parse, validate, check, repair, normalize — what each stage owns, and why numeric bounds
are deliberately not enforced at parse time.

</div>

<div class="feature-card" markdown>

### [LLM-first design](llm-first-design.md)

Why every operation is a Pydantic model, what `llm_exposed` and `llm_hidden` are for, and
why the auto-editor makes the model select scenes by id.

</div>

<div class="feature-card" markdown>

### [Local AI](local-ai.md)

Why inference stays under your control, what that costs, and which parts depend on
Ollama.

</div>

<div class="feature-card" markdown>

### [The dubbing pipeline](dubbing.md)

Why source turns, translation phrases, and generated speech use different units.

</div>

<div class="feature-card" markdown>

### [MCP security boundary](mcp-security.md)

What the local MCP process can read, write, execute, and send to its client or Ollama
host.

</div>

</div>

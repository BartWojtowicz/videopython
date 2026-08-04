# Explanation

Background on why videopython is shaped the way it is. Nothing here is needed to get
work done — it is here so that when the library behaves in a way that surprises you, the
behavior has a reason you can find.

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

### [Architecture](architecture.md)

The four subpackages, the dependency layering that keeps AI optional, and why importing
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

### [Local-only AI](local-ai.md)

Why every model runs on your machine, what that costs, and which parts depend on a local
Ollama server.

</div>

</div>

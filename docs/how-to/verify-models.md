# Verify local AI models

Use this procedure when an AI integration, dependency, or default model changes.
The unit suite uses fakes; it cannot establish that real models load or produce
useful media. Run these checks from a repository checkout with representative
media and the hardware required by the selected features.

## Prepare the environment

Follow [Install](../install.md) for FFmpeg and Ollama, then install the development
and model dependencies:

```bash
uv sync --all-extras --group ai --frozen
uv run python scripts/verify_ai_models.py --help
```

Download the selected Ollama model before the run. The first harness run also
populates other model caches. Use a local model service; no paid APIs are needed.

## Run and measure

Run once to populate caches, then start a fresh process with a new output directory:

```bash
OLLAMA_HOST=127.0.0.1:11434 uv run python scripts/verify_ai_models.py \
  --all \
  --video verification-input/cam1_1min.mp4 \
  --workdir verify-results/measured
```

Supply your own representative input at `--video`; that media is not distributed
with the repository. Use `--only` to select the checks affected by your change.
Use the public API defaults for performance baselines. Label smaller-model or
reduced-setting runs as compatibility checks.

Record the commit, dependency versions, hardware, source hash, settings, and measured
scope. Include cached model loading, inference, output writing, semantic checks,
and model cleanup when reporting a full harness time. For matched comparisons,
keep input and generation settings fixed across both versions.

## Review the output

Inspect the generated media and the harness results. The dub check fails on missing
timing measurements or translation/synthesis failures. It reports excessive speeds
for listening review. Successful synthesis and matching word counts alone do not
prove meaning, intelligibility, or voice consistency.

Do not release an affected integration until its selected checks pass. Record
limitations alongside results in the [verification records](../reference/verification.md).
The release workflow itself does not enforce this manual check.

## Check an MCP session

The stdio harness exercises analysis, catalog construction, validation, and rendering:

```bash
OLLAMA_HOST=127.0.0.1:11434 uv run python scripts/verify_mcp_workflow.py \
  --source verification-input/cam1_1min.mp4 \
  --workdir verify-results/mcp \
  --vision-model qwen3.6:27b
```

Check analyzer outcomes and inspect the rendered output. Existing machine-specific
results are in [MCP workflow verification](../reference/verification.md#mcp-workflow-verification).

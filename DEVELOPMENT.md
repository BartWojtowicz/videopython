# Development

## Project structure

```
.
└── src
    ├── stubs       # mypy stubs for untyped third-party packages
    ├── tests       # Unit tests (mirrors the package tree)
    └── videopython # Library code
```

The `videopython` library is split into four subpackages, layered by dependency:

* `videopython.base` — `Video`, I/O primitives, shared result types. No AI imports.
* `videopython.audio` — `Audio` container and analysis. Depends on `base`.
* `videopython.editing` — `Operation`/`Effect` foundation and the `VideoEdit` plan runner. Depends on `base` and `audio`.
* `videopython.ai` — generation, understanding, dubbing, and AI-only transforms. Depends on `base`, `audio`, and optionally `editing`. Only this subpackage requires the `[ai]` extra.

The "no AI imports in `base`/`audio`/`editing`" invariant is enforced by `src/tests/test_import_isolation.py`.

Why the layering (and the lazy AI re-exports) look like this is written up for users in
[Architecture](https://videopython.com/explanation/architecture/) — update that page when
the structure changes.

## Running locally

We use [uv](https://docs.astral.sh/uv/) as project and package manager. Once you clone the repo and install uv:

```bash
uv sync
# Add the model runtimes only when you work on real AI integrations.
uv sync --all-extras --group ai
```

### Running tests

```bash
# Everything (this is what CI runs)
uv run pytest

# Just one area
uv run pytest src/tests/editing
uv run pytest src/tests/ai
```

There are no markers and no skipped tiers: **every test in the suite runs on a
GitHub runner** with the base and development dependencies — no GPU, AI extra,
or model downloads. The AI tests use lightweight fakes for the model runtimes;
small dependencies needed to test algorithms directly belong to the development
dependency group.

That means the suite cannot tell you whether a *model* works, only whether the code
around it does. A fake returns whatever the test handed it. A maintainer verifies
real-model behaviour with `scripts/verify_ai_models.py`.

To check a test really is runner-feasible, run it without the AI dependency group
and against an empty model cache:

```bash
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --isolated --no-group ai pytest src/tests/ai
```

### Verifying AI models

Before a release that changes an AI integration, dependency, or default model, run
`scripts/verify_ai_models.py` on a GPU machine with a representative video. Follow the
cache-warming and timing protocol in the [verification
records](docs/reference/verification.md). Do not release an applicable change until all
selected checks pass. Published performance baselines must use the public API defaults;
label reduced settings as compatibility checks. The dub check fails if its timing
summary is missing or if one segment loses more than 3.0 seconds during synchronization.

### Linting & type checking

[Pre-commit](https://pre-commit.com/) runs [Ruff](https://docs.astral.sh/ruff/) and [mypy](https://github.com/python/mypy) locally and in CI.

```bash
# Install git pre-commit hook
uv run pre-commit install

# Run all configured hooks manually
uv run pre-commit run --all-files

# Or run tools directly
uv run ruff format src
uv run ruff check src
uv run mypy src
```

mypy stubs for untyped third-party packages live in `src/stubs/`.

### Docs

The docs site (published at [videopython.com](https://videopython.com)) is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) from the `docs/` directory:

```bash
uv run mkdocs serve          # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict # render to ./site; fails on broken internal links
```

Run the `--strict` build before opening a docs PR — it is what catches a link to a page
or anchor that no longer exists.

#### Structure

`docs/` follows [Diátaxis](https://diataxis.fr). Every page belongs to exactly one of four
modes, and mixing them is the thing to avoid:

| Directory | Mode | Answers | Rule of thumb |
|---|---|---|---|
| `tutorials/` | Learning | "Teach me to use this" | Must work start to finish, no choices, no digressions |
| `how-to/` | Task | "How do I achieve X?" | Titled with a verb; assumes competence; links out instead of explaining |
| `reference/` | Information | "What are the parameters?" | Factual and dry; mkdocstrings blocks plus tables |
| `explanation/` | Understanding | "Why is it like this?" | Design rationale and trade-offs; no step-by-step |

Practical consequences when you add something:

* A new operation → a row in `reference/operations.md` plus its `:::` block on the
  matching reference page. Rationale, if any, goes in `explanation/`, not the table.
* A new capability → usually one `how-to/` page. Add a tutorial only if it is part of the
  first hour of using the library.
* Design decisions belong in `explanation/`, so reference pages stay skimmable. If you
  find yourself writing "because" on a reference page, move it.
* Renaming or moving a page → add an entry to `redirect_maps` in `mkdocs.yml`. The site is
  published and linked from PyPI, so URLs are part of the contract.

Docstrings are the source for reference content: mkdocstrings pulls them in Google style,
so a well-documented `Operation` needs almost nothing hand-written on the page.

## Dependencies

### The consumer-install invariant

`[ai]` must resolve for someone running plain `pip install "videopython[ai]"`.

`[tool.uv]` tables (`override-dependencies`, `constraint-dependencies`) are a uv
*workspace* feature — they do not ship in the built wheel. Anything reconciled only
there is invisible to consumers, so it makes CI green while every downstream install
fails. That is exactly what happened in 0.54.0.

There are deliberately no overrides today. If a dependency ships metadata that cannot
be satisfied, fix the metadata rather than patching it locally:

* upstream a fix, or
* fork the package and publish corrected metadata (what 0.54.1 did), or
* drop the dependency.

The `pip_resolve` CI job (`.github/workflows/pip-resolve.yml`) builds the wheel and
resolves the core, `[ai]`, and `[mcp]` dependency graphs with pip on every supported
Python version, on every push and weekly on a schedule. The schedule matters because
these breakages arrive from upstream releases tightening their pins, not from our own
commits.

The `platform_smoke` CI job installs the built wheel on Ubuntu, macOS, and Windows. It
checks the required FFmpeg capabilities, renders a short clip, imports the public
package layers, and performs an MCP tool and resource handshake.

### `videopython-chatterbox`

`[ai]` depends on
[`videopython-chatterbox`](https://github.com/BartWojtowicz/videopython-chatterbox),
our fork of `chatterbox-tts`, published to PyPI. It is upstream's source with
corrected dependency metadata — upstream pins `torch==2.6.0`, `diffusers==0.29.0`
and `transformers==5.2.0` with `==`, which cannot be satisfied alongside the rest of
`[ai]` (`pyannote-audio` alone needs `torch>=2.8`). The import name is still
`chatterbox`, so no application code changes.

Resync when upstream ships a release we want. Both distributions install a top-level
`chatterbox` package, so they must never be installed together.

## Releasing

To release a new version:
1. Update `version` in `pyproject.toml`
2. Add a new section in `RELEASE_NOTES.md` with the matching version (e.g., `## 0.7.0`)
3. Push to `main`

CI will validate that the versions match, run tests, create a GitHub release, and publish to PyPI.

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

## Running locally

We use [uv](https://docs.astral.sh/uv/) as project and package manager. Once you clone the repo and install uv:

```bash
uv sync --all-extras
```

### Running tests

```bash
# Non-AI tests (runs in CI)
uv run pytest --ignore=src/tests/ai

# AI tests, lightweight only (runs in CI)
uv run pytest src/tests/ai -m "not requires_model_download"

# AI tests, full suite (downloads model weights, run locally)
uv run pytest src/tests/ai
```

Heavy tests are marked `@pytest.mark.requires_model_download` and skipped in CI. The rest of the AI suite stays fast by monkey-patching the model classes with lightweight fakes.

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
uv run mkdocs serve   # live preview at http://127.0.0.1:8000
uv run mkdocs build   # render the static site to ./site
```

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
resolves `[ai]` and `[ai,mcp]` with pip in a clean venv, on every push and weekly on
a schedule. The schedule matters because these breakages arrive from upstream
releases tightening their pins, not from our own commits.

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

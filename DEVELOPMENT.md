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

## Releasing

To release a new version:
1. Update `version` in `pyproject.toml`
2. Add a new section in `RELEASE_NOTES.md` with the matching version (e.g., `## 0.7.0`)
3. Push to `main`

CI will validate that the versions match, run tests, create a GitHub release, and publish to PyPI.

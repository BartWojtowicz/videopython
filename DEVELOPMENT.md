# Development

## Project structure

```
.
└── src
    ├── stubs       # mypy stubs for untyped third-party packages
    ├── tests       # Unit tests (mirrors the package tree)
    └── videopython # Library code
```

Package boundaries and lazy imports are described in
[Architecture](docs/explanation/architecture.md). The dependency direction is
checked by `src/tests/test_import_isolation.py`.

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

The default suite runs on a GitHub runner with base and development dependencies.
It needs no GPU, AI extra, or model downloads. The AI tests use lightweight fakes for the model runtimes;
small dependencies needed to test algorithms directly belong to the development
dependency group. Tests must not call paid APIs.

That means the suite cannot tell you whether a *model* works, only whether the code
around it does. A fake returns whatever the test handed it. A maintainer verifies
real-model behaviour with `scripts/verify_ai_models.py`.

To check a test really is runner-feasible, run it without the AI dependency group
and against an empty model cache (Bash):

```bash
HF_HOME=$(mktemp -d) HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --isolated --no-group ai pytest src/tests/ai
```

### Verifying AI models

Before releasing a change to an AI integration, dependency, or default model, run
all affected real-model checks. Follow [Verify local AI models](docs/how-to/verify-models.md)
and record results in the [verification records](docs/reference/verification.md).
The release workflow does not enforce this manual check.

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

Run the strict build before opening a documentation PR. Link and anchor warnings
are errors with the repository's `mkdocs.yml` settings. The build cannot prove that
an example runs or that a statement matches the code; check those separately.

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
* Renaming or moving a page → update navigation, links, and anchors in the same change.

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
Python version, in main-branch and pull-request CI, and weekly on a schedule.
The scheduled check detects conflicts introduced by upstream dependency releases.

The `platform_smoke` CI job installs the built wheel on Ubuntu, macOS, and Windows. It
checks the required FFmpeg capabilities, renders a short clip, imports the public
package layers, and performs an MCP tool and resource handshake.

### `videopython-chatterbox`

`[ai]` depends on
[`videopython-chatterbox`](https://github.com/BartWojtowicz/videopython-chatterbox),
our fork of `chatterbox-tts`, published to PyPI. It corrects dependency metadata
that conflicts with the AI stack and fixes short-text alignment. The import name
is `chatterbox`. See the dependency declarations in `pyproject.toml` and the
[0.54.1 release notes](RELEASE_NOTES.md#0541) for the original resolver failure.

Resync when upstream ships a release we want. Both distributions install a top-level
`chatterbox` package, so they must never be installed together.

## Releasing

To release a new version:

1. Complete the applicable real-model verification described above.
2. Update `version` in `pyproject.toml` and refresh `uv.lock` with `uv lock`.
3. Add a matching version section at the top of `RELEASE_NOTES.md`.
4. Run tests, `uv run pre-commit run --all-files`, and the strict documentation build.
5. Review the diff and merge the release changes to `main` after push approval.

A push to `main` that changes `RELEASE_NOTES.md` starts `.github/workflows/publish.yml`.
It checks that the first release heading matches the package version. If that tag
already exists, it skips publication. Otherwise it runs CI, creates a GitHub release,
and publishes to PyPI. Documentation deploys separately on pushes to `main`.

# Development

## Project structure

Source code of the project can be found under `src/` directory, along with separate directories for unit tests and mypy stubs.
```
.
└── src
    ├── stubs # Contains stubs for mypy
    ├── tests # Unit tests
    └── videopython # Library code
```

----

The `videopython` library is divided into four top-level subpackages:
* `videopython.base`: Data containers and I/O primitives — `Video`, `VideoMetadata`, `FrameIterator`, `ImageText`, `Transcription`, and the shared result types (`BoundingBox`, `DetectedFace`, `SceneBoundary`, ...). No editing logic; no AI imports.
* `videopython.audio`: `Audio` data container plus audio analysis (`AudioLevels`, `SilentSegment`, segment classification). Depends on `base`.
* `videopython.editing`: All editing primitives (`Operation`, `Effect`, transforms, effects, `TranscriptionOverlay`) and the plan runner (`VideoEdit`, `SegmentConfig`). Depends on `base` and `audio`.
* `videopython.ai`: AI-powered generation, understanding, dubbing, and AI-only transforms. Has its own `ai` extra with the model dependencies. Depends on `base`, `audio`, and optionally `editing`.

Only `videopython.ai` requires the `[ai]` extra; the other three install with the default `pip install videopython` and contain no AI imports (enforced by `src/tests/test_import_isolation.py`).

## Running locally

We are using [uv](https://docs.astral.sh/uv/) as project and package manager. Once you clone the repo and install uv locally, you can use it to sync the dependencies.
```bash
uv sync --all-extras
```

### Running tests

Tests mirror the package tree:

- `src/tests/base/` - Tests for `videopython.base`
- `src/tests/audio/` - Tests for `videopython.audio`
- `src/tests/editing/` - Tests for `videopython.editing`
- `src/tests/ai/` - Tests for `videopython.ai` (requires AI extras)
- `src/tests/test_import_isolation.py` - Cross-subpackage check that `base`, `audio`, and `editing` import without pulling in any AI dependency

```bash
# Non-AI tests (runs in CI)
uv run pytest --ignore=src/tests/ai

# AI tests - all (requires model downloads, run locally)
uv run pytest src/tests/ai

# AI tests - lightweight only (runs in CI)
uv run pytest src/tests/ai -m "not requires_model_download"
```

### AI test markers

Tests requiring models 100MB+ are marked with `@pytest.mark.requires_model_download` and excluded from CI. The bulk of the AI test surface relies on monkey-patched fakes (see `_FakeSceneVLM`, `_FakeFaceTracker`, etc. in `src/tests/ai/test_video_analysis_scene_first.py`) so it stays fast and CI-friendly.

`requires_model_download` is configured in `pyproject.toml` under `[tool.pytest.ini_options].markers` and is respected by `conftest.py`.

We use [pre-commit](https://pre-commit.com/) to run [Ruff](https://docs.astral.sh/ruff/) and [mypy](https://github.com/python/mypy) checks locally, and CI runs the same hooks.
```bash
# Install git pre-commit hook
uv run pre-commit install

# Run all configured hooks manually
uv run pre-commit run --all-files

# (Optional) run tools directly
uv run ruff format src
uv run ruff check src
uv run mypy src
```

## Releasing

To release a new version:
1. Update `version` in `pyproject.toml`
2. Add a new section in `RELEASE_NOTES.md` with matching version (e.g., `## 0.7.0`)
3. Push to `main`

CI will validate that versions match, run tests, create a GitHub release, and publish to PyPI.

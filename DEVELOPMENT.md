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

The `videopython` library is divided into 2 separate high-level modules:
* `videopython.base`: Contains base classes for handling videos and for basic video editing. There are no imports from `videopython.ai` within the `base` module, which allows users to install light-weight base dependencies to do simple video operations.
* `videopython.ai`: Contains AI-powered functionalities for video generation. It has its own `ai` dependency group, which contains all dependencies required to run AI models.

## Running locally

We are using [uv](https://docs.astral.sh/uv/) as project and package manager. Once you clone the repo and install uv locally, you can use it to sync the dependencies.
```bash
uv sync --all-extras
```

### Running tests

Tests are organized into two directories:

- `src/tests/base/` - Tests for `videopython.base` module (no AI dependencies)
- `src/tests/ai/` - Tests for `videopython.ai` module (requires AI extras)

```bash
# Base tests (runs in CI)
uv run pytest src/tests/base

# AI tests - all (requires model downloads, run locally)
uv run pytest src/tests/ai

# AI tests - lightweight only (runs in CI)
uv run pytest src/tests/ai -m "not requires_model_download"
```

### AI test markers

Tests requiring models 100MB+ are marked with `@pytest.mark.requires_model_download` and excluded from CI:

| Test Class | Model | Size | Runs in CI |
|------------|-------|------|------------|
| `TestObjectDetectorLocal` | YOLO | ~6MB | Yes |
| `TestTextDetectorLocal` | EasyOCR | ~100MB+ | No (marked) |
| `TestAudioClassifier` | PANNs | ~80MB | Yes |
| `TestFaceDetector` | OpenCV cascade | bundled | Yes |
| `TestCameraMotionDetector` | OpenCV optical flow | bundled | Yes |
| `test_transforms.py` | mocked | none | Yes |

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

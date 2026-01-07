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

To run the unit tests:
```bash
uv run pytest src/tests/base  # Base tests (no AI, runs in CI)
uv run pytest src/tests/ai    # AI tests (requires ai extras, excluded from CI)
```

We also use [Ruff](https://docs.astral.sh/ruff/) for linting/formatting and [mypy](https://github.com/python/mypy) as type checker.
```bash
# Run formatting
uv run ruff format
# Run linting and apply fixes
uv run ruff check --fix
# Run type checks
uv run mypy src/
```

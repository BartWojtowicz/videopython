import time
import uuid
from pathlib import Path
from typing import Callable


def generate_random_name(suffix=".mp4"):
    """Generates random name."""
    return f"{uuid.uuid4()}{suffix}"


def timeit(func: Callable):
    """Decorator to measure execution time of a function."""

    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Execution time: {end - start:.3f} seconds.")
        return result

    return timed


def check_path(path: str, dir_exists: bool = True, suffix: str | None = None) -> str:
    fullpath = Path(path).resolve()
    if dir_exists and not fullpath.parent.exists():
        raise ValueError(f"Directory `{fullpath.parent}` does not exist!")
    if suffix and suffix != fullpath.suffix:
        raise ValueError(f"Required suffix `{suffix}` does not match the file suffix `{fullpath.suffix}`")
    return str(fullpath)

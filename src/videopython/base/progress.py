from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypeVar

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback if tqdm isn't available
    tqdm = None  # type: ignore

__all__ = ["configure", "set_verbose", "set_progress", "get_config", "log", "progress_iter"]

T = TypeVar("T")


@dataclass
class _BaseConfig:
    verbose: bool = False
    progress: bool = False


_CONFIG = _BaseConfig()


def configure(*, verbose: bool | None = None, progress: bool | None = None) -> None:
    """Configure base module logging and progress behavior."""
    if verbose is not None:
        _CONFIG.verbose = bool(verbose)
    if progress is not None:
        _CONFIG.progress = bool(progress)


def set_verbose(value: bool) -> None:
    """Enable or disable verbose logging in base operations."""
    _CONFIG.verbose = bool(value)


def set_progress(value: bool) -> None:
    """Enable or disable progress bars in base operations."""
    _CONFIG.progress = bool(value)


def get_config() -> _BaseConfig:
    """Return the current base configuration."""
    return _CONFIG


def log(message: str) -> None:
    """Log a message if verbose mode is enabled."""
    if _CONFIG.verbose:
        print(message)


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str | None = None,
    total: int | None = None,
) -> Iterable[T]:
    """Return an iterator with an optional progress bar."""
    if _CONFIG.progress and tqdm is not None:
        return tqdm(iterable, desc=desc, total=total)
    return iterable

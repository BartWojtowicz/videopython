"""Configuration loader for videopython.ai module."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib

# Default backends per task (used when no config file is found)
DEFAULT_BACKENDS: dict[str, str] = {
    "text_to_video": "local",
    "image_to_video": "local",
    "video_upscaler": "local",
    "text_to_speech": "local",
    "text_to_music": "local",
    "text_to_image": "local",
    "image_to_text": "local",
    "audio_to_text": "local",
    "audio_classifier": "local",
    "llm_summarizer": "local",
}


def _find_config_file() -> Path | None:
    """Find the configuration file in current directory or parents.

    Looks for:
    1. videopython.toml in current directory
    2. pyproject.toml in current directory

    Returns:
        Path to config file if found, None otherwise.
    """
    cwd = Path.cwd()

    # Check for videopython.toml
    videopython_toml = cwd / "videopython.toml"
    if videopython_toml.exists():
        return videopython_toml

    # Check for pyproject.toml
    pyproject_toml = cwd / "pyproject.toml"
    if pyproject_toml.exists():
        return pyproject_toml

    return None


def _load_toml(path: Path) -> dict[str, Any]:
    """Load and parse a TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _extract_config(data: dict[str, Any], filename: str) -> dict[str, Any]:
    """Extract videopython config from parsed TOML data.

    Args:
        data: Parsed TOML data
        filename: Name of the file (to determine extraction method)

    Returns:
        The videopython configuration section, or empty dict if not found.
    """
    if filename == "videopython.toml":
        return data
    elif filename == "pyproject.toml":
        return data.get("tool", {}).get("videopython", {})
    return {}


@lru_cache(maxsize=1)
def _get_cached_config() -> dict[str, Any]:
    """Load and cache the configuration.

    Returns:
        The loaded configuration, or empty dict if no config file found.
    """
    config_path = _find_config_file()
    if config_path is None:
        return {}

    try:
        data = _load_toml(config_path)
        return _extract_config(data, config_path.name)
    except tomllib.TOMLDecodeError as e:
        warnings.warn(f"Invalid TOML in config file {config_path}: {e}", RuntimeWarning)
        return {}
    except (OSError, IOError) as e:
        warnings.warn(f"Cannot read config file {config_path}: {e}", RuntimeWarning)
        return {}


def get_config() -> dict[str, Any]:
    """Get the current configuration.

    Returns:
        The configuration dictionary.
    """
    return _get_cached_config()


def get_default_backend(task: str) -> str:
    """Get the default backend for a task.

    Priority:
    1. Config file setting
    2. Hardcoded default ("local")

    Args:
        task: Task name (e.g., 'text_to_video', 'text_to_speech')

    Returns:
        The backend name to use.
    """
    config = get_config()
    ai_defaults = config.get("ai", {}).get("defaults", {})

    if task in ai_defaults:
        return ai_defaults[task]

    return DEFAULT_BACKENDS.get(task, "local")


def clear_config_cache() -> None:
    """Clear the configuration cache. Useful for testing."""
    _get_cached_config.cache_clear()

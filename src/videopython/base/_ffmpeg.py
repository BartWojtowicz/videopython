"""Internal wrappers for ffmpeg / ffprobe subprocess calls.

Centralises subprocess invocation patterns so that every call site shares
the same flag boilerplate, JSON parsing, and failure translation. Public
modules should keep raising their own domain exceptions (VideoLoadError,
AudioLoadError, etc.) and call into the helpers here, mapping
``FFmpegError`` to whichever public exception they document.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

from videopython.base.exceptions import FFmpegProbeError

BASE_FLAGS: tuple[str, ...] = ("-hide_banner", "-loglevel", "error")


def probe(path: str | Path, *, extra_args: Sequence[str] | None = None) -> dict:
    """Run ffprobe and return the parsed JSON payload.

    Args:
        path: Path to the media file.
        extra_args: Optional extra ffprobe flags inserted before ``-print_format``.
            Defaults to ``("-show_streams", "-show_format")`` when omitted,
            which mirrors the historical "everything" probe used by Audio.

    Returns:
        The decoded ffprobe JSON payload.

    Raises:
        FFmpegProbeError: On non-zero exit, JSON decode failure, or missing
            ffprobe binary.
    """
    args = list(extra_args) if extra_args is not None else ["-show_streams", "-show_format"]
    cmd = ["ffprobe", "-v", "error", *args, "-print_format", "json", str(path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegProbeError(f"ffprobe error: {e.stderr}") from e
    except FileNotFoundError as e:
        raise FFmpegProbeError("ffprobe binary not found on PATH") from e

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise FFmpegProbeError(f"Error parsing ffprobe output: {e}") from e

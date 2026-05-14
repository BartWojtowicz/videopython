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

from videopython.base.exceptions import FFmpegProbeError, FFmpegRunError


def run(cmd: Sequence[str], *, stdin: bytes | None = None) -> bytes:
    """Run a blocking ffmpeg/ffprobe command and return stdout.

    Centralises non-zero exit handling so callers can map a single
    ``FFmpegRunError`` to their own domain exception.

    Args:
        cmd: Full argv, starting with ``"ffmpeg"`` or ``"ffprobe"``.
        stdin: Optional bytes to feed to the process's stdin (used by
            the stdin-piped remux variant).

    Returns:
        Process stdout bytes (usually empty for muxing/concat commands).

    Raises:
        FFmpegRunError: On non-zero exit or missing binary.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, input=stdin)
    except FileNotFoundError as e:
        raise FFmpegRunError(f"binary not found on PATH: {cmd[0]}") from e
    if result.returncode != 0:
        raise FFmpegRunError(f"ffmpeg failed (exit {result.returncode}): {result.stderr.decode(errors='replace')}")
    return result.stdout


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

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
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterator, Sequence, cast

from videopython._exceptions import FFmpegProbeError, FFmpegRunError


def _strict_decode_command(cmd: Sequence[str]) -> list[str]:
    argv = list(cmd)
    return [argv[0], "-hide_banner", "-loglevel", "error", "-xerror", *argv[1:]]


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


def run_with_progress(cmd: Sequence[str], on_frame: Callable[[int], None]) -> None:
    """Run an FFmpeg file output and drain frame progress until the process exits."""
    argv = [cmd[0], "-progress", "pipe:1", "-stats_period", "0.25", "-nostats", *cmd[1:]]
    with tempfile.TemporaryFile() as errors:
        try:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=errors)
        except FileNotFoundError as e:
            raise FFmpegRunError(f"binary not found on PATH: {cmd[0]}") from e
        try:
            for line in cast(BinaryIO, proc.stdout):
                key, _, value = line.partition(b"=")
                if key == b"frame":
                    on_frame(int(value))
            if proc.wait() != 0:
                errors.seek(0)
                raise FFmpegRunError(
                    f"ffmpeg failed (exit {proc.returncode}): {errors.read().decode(errors='replace')}"
                )
        finally:
            _terminate(proc)
            cast(BinaryIO, proc.stdout).close()


def probe(path: str | Path, *, extra_args: Sequence[str] | None = None) -> dict[str, Any]:
    """Run ffprobe and return the parsed JSON payload.

    Args:
        path: Path to the media file.
        extra_args: Optional extra ffprobe flags inserted before ``-print_format``.
            Defaults to ``("-show_streams", "-show_format")`` when omitted.

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


def _terminate(proc: subprocess.Popen[bytes], *, timeout: float = 5) -> None:
    """Terminate a still-running process, escalating to kill after ``timeout``."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class _DecodeProcess:
    """FFmpeg decode process with diagnostics stored outside the pipe."""

    def __init__(self, process: subprocess.Popen[bytes], errors: BinaryIO):
        self._process = process
        self._errors = errors

    @property
    def stdout(self) -> BinaryIO:
        return cast(BinaryIO, self._process.stdout)

    def communicate(self) -> tuple[bytes, None]:
        stdout, _ = self._process.communicate()
        return stdout, None

    def check(self) -> None:
        if self._process.wait() == 0:
            return
        self._errors.seek(0)
        detail = self._errors.read().decode(errors="replace")
        raise FFmpegRunError(f"ffmpeg failed (exit {self._process.returncode}): {detail}")


@contextmanager
def popen_decode(cmd: Sequence[str], *, bufsize: int = -1) -> Iterator[_DecodeProcess]:
    """Context manager wrapping an ffmpeg decode process.

    Yields a :class:`_DecodeProcess` with ``stdout=PIPE`` and diagnostics
    stored in a temporary file. A caller that drains stdout must call
    :meth:`_DecodeProcess.check`; a caller that stops early can leave the
    context and the process is terminated without a decode error.

    Args:
        cmd: Full ffmpeg argv. The output target is typically ``pipe:1``.
        bufsize: Forwarded to ``subprocess.Popen``. Use a large value
            (e.g. ``10**8``) for batched reads or a frame-sized value
            for streaming reads.
    """
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(
            _strict_decode_command(cmd),
            stdout=subprocess.PIPE,
            stderr=errors,
            bufsize=bufsize,
        )
        try:
            yield _DecodeProcess(proc, errors)
        finally:
            _terminate(proc)
            if proc.stdout is not None and not proc.stdout.closed:
                proc.stdout.close()


@contextmanager
def popen_encode(cmd: Sequence[str]) -> Iterator[subprocess.Popen[bytes]]:
    """Context manager wrapping an ffmpeg encode process via stdin pipe.

    Yields a Popen with ``stdin=PIPE``, ``stdout=DEVNULL``, and
    ``stderr=PIPE``. Callers write raw frames to ``proc.stdin``.

    On clean exit, stdin and stderr are drained via ``communicate()``
    and ``FFmpegRunError`` is raised if ffmpeg returns non-zero. On
    exception exit, the process is killed and the caller's exception
    propagates unmodified.
    """
    proc = subprocess.Popen(
        list(cmd),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        yield proc
    except BaseException:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for pipe in (proc.stdin, proc.stderr):
            if pipe is not None and not pipe.closed:
                try:
                    pipe.close()
                except Exception:
                    pass
        raise

    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise FFmpegRunError(f"ffmpeg failed (exit {proc.returncode}): {stderr.decode(errors='replace')}")


def escape_filter_value(value: str) -> str:
    """Quote a string for use as an ffmpeg filter option value.

    Escapes the option-value level metacharacters (``\\``, ``'``, ``:``) and
    wraps the result in single quotes so the filtergraph-level separators
    (``,``, ``;``, ``[``, ``]``) pass through untouched.
    """
    escaped = value.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
    return f"'{escaped}'"

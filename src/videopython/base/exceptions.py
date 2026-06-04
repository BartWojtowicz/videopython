"""Exception hierarchy for videopython.base module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class VideoPythonError(Exception):
    """Base exception for all videopython errors."""

    pass


class FFmpegError(VideoPythonError):
    """Base exception for ffmpeg/ffprobe subprocess failures."""

    pass


class FFmpegProbeError(FFmpegError):
    """Raised when an ffprobe invocation or its JSON output fails."""

    pass


class FFmpegRunError(FFmpegError):
    """Raised when a blocking ffmpeg run returns a non-zero exit code."""

    pass


class VideoError(VideoPythonError):
    """Base exception for video-related errors."""

    pass


class VideoLoadError(VideoError):
    """Raised when there's an error loading a video file."""

    pass


class VideoMetadataError(VideoError):
    """Raised when there's an error getting video metadata."""

    pass


class AudioError(VideoPythonError):
    """Base exception for audio-related errors."""

    pass


class AudioLoadError(AudioError):
    """Raised when there's an error loading audio."""

    pass


class TransformError(VideoPythonError):
    """Base exception for transformation errors."""

    pass


class TextRenderError(VideoPythonError):
    """Base exception for text rendering errors."""

    pass


class OutOfBoundsError(TextRenderError):
    """Raised when text would be rendered outside image bounds."""

    pass


class PlanErrorCode(str, Enum):
    """Machine-readable failure classes raised while validating a ``VideoEdit``.

    Scoped to what videopython owns. A consumer can branch on these codes
    instead of substring-matching the human message text.
    """

    SEGMENT_END_EXCEEDS_SOURCE = "segment_end_exceeds_source"
    EFFECT_WINDOW_EXCEEDS_DURATION = "effect_window_exceeds_duration"
    CUT_EXCEEDS_DURATION = "cut_exceeds_duration"
    UNKNOWN_OP = "unknown_op"
    CONCAT_MISMATCH = "concat_mismatch"
    SUBTITLE_UNFITTABLE = "subtitle_unfittable"


@dataclass
class PlanError:
    """A single structured validation failure within a plan.

    ``location`` is a path into the plan (e.g. ``'segments[1].operations[0]'``);
    the remaining fields are populated when meaningful for the ``code``.
    """

    code: PlanErrorCode
    location: str | None = None
    op: str | None = None
    field: str | None = None
    value: float | None = None
    limit: float | None = None
    predicted_duration: float | None = None


class PlanValidationError(ValueError):
    """Typed plan-validation failure carrying structured :class:`PlanError`s.

    Subclasses ``ValueError`` so ``str(e)`` stays byte-identical to the bare
    ``ValueError`` prose emitted before this type existed -- existing
    ``pytest.raises(match=...)`` and consumer substring fallbacks keep working.
    """

    def __init__(self, message: str, errors: list[PlanError]):
        super().__init__(message)
        self.errors = errors

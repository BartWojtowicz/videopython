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


class PlanErrorCode(str, Enum):
    """Machine-readable failure classes raised while validating a ``VideoEdit``.

    Scoped to what videopython owns. A consumer can branch on these codes
    instead of substring-matching the human message text.
    """

    # Segment range vs source / shape.
    SEGMENT_END_EXCEEDS_SOURCE = "segment_end_exceeds_source"
    SEGMENT_NEGATIVE = "segment_negative"
    SEGMENT_RANGE = "segment_range"
    # Effect windows.
    EFFECT_WINDOW_EXCEEDS_DURATION = "effect_window_exceeds_duration"
    WINDOW_NEGATIVE = "window_negative"
    WINDOW_ORDER = "window_order"
    # Operation-level, metadata-relative checks.
    CUT_EXCEEDS_DURATION = "cut_exceeds_duration"
    OP_TIMESTAMP_OUT_OF_RANGE = "op_timestamp_out_of_range"
    CROP_EXCEEDS_SOURCE = "crop_exceeds_source"
    DEGENERATE_DURATION = "degenerate_duration"
    SOURCE_UNREADABLE = "source_unreadable"
    OP_PREDICTION_FAILED = "op_prediction_failed"
    # Assembly / structural.
    UNKNOWN_OP = "unknown_op"
    CONCAT_MISMATCH = "concat_mismatch"
    POST_OP_REQUIRES_CONTEXT = "post_op_requires_context"
    CONTEXT_SOURCE_MISSING = "context_source_missing"
    TRANSITION_TOO_LONG = "transition_too_long"
    MUSIC_BED_DUCK_MULTISEGMENT = "music_bed_duck_multisegment"
    # Streaming: unstreamable op at its plan position (always reported).
    STREAMING_FALLBACK = "streaming_fallback"


@dataclass
class PlanError:
    """A single structured validation failure within a plan.

    ``location`` is a path into the plan (e.g. ``'segments[1].operations[0]'``);
    the remaining fields are populated when meaningful for the ``code``.
    ``detail`` carries a short human-readable cause when the code alone is not
    actionable (e.g. *why* an op cannot stream at its plan position) -- prose meant for
    LLM refine-loop feedback, not for branching.
    """

    code: PlanErrorCode
    location: str | None = None
    op: str | None = None
    field: str | None = None
    value: float | None = None
    limit: float | None = None
    predicted_duration: float | None = None
    detail: str | None = None

    def to_prompt_line(self) -> str:
        """Render this error as one deterministic, actionable feedback line.

        Composes a single line from the structured fields so a *standalone*
        :class:`PlanError` can reproduce a human message without the ad-hoc
        prose built at the validation call sites. Intended for an LLM
        refine-loop: feed these lines back so the model can repair the plan.

        Format (each clause is appended only when its field is populated;
        ``None`` fields are skipped)::

            <CODE_NAME> [at <location>] [(op '<op>')]: [<field>=<value>]
            [(limit <limit>)] [(predicted duration <predicted_duration>s)]
            [-- <detail>]

        Every :class:`PlanErrorCode` yields a non-empty line: the code name is
        always present, so a code carrying no other fields still renders
        (e.g. ``"SOURCE_UNREADABLE"``). Numeric fields are formatted with
        :func:`_fmt_num`, which drops trailing zeros for readability.
        """
        line = self.code.name
        if self.location is not None:
            line += f" at {self.location}"
        if self.op is not None:
            line += f" (op '{self.op}')"

        clauses: list[str] = []
        if self.field is not None:
            if self.value is not None:
                clauses.append(f"{self.field}={_fmt_num(self.value)}")
            else:
                clauses.append(self.field)
        elif self.value is not None:
            clauses.append(f"value={_fmt_num(self.value)}")
        if self.limit is not None:
            clauses.append(f"limit {_fmt_num(self.limit)}")
        if self.predicted_duration is not None:
            clauses.append(f"predicted duration {_fmt_num(self.predicted_duration)}s")

        if clauses:
            line += ": " + ", ".join(clauses)
        if self.detail is not None:
            line += f" -- {self.detail}"
        return line


def _fmt_num(value: float) -> str:
    """Format a numeric field readably: integers lose the ``.0``, floats keep it.

    Keeps :meth:`PlanError.to_prompt_line` deterministic and free of noisy
    trailing zeros (``3`` not ``3.0``, but ``2.5`` stays ``2.5``).
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@dataclass
class PlanRepair:
    """A single change a repair/normalize pass made to a plan.

    The structured changelog returned by :meth:`VideoEdit.repair` and
    :meth:`VideoEdit.normalize_dimensions`. ``location`` is a path into the
    plan (e.g. ``'segments[0].operations[1]'``); ``field`` is the changed
    field (``'window.stop'``, ``'timestamp'``, ``'dimensions'``, ...). ``old``
    and ``new`` carry the before/after values -- a ``float`` for numeric
    clamps, a ``str`` for composite values like ``'768x432'``. ``code`` is the
    :class:`PlanErrorCode` of the violation that was repaired, so a consumer
    can surface "we trimmed your effect to fit" wording keyed on the class.
    """

    location: str
    field: str
    old: float | str | None
    new: float | str | None
    code: PlanErrorCode


class PlanValidationError(ValueError):
    """Typed plan-validation failure carrying structured :class:`PlanError`s.

    Subclasses ``ValueError`` so ``str(e)`` stays byte-identical to the bare
    ``ValueError`` prose emitted before this type existed -- existing
    ``pytest.raises(match=...)`` and consumer substring fallbacks keep working.

    ``str(e)`` is the first error's human message; ``.errors`` carries every
    structured :class:`PlanError`. The non-raising :meth:`VideoEdit.check`
    returns the same ``PlanError`` list directly.
    """

    def __init__(self, message: str, errors: list[PlanError]):
        super().__init__(message)
        self.errors = errors

    def prompt_feedback(self) -> str:
        """Every carried :class:`PlanError` as newline-joined feedback lines.

        Joins :meth:`PlanError.to_prompt_line` over :attr:`errors`, one per
        line -- a ready-to-feed block for an LLM refine loop. Returns an empty
        string when no structured errors are attached.
        """
        return "\n".join(e.to_prompt_line() for e in self.errors)

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PlanErrorCode(str, Enum):
    """Machine-readable edit-plan failure classes."""

    SEGMENT_END_EXCEEDS_SOURCE = "segment_end_exceeds_source"
    SEGMENT_NEGATIVE = "segment_negative"
    SEGMENT_RANGE = "segment_range"
    EFFECT_WINDOW_EXCEEDS_DURATION = "effect_window_exceeds_duration"
    WINDOW_NEGATIVE = "window_negative"
    WINDOW_ORDER = "window_order"
    CUT_EXCEEDS_DURATION = "cut_exceeds_duration"
    OP_TIMESTAMP_OUT_OF_RANGE = "op_timestamp_out_of_range"
    CROP_EXCEEDS_SOURCE = "crop_exceeds_source"
    DEGENERATE_DURATION = "degenerate_duration"
    SOURCE_UNREADABLE = "source_unreadable"
    OP_PREDICTION_FAILED = "op_prediction_failed"
    UNKNOWN_OP = "unknown_op"
    CONCAT_MISMATCH = "concat_mismatch"
    POST_OP_REQUIRES_CONTEXT = "post_op_requires_context"
    CONTEXT_SOURCE_MISSING = "context_source_missing"
    TRANSITION_TOO_LONG = "transition_too_long"
    MUSIC_BED_DUCK_MULTISEGMENT = "music_bed_duck_multisegment"
    STREAMING_UNSUPPORTED = "streaming_unsupported"


@dataclass
class PlanError:
    """A structured validation failure within an edit plan.

    Consumers branch on ``code``; ``detail`` is human-readable feedback.
    """

    code: PlanErrorCode
    location: str | None = None
    op: str | None = None
    field: str | None = None
    value: float | None = None
    limit: float | None = None
    detail: str | None = None

    def to_prompt_line(self) -> str:
        """Render this error as one deterministic feedback line."""
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

        if clauses:
            line += ": " + ", ".join(clauses)
        if self.detail is not None:
            line += f" -- {self.detail}"
        return line


def _fmt_num(value: float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@dataclass
class PlanRepair:
    """A field change made while repairing or normalizing an edit plan."""

    location: str
    field: str
    old: float | str | None
    new: float | str | None
    code: PlanErrorCode


class PlanValidationError(ValueError):
    """A ``ValueError`` whose ``errors`` attribute contains structured failures."""

    def __init__(self, message: str, errors: list[PlanError]):
        super().__init__(message)
        self.errors = errors

    def prompt_feedback(self) -> str:
        """Return every carried error as newline-separated feedback."""
        return "\n".join(error.to_prompt_line() for error in self.errors)

"""Tests for ``PlanError.to_prompt_line`` and ``PlanValidationError.prompt_feedback``."""

from __future__ import annotations

import pytest

from videopython.base.exceptions import PlanError, PlanErrorCode, PlanValidationError


@pytest.mark.parametrize("code", list(PlanErrorCode))
def test_every_code_renders_nonempty_line(code: PlanErrorCode):
    """Every code must yield a non-empty line, even with no other fields set."""
    line = PlanError(code=code).to_prompt_line()
    assert line
    assert line.strip()
    # The code name always leads the line.
    assert line.startswith(code.name)


def test_bare_code_is_just_the_name():
    """A code carrying no structured fields renders as its name alone."""
    assert PlanError(code=PlanErrorCode.SOURCE_UNREADABLE).to_prompt_line() == "SOURCE_UNREADABLE"


def test_segment_end_exceeds_source_includes_numerics():
    """A representative segment error renders field/value/limit."""
    err = PlanError(
        code=PlanErrorCode.SEGMENT_END_EXCEEDS_SOURCE,
        location="segments[1]",
        field="end",
        value=12.0,
        limit=10.0,
    )
    line = err.to_prompt_line()
    assert line == ("SEGMENT_END_EXCEEDS_SOURCE at segments[1]: end=12, limit 10")


def test_window_error_includes_op_and_detail():
    """An op-scoped error renders the op tag and appends the detail prose."""
    err = PlanError(
        code=PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION,
        location="segments[0].operations[2]",
        op="blur",
        field="window.stop",
        value=8.5,
        limit=6.0,
        detail="window clamped past the predicted segment end",
    )
    line = err.to_prompt_line()
    assert line == (
        "EFFECT_WINDOW_EXCEEDS_DURATION at segments[0].operations[2] (op 'blur'): "
        "window.stop=8.5, limit 6 "
        "-- window clamped past the predicted segment end"
    )


def test_streaming_fallback_detail_only():
    """STREAMING_FALLBACK carries only ``detail``; it appends after the code name."""
    err = PlanError(
        code=PlanErrorCode.STREAMING_FALLBACK,
        detail="text overlay cannot stream at this plan position",
    )
    assert err.to_prompt_line() == ("STREAMING_FALLBACK -- text overlay cannot stream at this plan position")


def test_field_without_value_renders_bare_field():
    """A field with no numeric value (e.g. dimensions mismatch) renders the field name."""
    err = PlanError(code=PlanErrorCode.CONCAT_MISMATCH, location="segments[2]", field="dimensions")
    assert err.to_prompt_line() == "CONCAT_MISMATCH at segments[2]: dimensions"


def test_none_fields_are_omitted_cleanly():
    """No ``None`` field leaks into the line as text or stray punctuation."""
    err = PlanError(code=PlanErrorCode.UNKNOWN_OP, op="frobnicate")
    line = err.to_prompt_line()
    assert line == "UNKNOWN_OP (op 'frobnicate')"
    assert "None" not in line
    # No location/field/value/limit/detail -> no separator artifacts.
    assert ":" not in line
    assert "--" not in line


def test_integer_floats_drop_trailing_zero():
    """Integer-valued floats render without the ``.0`` noise; real decimals keep it."""
    integral = PlanError(code=PlanErrorCode.CUT_EXCEEDS_DURATION, field="end", value=5.0, limit=4.0)
    assert integral.to_prompt_line() == "CUT_EXCEEDS_DURATION: end=5, limit 4"

    fractional = PlanError(code=PlanErrorCode.CUT_EXCEEDS_DURATION, field="end", value=5.25, limit=4.5)
    assert fractional.to_prompt_line() == "CUT_EXCEEDS_DURATION: end=5.25, limit 4.5"


def test_value_without_field_labels_as_value():
    """A populated ``value`` with no ``field`` still renders, labelled ``value``."""
    err = PlanError(code=PlanErrorCode.DEGENERATE_DURATION, value=0.0)
    assert err.to_prompt_line() == "DEGENERATE_DURATION: value=0"


def test_prompt_feedback_joins_lines():
    """``prompt_feedback`` newline-joins every carried error's line."""
    errors = [
        PlanError(code=PlanErrorCode.UNKNOWN_OP, op="frobnicate"),
        PlanError(code=PlanErrorCode.SOURCE_UNREADABLE, field="source"),
    ]
    exc = PlanValidationError("first error message", errors)
    assert exc.prompt_feedback() == "UNKNOWN_OP (op 'frobnicate')\nSOURCE_UNREADABLE: source"
    # str(exc) is unchanged: still the human message passed in.
    assert str(exc) == "first error message"


def test_prompt_feedback_empty_when_no_errors():
    """No structured errors -> empty feedback block."""
    assert PlanValidationError("boom", []).prompt_feedback() == ""

"""Tests for the LLM-refine-loop primitives on ``VideoEdit``.

Covers the validation/repair substrate added for grammar-constrained,
auto-repairable plan generation: collect-all :meth:`VideoEdit.check`, the
extended :meth:`VideoEdit.repair`, :meth:`VideoEdit.normalize_dimensions`, and
the strict :meth:`VideoEdit.json_schema` / :meth:`Operation.json_schema` grammar.
Byte-stable raising behaviour lives in ``test_video_edit.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.test_config import SMALL_VIDEO_METADATA
from videopython.base.exceptions import PlanErrorCode, PlanValidationError
from videopython.base.video import VideoMetadata
from videopython.editing import Operation, VideoEdit

META = VideoMetadata(height=720, width=1280, fps=24, frame_count=1200, total_seconds=50.0)


def _segment(
    *, start: float = 0.0, end: float = 10.0, operations: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {"source": "a.mp4", "start": start, "end": end, "operations": operations or []}


def _codes(errors: list[Any]) -> list[PlanErrorCode]:
    return [e.code for e in errors]


# --------------------------------------------------------------- check (collect-all)


class TestCheck:
    def test_valid_plan_returns_empty(self):
        edit = VideoEdit.from_dict({"segments": [_segment(start=0.0, end=5.0)]})
        assert edit.check(SMALL_VIDEO_METADATA) == []

    def test_collects_independent_segment_errors_in_one_pass(self):
        # Two segments each with their own latent fault: both must surface.
        plan = {
            "segments": [
                _segment(start=0.0, end=10.0, operations=[{"op": "freeze_frame", "timestamp": 999.0}]),
                _segment(start=0.0, end=10.0, operations=[{"op": "crop", "width": 9999, "height": 9999}]),
            ]
        }
        errs = VideoEdit.from_dict(plan).check(META)
        assert PlanErrorCode.OP_TIMESTAMP_OUT_OF_RANGE in _codes(errs)
        assert PlanErrorCode.CROP_EXCEEDS_SOURCE in _codes(errs)
        assert {e.location for e in errs} == {"segments[0].operations[0]", "segments[1].operations[0]"}

    def test_failed_segment_isolates_dependent_checks(self):
        # An op failure breaks the segment's chain and (since the timeline can't
        # assemble) skips the dependent post-op checks -- best-effort isolation,
        # as documented. The repair->check flow handles the post-op window: repair
        # clamps the freeze so the segment predicts (see TestRepair).
        plan = {
            "segments": [_segment(operations=[{"op": "freeze_frame", "timestamp": 999.0}])],
            "post_operations": [{"op": "blur_effect", "mode": "constant", "iterations": 1, "window": {"start": -2.0}}],
        }
        codes = _codes(VideoEdit.from_dict(plan).check(META))
        assert codes == [PlanErrorCode.OP_TIMESTAMP_OUT_OF_RANGE]  # post-op window deferred

    def test_chain_break_isolates_remaining_ops(self):
        # op0 fails prediction; the rest of that segment's chain is skipped.
        plan = {
            "segments": [
                _segment(
                    start=0.0,
                    end=10.0,
                    operations=[
                        {"op": "crop", "width": 9999, "height": 9999},
                        {"op": "blur_effect", "mode": "constant", "iterations": 1, "window": {"start": -5.0}},
                    ],
                ),
            ]
        }
        codes = _codes(VideoEdit.from_dict(plan).check(META))
        assert codes == [PlanErrorCode.CROP_EXCEEDS_SOURCE]

    def test_uncuttable_segment_reports_bounds_only(self):
        # A segment that cannot be cut (bad range) isolates to its bounds error;
        # its op chain is skipped (best-effort, as documented).
        plan = {
            "segments": [
                _segment(
                    start=-1.0,
                    end=5.0,
                    operations=[
                        {
                            "op": "blur_effect",
                            "mode": "constant",
                            "iterations": 1,
                            "window": {"start": 3.0, "stop": 1.0},
                        }
                    ],
                ),
            ]
        }
        codes = _codes(VideoEdit.from_dict(plan).check(SMALL_VIDEO_METADATA))
        assert PlanErrorCode.SEGMENT_NEGATIVE in codes

    def test_concat_mismatch_collected_when_all_segments_predict(self):
        plan = {
            "segments": [
                _segment(start=0.0, end=5.0, operations=[{"op": "resize", "width": 640, "height": 360}]),
                _segment(start=0.0, end=5.0, operations=[{"op": "resize", "width": 800, "height": 450}]),
            ],
            "match_to_lowest_resolution": False,
        }
        errs = VideoEdit.from_dict(plan).check(SMALL_VIDEO_METADATA)
        assert _codes(errs) == [PlanErrorCode.CONCAT_MISMATCH]

    def test_check_matches_validate_first_error(self):
        # The first collected error is exactly what validate would raise.
        plan = {"segments": [_segment(start=0.0, end=99.0)]}
        edit = VideoEdit.from_dict(plan)
        errs = edit.check(SMALL_VIDEO_METADATA)
        with pytest.raises(PlanValidationError) as exc:
            edit.validate_with_metadata(SMALL_VIDEO_METADATA)
        assert errs[0].code == exc.value.errors[0].code == PlanErrorCode.SEGMENT_END_EXCEEDS_SOURCE

    def test_clamp_windows_suppresses_clampable_stop_overrun(self):
        plan = {
            "segments": [
                _segment(
                    start=0.0,
                    end=12.0,
                    operations=[
                        {"op": "speed_change", "speed": 1.5, "adjust_audio": False},
                        {
                            "op": "blur_effect",
                            "mode": "constant",
                            "iterations": 1,
                            "window": {"start": 0, "stop": 10.0},
                        },
                    ],
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        assert PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION in _codes(edit.check(SMALL_VIDEO_METADATA))
        assert edit.check(SMALL_VIDEO_METADATA, clamp_windows=True) == []

    def test_post_op_requires_context_is_structured(self):
        from videopython.base.transcription import Transcription, TranscriptionWord

        plan = {
            "segments": [_segment(), _segment()],
            "post_operations": [{"op": "silence_removal"}],
        }
        tx = Transcription(words=[TranscriptionWord(start=1.0, end=2.0, word="hi")])
        errs = VideoEdit.from_dict(plan).check(SMALL_VIDEO_METADATA, context={"transcription": tx})
        assert PlanErrorCode.POST_OP_REQUIRES_CONTEXT in _codes(errs)


# --------------------------------------------------------------------- repair


class TestRepair:
    def test_clamps_freeze_timestamp_into_range(self):
        plan = {"segments": [_segment(start=0.0, end=10.0, operations=[{"op": "freeze_frame", "timestamp": 999.0}])]}
        edit = VideoEdit.from_dict(plan)
        repaired, changes = edit.repair(META)
        assert [c.code for c in changes] == [PlanErrorCode.OP_TIMESTAMP_OUT_OF_RANGE]
        assert changes[0].location == "segments[0].operations[0]"
        assert changes[0].old == 999.0
        # Clamped to the last addressable frame, strictly < the 10s clip.
        assert 0 <= repaired.segments[0].operations[0].timestamp < 10.0
        assert repaired.check(META) == []
        # Source plan is untouched.
        assert edit.segments[0].operations[0].timestamp == 999.0

    def test_clamps_negative_and_overrun_window_bounds(self):
        plan = {
            "segments": [
                _segment(
                    start=0.0,
                    end=10.0,
                    operations=[
                        {
                            "op": "blur_effect",
                            "mode": "constant",
                            "iterations": 1,
                            "window": {"start": -1.0, "stop": 999.0},
                        }
                    ],
                )
            ]
        }
        repaired, changes = VideoEdit.from_dict(plan).repair(META)
        codes = {c.code for c in changes}
        assert PlanErrorCode.WINDOW_NEGATIVE in codes
        assert PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION in codes
        win = repaired.segments[0].operations[0].window
        assert win.start == 0.0 and win.stop == 10.0
        assert repaired.check(META) == []

    def test_clamps_negative_post_op_window(self):
        # The canonical attempt-3 failure: a negative post_operations window.
        plan = {
            "segments": [_segment(start=0.0, end=10.0)],
            "post_operations": [{"op": "blur_effect", "mode": "constant", "iterations": 1, "window": {"start": -2.0}}],
        }
        repaired, changes = VideoEdit.from_dict(plan).repair(META)
        assert [c.code for c in changes] == [PlanErrorCode.WINDOW_NEGATIVE]
        assert changes[0].location == "post_operations[0]"
        assert repaired.post_operations[0].window.start == 0.0
        assert repaired.check(META) == []

    def test_clamps_negative_segment_start(self):
        plan = {"segments": [_segment(start=-3.0, end=10.0)]}
        repaired, changes = VideoEdit.from_dict(plan).repair(META)
        assert [c.code for c in changes] == [PlanErrorCode.SEGMENT_NEGATIVE]
        assert repaired.segments[0].start == 0.0

    def test_segment_end_overrun_raises_without_flag(self):
        edit = VideoEdit.from_dict({"segments": [_segment(start=0.0, end=99.0)]})
        with pytest.raises(PlanValidationError) as exc:
            edit.repair(SMALL_VIDEO_METADATA)
        assert exc.value.errors[0].code is PlanErrorCode.SEGMENT_END_EXCEEDS_SOURCE

    def test_segment_end_overrun_clamped_with_flag(self):
        edit = VideoEdit.from_dict({"segments": [_segment(start=0.0, end=99.0)]})
        repaired, changes = edit.repair(SMALL_VIDEO_METADATA, clamp_segment_end=True)
        assert [c.code for c in changes] == [PlanErrorCode.SEGMENT_END_EXCEEDS_SOURCE]
        assert repaired.segments[0].end == pytest.approx(SMALL_VIDEO_METADATA.total_seconds)
        assert repaired.check(SMALL_VIDEO_METADATA) == []

    def test_unrepairable_op_left_for_check_no_raise(self):
        # A crop exceeding source is not a declared clamp: repair leaves it,
        # returns the plan, and check() still reports it.
        plan = {"segments": [_segment(start=0.0, end=10.0, operations=[{"op": "crop", "width": 9999, "height": 9999}])]}
        edit = VideoEdit.from_dict(plan)
        repaired, changes = edit.repair(META)
        assert changes == []
        assert PlanErrorCode.CROP_EXCEEDS_SOURCE in _codes(repaired.check(META))

    def test_in_range_time_field_is_not_touched(self):
        # An already-valid freeze timestamp must not be rounded/recorded -- no
        # phantom changelog entry. 7.123456 has >4 decimals but is in range.
        plan = {"segments": [_segment(start=0.0, end=10.0, operations=[{"op": "freeze_frame", "timestamp": 7.123456}])]}
        repaired, changes = VideoEdit.from_dict(plan).repair(META)
        assert changes == []
        assert repaired.segments[0].operations[0].timestamp == 7.123456

    def test_in_range_window_is_not_touched(self):
        plan = {
            "segments": [
                _segment(
                    start=0.0,
                    end=10.0,
                    operations=[
                        {
                            "op": "blur_effect",
                            "mode": "constant",
                            "iterations": 1,
                            "window": {"start": 1.0, "stop": 8.0},
                        }
                    ],
                )
            ]
        }
        repaired, changes = VideoEdit.from_dict(plan).repair(META)
        assert changes == []
        assert repaired.segments[0].operations[0].window.stop == 8.0

    def test_clamp_op_params_false_is_noop_for_windows(self):
        plan = {
            "segments": [
                _segment(
                    start=0.0,
                    end=10.0,
                    operations=[{"op": "blur_effect", "mode": "constant", "iterations": 1, "window": {"start": -1.0}}],
                )
            ]
        }
        repaired, changes = VideoEdit.from_dict(plan).repair(META, clamp_op_params=False)
        assert changes == []
        assert repaired.segments[0].operations[0].window.start == -1.0


# ------------------------------------------------------------- normalize_dimensions


class TestNormalizeDimensions:
    def _mismatched_plan(self) -> VideoEdit:
        return VideoEdit.from_dict(
            {
                "segments": [
                    _segment(start=0.0, end=5.0, operations=[{"op": "resize", "width": 640, "height": 360}]),
                    _segment(start=0.0, end=5.0, operations=[{"op": "resize", "width": 1280, "height": 720}]),
                ],
                "match_to_lowest_resolution": False,
            }
        )

    def test_explicit_target_resizes_all_to_canvas(self):
        norm, repairs = self._mismatched_plan().normalize_dimensions((1920, 1080), SMALL_VIDEO_METADATA)
        assert {(r.location, r.new) for r in repairs} == {
            ("segments[0]", "1920x1080"),
            ("segments[1]", "1920x1080"),
        }
        meta = norm.validate_with_metadata(SMALL_VIDEO_METADATA)
        assert (meta.width, meta.height) == (1920, 1080)

    def test_first_target_only_resizes_divergent_segments(self):
        norm, repairs = self._mismatched_plan().normalize_dimensions("first", SMALL_VIDEO_METADATA)
        # Segment 0 already matches "first"; only segment 1 is rewritten.
        assert [r.location for r in repairs] == ["segments[1]"]
        assert norm.check(SMALL_VIDEO_METADATA) == []

    def test_largest_target_picks_max_area(self):
        norm, repairs = self._mismatched_plan().normalize_dimensions("largest", SMALL_VIDEO_METADATA)
        assert [r.location for r in repairs] == ["segments[0]"]
        meta = norm.validate_with_metadata(SMALL_VIDEO_METADATA)
        assert (meta.width, meta.height) == (1280, 720)

    def test_already_uniform_plan_is_unchanged(self):
        edit = VideoEdit.from_dict({"segments": [_segment(start=0.0, end=5.0), _segment(start=5.0, end=10.0)]})
        norm, repairs = edit.normalize_dimensions("first", SMALL_VIDEO_METADATA)
        assert repairs == []


# --------------------------------------------------------------- strict json schema


class TestStrictSchema:
    def test_all_objects_closed_and_required(self):
        schema = VideoEdit.json_schema(strict=True)

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if isinstance(node.get("properties"), dict):
                    assert node.get("additionalProperties") is False
                    assert set(node["required"]) == set(node["properties"].keys())
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(schema)

    def test_union_is_anyof_without_discriminator(self):
        blob = json.dumps(VideoEdit.json_schema(strict=True))
        assert "oneOf" not in blob
        assert "discriminator" not in blob
        assert "anyOf" in blob

    def test_optional_field_made_nullable(self):
        seg = VideoEdit.json_schema(strict=True)["properties"]["segments"]["items"]
        ops = seg["properties"]["operations"]
        assert any(isinstance(v, dict) and v.get("type") == "null" for v in ops["anyOf"])

    def test_discriminator_op_stays_non_nullable(self):
        # The `op` tag must remain a required, non-nullable const in every
        # variant -- a nullable discriminator yields an unroutable payload.
        schema = Operation.json_schema(strict=True)
        defs = schema.get("$defs") or schema.get("definitions") or {}
        variants = [s for s in defs.values() if "op" in s.get("properties", {})]
        assert variants, "no op variants found"
        for v in variants:
            op_prop = v["properties"]["op"]
            assert "const" in op_prop, f"op tag lost its const: {op_prop}"
            assert "anyOf" not in op_prop and op_prop.get("type") != "null"
            assert "op" in v["required"]

    def test_strips_format_and_schema_envelope(self):
        blob = json.dumps(VideoEdit.json_schema(strict=True))
        assert '"format"' not in blob  # e.g. the source "path" format
        assert "$schema" not in blob
        # non-strict keeps both
        nb = json.dumps(VideoEdit.json_schema())
        assert '"format"' in nb and "$schema" in nb

    def test_numeric_constraints_preserved(self):
        blob = json.dumps(Operation.json_schema(strict=True))
        assert "exclusiveMinimum" in blob or "minimum" in blob

    def test_strips_default_keyword(self):
        # Strict mode rejects `default`; non-strict keeps it.
        assert '"default"' not in json.dumps(VideoEdit.json_schema(strict=True))
        assert '"default"' in json.dumps(VideoEdit.json_schema())

    def test_non_strict_schema_unchanged(self):
        blob = json.dumps(VideoEdit.json_schema())
        assert "discriminator" in blob or "oneOf" in blob

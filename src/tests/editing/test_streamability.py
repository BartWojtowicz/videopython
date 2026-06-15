"""Tests for the per-op streamability report and ``strict_streaming`` (P0.2)."""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_config import SMALL_VIDEO_METADATA, SMALL_VIDEO_PATH
from videopython.base.exceptions import PlanErrorCode, PlanValidationError
from videopython.editing import StreamingClass
from videopython.editing.effects import Effect, Fade
from videopython.editing.operation import Operation
from videopython.editing.transforms import Resize
from videopython.editing.video_edit import VideoEdit

FADE = {"op": "fade", "mode": "in", "duration": 0.5}
RESIZE = {"op": "resize", "width": 640, "height": 480}
SUBTITLES = {"op": "add_subtitles", "font_scale": 0.1}
CUT = {"op": "cut", "start": 0.0, "end": 1.0}
SILENCE = {"op": "silence_removal"}


def _plan(
    operations: list[dict[str, Any]],
    *,
    post_operations: list[dict[str, Any]] | None = None,
    n_segments: int = 1,
) -> VideoEdit:
    return VideoEdit.model_validate(
        {
            "segments": [
                {"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 2.0, "operations": operations}
                for _ in range(n_segments)
            ],
            "post_operations": post_operations or [],
        }
    )


class TestStreamabilityReport:
    """The pure structural classification behind ``VideoEdit.streamability()``."""

    def test_filter_and_frame_effect_classes(self):
        report = _plan([RESIZE, FADE, SUBTITLES]).streamability()

        # add_subtitles compiles to a filter (encode-stage here, since it
        # follows a frame effect) -- FILTER either way.
        assert [e.streaming_class for e in report.entries] == [
            StreamingClass.FILTER,
            StreamingClass.FRAME_EFFECT,
            StreamingClass.FILTER,
        ]
        assert [e.location for e in report.entries] == [
            "segments[0].operations[0]",
            "segments[0].operations[1]",
            "segments[0].operations[2]",
        ]
        assert report.streamable
        assert report.fallbacks == ()
        assert report.errors() == []
        assert all(e.reason is None for e in report.entries)

    def test_transform_after_effect_streams_via_encode_stage(self):
        # 0.42.0: transforms following frame effects join the encode-stage
        # filter chain instead of forcing the whole-plan eager fallback.
        report = _plan([FADE, RESIZE]).streamability()

        fade, resize = report.entries
        assert fade.streaming_class is StreamingClass.FRAME_EFFECT
        assert resize.streaming_class is StreamingClass.FILTER
        assert report.streamable

    def test_effect_after_encode_stage_is_unstreamable(self):
        report = _plan([FADE, RESIZE, {"op": "color_adjust", "brightness": 0.2}]).streamability()

        trailing = report.entries[2]
        assert trailing.streaming_class is StreamingClass.UNSTREAMABLE
        assert trailing.reason is not None and "encode-stage" in trailing.reason
        assert not report.streamable

    def test_context_requiring_transform_streams_as_filter(self):
        # silence_removal consumes its transcription at plan compile (0.42.0).
        report = _plan([SILENCE]).streamability()

        (entry,) = report.entries
        assert entry.streaming_class is StreamingClass.FILTER

    def test_context_transform_after_duration_change_is_unstreamable(self):
        report = _plan([{"op": "speed_change", "speed": 2.0}, SILENCE]).streamability()

        speed, silence = report.entries
        assert speed.streaming_class is StreamingClass.FILTER
        assert silence.streaming_class is StreamingClass.UNSTREAMABLE
        assert silence.reason is not None and "duration-changing" in silence.reason

    def test_unfilterable_transform_is_unstreamable(self):
        report = _plan([CUT]).streamability()

        (entry,) = report.entries
        assert entry.streaming_class is StreamingClass.UNSTREAMABLE
        assert entry.reason is not None and "no ffmpeg filter" in entry.reason

    def test_non_streamable_effect_is_unstreamable(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(Fade, "streamable", False)
        report = _plan([FADE]).streamability()

        (entry,) = report.entries
        assert entry.streaming_class is StreamingClass.UNSTREAMABLE
        assert entry.reason is not None and "streamable=False" in entry.reason

    def test_post_op_effect_on_single_segment_streams(self):
        report = _plan([], post_operations=[FADE]).streamability()

        (entry,) = report.entries
        assert entry.location == "post_operations[0]"
        assert entry.streaming_class is StreamingClass.FRAME_EFFECT
        assert report.streamable

    def test_audio_coupled_post_op_on_multi_segment_plan_is_unstreamable(self):
        report = _plan([], post_operations=[FADE], n_segments=2).streamability()

        (entry,) = report.entries
        assert entry.streaming_class is StreamingClass.UNSTREAMABLE
        assert entry.reason is not None and "multi-segment" in entry.reason

    def test_context_requiring_post_op_is_unstreamable(self):
        report = _plan([], post_operations=[SUBTITLES]).streamability()

        (entry,) = report.entries
        assert entry.streaming_class is StreamingClass.UNSTREAMABLE
        assert entry.reason is not None and "transcription" in entry.reason

    def test_transform_post_op_is_unstreamable(self):
        report = _plan([], post_operations=[RESIZE]).streamability()

        (entry,) = report.entries
        assert entry.streaming_class is StreamingClass.UNSTREAMABLE
        assert entry.reason is not None and "post-operations" in entry.reason

    def test_errors_carry_location_op_and_detail(self):
        errors = _plan([FADE, RESIZE, {"op": "color_adjust", "brightness": 0.2}]).streamability().errors()

        (err,) = errors
        assert err.code is PlanErrorCode.STREAMING_FALLBACK
        assert err.location == "segments[0].operations[2]"
        assert err.op == "color_adjust"
        assert err.detail is not None and "encode-stage" in err.detail


class TestRegistryAlignment:
    def test_streamable_flag_matches_filter_compilation(self):
        """Every registered transform must declare ``streamable`` coherently.

        Both the report and the plan builder treat the ``streamable`` ClassVar
        as authoritative, but a flag-True transform without a working
        ``to_ffmpeg_filter`` only fails at runtime (the strict-mode drift
        guard), and a flag-False transform with one carries dead filter code.
        This covers ops registered by the editing layer; a twin test under
        ``src/tests/ai`` covers the ai layer, which this suite must not
        import.
        """
        for op_id, cls in Operation._registry.items():
            if issubclass(cls, Effect):
                continue
            overrides_filter = cls.to_ffmpeg_filter is not Operation.to_ffmpeg_filter
            assert overrides_filter == cls.streamable, (
                f"op '{op_id}': streamable={cls.streamable} but "
                f"{'overrides' if overrides_filter else 'does not override'} to_ffmpeg_filter"
            )


class TestCheckStrictStreaming:
    def test_check_reports_unstreamable_ops(self):
        # Streaming is the only engine: unstreamable shapes are plan errors.
        plan = _plan([FADE, RESIZE, {"op": "color_adjust", "brightness": 0.2}])
        errors = plan.check(SMALL_VIDEO_METADATA)

        (err,) = errors
        assert err.code is PlanErrorCode.STREAMING_FALLBACK
        assert err.location == "segments[0].operations[2]"
        assert err.op == "color_adjust"

    def test_check_on_streamable_plan_is_clean(self):
        plan = _plan([RESIZE, FADE])
        assert plan.check(SMALL_VIDEO_METADATA) == []

    def test_fallback_errors_append_after_validity_errors(self):
        bad_window = {"op": "fade", "mode": "in", "duration": 0.5, "window": {"start": 50.0, "stop": 60.0}}
        plan = _plan([bad_window, RESIZE, {"op": "color_adjust", "brightness": 0.2}])
        errors = plan.check(SMALL_VIDEO_METADATA)

        assert len(errors) >= 2
        assert errors[0].code is not PlanErrorCode.STREAMING_FALLBACK
        assert errors[-1].code is PlanErrorCode.STREAMING_FALLBACK


class TestRunToFileRejection:
    def test_streamable_plan_streams(self, tmp_path):
        plan = _plan([RESIZE, FADE])
        out = plan.run_to_file(tmp_path / "out.mp4")
        assert out.exists()

    def test_multi_segment_streamable_plan_streams(self, tmp_path):
        plan = VideoEdit.model_validate(
            {
                "segments": [
                    {"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 1.0, "operations": [RESIZE]},
                    {"source": SMALL_VIDEO_PATH, "start": 1.0, "end": 2.0, "operations": [RESIZE, FADE]},
                ],
            }
        )
        out = plan.run_to_file(tmp_path / "out.mp4")
        assert out.exists()

    def test_unstreamable_plan_raises(self, tmp_path):
        plan = _plan([FADE, RESIZE, {"op": "color_adjust", "brightness": 0.2}])
        out_path = tmp_path / "out.mp4"

        with pytest.raises(PlanValidationError, match="cannot stream") as exc_info:
            plan.run_to_file(out_path)

        assert not out_path.exists()
        (err,) = exc_info.value.errors
        assert err.code is PlanErrorCode.STREAMING_FALLBACK
        assert err.op == "color_adjust"

    def test_run_raises_the_same_errors(self, tmp_path):
        plan = _plan([FADE, RESIZE, {"op": "color_adjust", "brightness": 0.2}])
        with pytest.raises(PlanValidationError, match="cannot stream"):
            plan.run()

    def test_builder_drift_raises_instead_of_silent_eager(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """A streamable-flagged transform that fails to compile must not slip through."""
        monkeypatch.setattr(Resize, "to_ffmpeg_filter", lambda self, ctx: None)
        plan = _plan([RESIZE])

        with pytest.raises(PlanValidationError, match="despite a clean streamability report"):
            plan.run_to_file(tmp_path / "out.mp4")

    def test_flag_false_transform_is_rejected_even_with_working_filter(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """The ``streamable`` flag is authoritative in both directions: a
        transform declaring ``streamable=False`` is rejected even if its
        ``to_ffmpeg_filter`` compiles, so the report and the runtime can
        never disagree."""
        monkeypatch.setattr(Resize, "streamable", False)
        plan = _plan([RESIZE])

        assert not plan.streamability().streamable
        with pytest.raises(PlanValidationError, match="cannot stream"):
            plan.run_to_file(tmp_path / "out.mp4")

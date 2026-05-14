"""Tests for the Operation/Effect base machinery in editing/operation.py.

Covers auto-registration, the discriminated-union JSON schema, ``TimeRange``
validation, and the ``Effect.apply`` window/invariant logic. Per-op
behavioural tests live in ``test_transforms.py`` / ``test_effects.py``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import pytest
from pydantic import Field, ValidationError

from videopython.base.video import Video
from videopython.editing.operation import (
    Effect,
    FilterCtx,
    OpCategory,
    Operation,
    TimeRange,
)


@pytest.fixture(autouse=True)
def _isolate_operation_registry():
    """Snapshot Operation._registry around every test so toy subclasses
    defined in test bodies don't leak across modules."""
    snapshot = dict(Operation._registry)
    yield
    Operation._registry.clear()
    Operation._registry.update(snapshot)


# --- TimeRange ---------------------------------------------------------------


class TestTimeRange:
    def test_defaults_to_open_ended(self):
        tr = TimeRange()
        assert tr.start is None and tr.stop is None

    def test_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            TimeRange(start=-1.0)

    def test_stop_before_start_rejected(self):
        with pytest.raises(ValidationError):
            TimeRange(start=3.0, stop=1.0)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            TimeRange.model_validate({"start": 0, "stop": 1, "bogus": True})


# --- Operation registry -----------------------------------------------------


class _DocOp(Operation):
    """A toy op for testing."""

    op: Literal["_doc_op"] = "_doc_op"
    category: ClassVar[OpCategory] = OpCategory.TRANSFORM

    width: int = Field(1, gt=0, description="Target width in pixels.")
    height: int = Field(1, gt=0, description="Target height in pixels.")


class TestRegistry:
    def test_subclass_is_registered(self):
        assert Operation.registry()["_doc_op"] is _DocOp

    def test_lookup_via_get(self):
        assert Operation.get("_doc_op") is _DocOp

    def test_get_unknown_raises_keyerror(self):
        with pytest.raises(KeyError, match="Unknown op_id"):
            Operation.get("never_registered")

    def test_duplicate_op_id_raises(self):
        with pytest.raises(ValueError, match="Duplicate op_id"):

            class _Dup(Operation):
                op: Literal["_doc_op"] = "_doc_op"

    def test_op_id_property_mirrors_op(self):
        assert _DocOp(width=2, height=3).op_id == "_doc_op"

    def test_field_descriptions_flow_into_schema(self):
        schema = _DocOp.model_json_schema()
        assert schema["properties"]["width"]["description"] == "Target width in pixels."
        assert schema["properties"]["height"]["description"] == "Target height in pixels."


# --- JSON schema -------------------------------------------------------------


class TestJsonSchema:
    def test_discriminated_union(self):
        schema = Operation.json_schema()
        assert schema["discriminator"]["propertyName"] == "op"
        mapping = schema["discriminator"]["mapping"]
        assert "_doc_op" in mapping
        assert len(schema["oneOf"]) == len(Operation.registry())

    def test_per_op_schema_has_op_const(self):
        schema = _DocOp.model_json_schema()
        # In Pydantic v2, a single-value Literal becomes a const-ish enum.
        op_schema = schema["properties"]["op"]
        assert op_schema.get("const") == "_doc_op" or op_schema.get("enum") == ["_doc_op"]


# --- Operation.apply default ------------------------------------------------


class TestOperationApplyDefault:
    def test_default_apply_raises(self, black_frames_test_video: Video):
        class _UnimplOp(Operation):
            op: Literal["_unimpl"] = "_unimpl"

        with pytest.raises(NotImplementedError):
            _UnimplOp().apply(black_frames_test_video)

    def test_default_predict_metadata_is_identity(self):
        class _NoOp(Operation):
            op: Literal["_no_op"] = "_no_op"

        from videopython.base.video import VideoMetadata

        meta = VideoMetadata(height=10, width=20, fps=30.0, frame_count=60, total_seconds=2.0)
        assert _NoOp().predict_metadata(meta) is meta

    def test_default_to_ffmpeg_filter_returns_none(self):
        class _Eager(Operation):
            op: Literal["_eager"] = "_eager"

        assert _Eager().to_ffmpeg_filter(FilterCtx(width=100, height=100, fps=30.0)) is None


# --- Effect window / invariant ----------------------------------------------


class _Brighten(Effect):
    """Add a constant brightness offset to every frame."""

    op: Literal["_brighten"] = "_brighten"

    delta: int = Field(10, ge=-255, le=255, description="Amount added to every pixel (clipped to uint8).")

    def _apply(self, video: Video) -> Video:
        from videopython.base.video import Video as _V

        out = np.clip(video.frames.astype(np.int16) + self.delta, 0, 255).astype(np.uint8)
        result = _V.from_frames(out, fps=video.fps)
        result.audio = video.audio
        return result


class _ShapeBreaker(Effect):
    op: Literal["_shape_breaker"] = "_shape_breaker"

    def _apply(self, video: Video) -> Video:
        from videopython.base.video import Video as _V

        return _V.from_frames(video.frames[:-1], fps=video.fps)


class TestEffectApply:
    def test_no_window_applies_to_whole_video(self, black_frames_test_video: Video):
        original_mean = black_frames_test_video.frames.mean()
        out = _Brighten(delta=20).apply(black_frames_test_video)
        assert out.video_shape == black_frames_test_video.video_shape
        assert out.frames.mean() > original_mean

    def test_window_applies_only_inside_range(self, black_frames_test_video: Video):
        fps = black_frames_test_video.fps
        win_start_f = round(1.0 * fps)
        win_end_f = round(2.0 * fps)

        out = _Brighten(delta=20, window=TimeRange(start=1.0, stop=2.0)).apply(black_frames_test_video)

        assert out.video_shape == black_frames_test_video.video_shape
        # Outside the window: untouched.
        np.testing.assert_array_equal(out.frames[:win_start_f], black_frames_test_video.frames[:win_start_f])
        np.testing.assert_array_equal(out.frames[win_end_f:], black_frames_test_video.frames[win_end_f:])
        # Inside the window: brightened.
        assert out.frames[win_start_f:win_end_f].mean() > black_frames_test_video.frames[win_start_f:win_end_f].mean()

    def test_window_clamps_to_video_duration(self, black_frames_test_video: Video):
        # stop past end -> clamped, still works.
        out = _Brighten(delta=5, window=TimeRange(start=0.0, stop=black_frames_test_video.total_seconds + 100)).apply(
            black_frames_test_video
        )
        assert out.video_shape == black_frames_test_video.video_shape

    def test_shape_breaking_apply_raises(self, black_frames_test_video: Video):
        with pytest.raises(RuntimeError, match="changed video shape"):
            _ShapeBreaker().apply(black_frames_test_video)

    def test_default_process_frame_raises(self):
        with pytest.raises(NotImplementedError, match="does not support streaming"):
            _Brighten(delta=1).process_frame(np.zeros((10, 10, 3), dtype=np.uint8), 0)

    def test_predict_metadata_accepts_context_kwargs(self):
        """Effects preserve shape/frame_count, so predict_metadata is identity —
        but it must accept arbitrary ``**context`` so requires-aware effects
        (e.g. ``TranscriptionOverlay``) don't blow up when the runner threads
        kwargs in. Symmetric with ``Effect.apply``'s ``**context``.
        """
        from videopython.base.video import VideoMetadata as _Meta

        meta = _Meta(height=720, width=1280, fps=30, frame_count=300, total_seconds=10.0)
        out = _Brighten(delta=1).predict_metadata(meta, transcription="ignored", anything=42)
        assert out == meta

"""Tests for VideoEdit editing plan model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.test_config import BIG_VIDEO_PATH, SMALL_VIDEO_METADATA, SMALL_VIDEO_PATH
from videopython.base.edit import EffectApplication, SegmentConfig, VideoEdit
from videopython.base.effects import Blur, ColorGrading
from videopython.base.transforms import Crop, ResampleFPS, Resize, SpeedChange
from videopython.base.video import Video, VideoMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_video(width: int, height: int, fps: float, seconds: float) -> Video:
    frame_count = round(fps * seconds)
    frames = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    return Video(frames=frames, fps=fps)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_segments_raises(self):
        with pytest.raises(ValueError, match="at least one segment"):
            VideoEdit(segments=[])

    def test_segments_stored_as_tuple(self):
        seg = SegmentConfig(source_video=Path(SMALL_VIDEO_PATH), start_second=0, end_second=2)
        edit = VideoEdit(segments=[seg])
        assert isinstance(edit.segments, tuple)
        assert len(edit.segments) == 1

    def test_defaults_for_post_ops(self):
        seg = SegmentConfig(source_video=Path(SMALL_VIDEO_PATH), start_second=0, end_second=2)
        edit = VideoEdit(segments=[seg])
        assert edit.post_transforms == ()
        assert edit.post_effects == ()


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


class TestExecution:
    def test_single_segment_no_ops(self, small_video):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=2,
        )
        edit = VideoEdit(segments=[seg])
        result = edit.run()
        assert isinstance(result, Video)
        assert result.total_seconds == pytest.approx(2.0, abs=0.2)

    def test_single_segment_with_transform(self, small_video):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            transforms=[Resize(width=400, height=250)],
        )
        edit = VideoEdit(segments=[seg])
        result = edit.run()
        assert result.frames.shape[2] == 400  # width
        assert result.frames.shape[1] == 250  # height

    def test_single_segment_with_effect(self, small_video):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            effects=[EffectApplication(effect=Blur(mode="constant", iterations=3), start=0.0, stop=1.0)],
        )
        edit = VideoEdit(segments=[seg])
        result = edit.run()
        assert isinstance(result, Video)

    def test_multi_segment_same_source(self, small_video):
        seg1 = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=2,
        )
        seg2 = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=4,
            end_second=6,
        )
        edit = VideoEdit(segments=[seg1, seg2])
        result = edit.run()
        assert result.total_seconds == pytest.approx(4.0, abs=0.3)

    def test_post_transform_applied(self, small_video):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=2,
        )
        edit = VideoEdit(
            segments=[seg],
            post_transforms=[Resize(width=320, height=200)],
        )
        result = edit.run()
        assert result.frames.shape[2] == 320
        assert result.frames.shape[1] == 200

    def test_post_effect_applied(self, small_video):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=2,
        )
        edit = VideoEdit(
            segments=[seg],
            post_effects=[EffectApplication(effect=ColorGrading(brightness=0.1))],
        )
        result = edit.run()
        assert isinstance(result, Video)

    def test_transforms_before_effects_in_segment(self, small_video):
        """Transforms change dimensions; effects run after on the resized video."""
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            transforms=[Resize(width=400, height=250)],
            effects=[EffectApplication(effect=Blur(mode="constant", iterations=2))],
        )
        edit = VideoEdit(segments=[seg])
        result = edit.run()
        assert result.frames.shape[2] == 400
        assert result.frames.shape[1] == 250


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_single_segment_returns_metadata(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=1,
            end_second=5,
        )
        edit = VideoEdit(segments=[seg])
        meta = edit.validate()
        assert isinstance(meta, VideoMetadata)
        assert meta.total_seconds == pytest.approx(4.0, abs=0.1)
        assert meta.width == SMALL_VIDEO_METADATA.width
        assert meta.height == SMALL_VIDEO_METADATA.height

    def test_valid_multi_segment_sums_duration(self):
        seg1 = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
        )
        seg2 = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=5,
            end_second=8,
        )
        edit = VideoEdit(segments=[seg1, seg2])
        meta = edit.validate()
        assert meta.total_seconds == pytest.approx(6.0, abs=0.2)

    def test_validation_with_resize_predicts_dimensions(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            transforms=[Resize(width=320, height=200)],
        )
        edit = VideoEdit(segments=[seg])
        meta = edit.validate()
        assert meta.width == 320
        assert meta.height == 200

    def test_validation_with_crop_predicts_dimensions(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            transforms=[Crop(width=400, height=300)],
        )
        edit = VideoEdit(segments=[seg])
        meta = edit.validate()
        assert meta.width == 400
        assert meta.height == 300

    def test_validation_with_crop_normalized_floats(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            transforms=[Crop(width=0.5, height=0.5)],
        )
        edit = VideoEdit(segments=[seg])
        meta = edit.validate()
        assert meta.width == SMALL_VIDEO_METADATA.width // 2
        assert meta.height == SMALL_VIDEO_METADATA.height // 2

    def test_validation_with_resample_fps(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            transforms=[ResampleFPS(fps=15)],
        )
        edit = VideoEdit(segments=[seg])
        meta = edit.validate()
        assert meta.fps == 15.0

    def test_validation_with_speed_change(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=4,
            transforms=[SpeedChange(speed=2.0)],
        )
        edit = VideoEdit(segments=[seg])
        meta = edit.validate()
        assert meta.total_seconds == pytest.approx(2.0, abs=0.1)

    def test_validation_with_post_transform(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
        )
        edit = VideoEdit(
            segments=[seg],
            post_transforms=[Resize(width=640, height=480)],
        )
        meta = edit.validate()
        assert meta.width == 640
        assert meta.height == 480

    def test_negative_start_raises(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=-1,
            end_second=3,
        )
        edit = VideoEdit(segments=[seg])
        with pytest.raises(ValueError, match="start_second.*must be >= 0"):
            edit.validate()

    def test_end_before_start_raises(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=5,
            end_second=3,
        )
        edit = VideoEdit(segments=[seg])
        with pytest.raises(ValueError, match="end_second.*must be > start_second"):
            edit.validate()

    def test_end_past_duration_raises(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=9999,
        )
        edit = VideoEdit(segments=[seg])
        with pytest.raises(ValueError, match="exceeds source duration"):
            edit.validate()

    def test_incompatible_segment_dimensions_raises(self):
        seg1 = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=2,
        )
        # big_video has different dimensions/fps
        seg2 = SegmentConfig(
            source_video=Path(BIG_VIDEO_PATH),
            start_second=0,
            end_second=2,
        )
        edit = VideoEdit(segments=[seg1, seg2])
        with pytest.raises(ValueError, match="(fps|dimensions)"):
            edit.validate()

    def test_incompatible_segment_fps_strict(self):
        """Validation must use exact fps equality, not rounded (matching Video.__add__)."""
        meta_a = VideoMetadata(height=100, width=100, fps=29.97, frame_count=90, total_seconds=3.0)
        meta_b = VideoMetadata(height=100, width=100, fps=30.0, frame_count=90, total_seconds=3.0)
        # Rounded fps would match, but exact does not
        assert meta_a.can_be_merged_with(meta_b)  # rounded check passes

        # Build a VideoEdit where segments predict these different fps values
        seg1 = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=2,
        )
        seg2 = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=2,
            end_second=4,
            transforms=[ResampleFPS(fps=23)],  # different from source fps=24
        )
        edit = VideoEdit(segments=[seg1, seg2])
        with pytest.raises(ValueError, match="fps"):
            edit.validate()

    def test_effect_start_past_duration_raises(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            effects=[EffectApplication(effect=Blur(mode="constant", iterations=3), start=10.0)],
        )
        edit = VideoEdit(segments=[seg])
        with pytest.raises(ValueError, match="start.*exceeds timeline duration"):
            edit.validate()

    def test_effect_stop_past_duration_raises(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            effects=[EffectApplication(effect=Blur(mode="constant", iterations=3), stop=10.0)],
        )
        edit = VideoEdit(segments=[seg])
        with pytest.raises(ValueError, match="stop.*exceeds timeline duration"):
            edit.validate()

    def test_effect_start_after_stop_raises(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            effects=[EffectApplication(effect=Blur(mode="constant", iterations=3), start=2.0, stop=1.0)],
        )
        edit = VideoEdit(segments=[seg])
        with pytest.raises(ValueError, match="start.*must be <= stop"):
            edit.validate()

    def test_post_effect_bounds_validated(self):
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
        )
        edit = VideoEdit(
            segments=[seg],
            post_effects=[EffectApplication(effect=Blur(mode="constant", iterations=3), start=100.0)],
        )
        with pytest.raises(ValueError, match="start.*exceeds timeline duration"):
            edit.validate()

    def test_unsupported_transform_raises(self):
        """A transform without metadata_method should fail validation."""
        from videopython.base.transforms import PictureInPicture

        # PictureInPicture has no metadata_method
        dummy_overlay = _make_synthetic_video(100, 100, 24, 1.0)
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
            transforms=[PictureInPicture(overlay=dummy_overlay)],
        )
        edit = VideoEdit(segments=[seg])
        with pytest.raises(ValueError, match="Metadata prediction is not supported"):
            edit.validate()

    def test_validate_does_not_load_video(self):
        """validate() should use VideoMetadata.from_path, not Video.from_path."""
        seg = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=3,
        )
        edit = VideoEdit(segments=[seg])

        with patch.object(Video, "from_path", side_effect=AssertionError("Video.from_path should not be called")):
            meta = edit.validate()
            assert isinstance(meta, VideoMetadata)


# ---------------------------------------------------------------------------
# Registry metadata_method tests (added here for convenience)
# ---------------------------------------------------------------------------


class TestRegistryMetadataMethod:
    @pytest.mark.parametrize(
        "op_id",
        ["cut", "cut_frames", "resize", "crop", "resample_fps", "speed_change"],
    )
    def test_base_transforms_have_metadata_method(self, op_id):
        from videopython.base.registry import get_operation_spec

        spec = get_operation_spec(op_id)
        assert spec is not None
        assert spec.metadata_method is not None
        assert spec.metadata_method == op_id

    def test_picture_in_picture_has_no_metadata_method(self):
        from videopython.base.registry import get_operation_spec

        spec = get_operation_spec("picture_in_picture")
        assert spec is not None
        assert spec.metadata_method is None

    @pytest.mark.parametrize(
        "op_id",
        ["cut", "cut_frames", "resize", "crop", "resample_fps", "speed_change"],
    )
    def test_metadata_method_exists_on_video_metadata(self, op_id):
        from videopython.base.registry import get_operation_spec

        spec = get_operation_spec(op_id)
        assert spec is not None
        assert hasattr(VideoMetadata, spec.metadata_method)


# ---------------------------------------------------------------------------
# VideoMetadata.speed_change tests
# ---------------------------------------------------------------------------


class TestSpeedChangeMetadata:
    def test_double_speed_halves_duration(self):
        meta = VideoMetadata(height=100, width=100, fps=24, frame_count=240, total_seconds=10.0)
        result = meta.speed_change(2.0)
        assert result.total_seconds == pytest.approx(5.0, abs=0.01)
        assert result.fps == 24

    def test_half_speed_doubles_duration(self):
        meta = VideoMetadata(height=100, width=100, fps=24, frame_count=240, total_seconds=10.0)
        result = meta.speed_change(0.5)
        assert result.total_seconds == pytest.approx(20.0, abs=0.01)

    def test_speed_preserves_dimensions(self):
        meta = VideoMetadata(height=200, width=300, fps=30, frame_count=300, total_seconds=10.0)
        result = meta.speed_change(1.5)
        assert result.height == 200
        assert result.width == 300

    def test_zero_speed_raises(self):
        meta = VideoMetadata(height=100, width=100, fps=24, frame_count=240, total_seconds=10.0)
        with pytest.raises(ValueError, match="must be positive"):
            meta.speed_change(0)

    def test_negative_speed_raises(self):
        meta = VideoMetadata(height=100, width=100, fps=24, frame_count=240, total_seconds=10.0)
        with pytest.raises(ValueError, match="must be positive"):
            meta.speed_change(-1.0)

    def test_frame_count_matches_runtime_semantics(self):
        """speed_change should use int(frame_count / speed) like SpeedChange.apply()."""
        meta = VideoMetadata(height=100, width=100, fps=24, frame_count=100, total_seconds=4.1667)
        result = meta.speed_change(3.0)
        # int(100 / 3.0) = 33, not round(4.1667 / 3 * 24) = 33.33 -> 33
        assert result.frame_count == 33

    def test_extreme_speed_zero_frames_raises(self):
        meta = VideoMetadata(height=100, width=100, fps=24, frame_count=1, total_seconds=0.0417)
        with pytest.raises(ValueError, match="0 frames"):
            meta.speed_change(100.0)


# ---------------------------------------------------------------------------
# Cache behavior tests
# ---------------------------------------------------------------------------


class TestSpecCache:
    def test_negative_lookup_not_cached(self):
        """Missing metadata_method should not be permanently cached."""
        from videopython.base.edit import _SPEC_CACHE, _get_metadata_method_for_class
        from videopython.base.transforms import PictureInPicture

        dummy = _make_synthetic_video(100, 100, 24, 1.0)
        pip = PictureInPicture(overlay=dummy)

        # First lookup: miss (PictureInPicture has no metadata_method)
        result = _get_metadata_method_for_class(pip)
        assert result is None

        # Verify it was NOT cached
        key = (PictureInPicture.__module__, PictureInPicture.__name__)
        assert key not in _SPEC_CACHE

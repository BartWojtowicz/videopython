"""Tests for VideoEdit JSON plan parsing, execution, and validation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.test_config import SMALL_VIDEO_METADATA, SMALL_VIDEO_PATH
from videopython.base.edit import SegmentConfig, VideoEdit, _StepRecord
from videopython.base.transforms import PictureInPicture
from videopython.base.video import Video, VideoMetadata


def _make_synthetic_video(width: int, height: int, fps: float, seconds: float) -> Video:
    frame_count = round(fps * seconds)
    frames = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    return Video(frames=frames, fps=fps)


def _segment_plan(
    *,
    source: str = SMALL_VIDEO_PATH,
    start: float = 0.0,
    end: float = 2.0,
    transforms: list[dict] | None = None,
    effects: list[dict] | None = None,
) -> dict:
    return {
        "source": source,
        "start": start,
        "end": end,
        "transforms": transforms or [],
        "effects": effects or [],
    }


class TestConstruction:
    def test_empty_segments_raises(self):
        with pytest.raises(ValueError, match="at least one segment"):
            VideoEdit(segments=[])

    def test_direct_record_based_construction(self):
        segment = SegmentConfig(source_video=Path(SMALL_VIDEO_PATH), start_second=0, end_second=1)
        edit = VideoEdit(segments=[segment])
        assert isinstance(edit.segments, tuple)
        assert len(edit.segments) == 1


class TestParsingAndSerialization:
    def test_from_dict_single_segment_no_ops(self):
        plan = {"segments": [_segment_plan(start=0.0, end=2.0)]}
        edit = VideoEdit.from_dict(plan)
        out = edit.to_dict()
        assert out["segments"][0]["source"] == SMALL_VIDEO_PATH
        assert out["segments"][0]["start"] == 0.0
        assert out["segments"][0]["end"] == 2.0
        assert out["segments"][0]["transforms"] == []
        assert out["segments"][0]["effects"] == []

    def test_from_json(self):
        plan = {"segments": [_segment_plan(start=0.0, end=1.0)]}
        edit = VideoEdit.from_json(json.dumps(plan))
        assert isinstance(edit, VideoEdit)
        assert edit.to_dict()["segments"][0]["end"] == 1.0

    def test_alias_canonicalized_in_to_dict(self):
        plan = {
            "segments": [
                _segment_plan(
                    effects=[{"op": "blur", "args": {"mode": "constant", "iterations": 1}}],
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        effect_step = edit.to_dict()["segments"][0]["effects"][0]
        assert effect_step["op"] == "blur_effect"

    def test_post_transforms_and_effects_roundtrip(self):
        plan = {
            "segments": [_segment_plan()],
            "post_transforms": [{"op": "resize", "args": {"width": 320, "height": 200}}],
            "post_effects": [{"op": "blur_effect", "args": {"mode": "constant", "iterations": 1}}],
        }
        edit = VideoEdit.from_dict(plan)
        out = edit.to_dict()
        assert out["post_transforms"][0]["op"] == "resize"
        assert out["post_effects"][0]["op"] == "blur_effect"

    def test_unknown_top_level_keys_ignored(self):
        plan = {"segments": [_segment_plan()], "future_key": {"x": 1}}
        edit = VideoEdit.from_dict(plan)
        out = edit.to_dict()
        assert "future_key" not in out

    def test_segment_omits_transforms_and_effects_keys(self):
        plan = {"segments": [{"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 1.0}]}
        edit = VideoEdit.from_dict(plan)
        out = edit.to_dict()
        assert out["segments"][0]["transforms"] == []
        assert out["segments"][0]["effects"] == []

    def test_to_dict_returns_deep_copies(self):
        plan = {
            "segments": [
                _segment_plan(
                    transforms=[{"op": "resize", "args": {"width": 320, "height": 200}}],
                    effects=[
                        {
                            "op": "blur_effect",
                            "args": {"mode": "constant", "iterations": 1},
                            "apply": {"start": 0.1},
                        }
                    ],
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        out1 = edit.to_dict()
        out1["segments"][0]["transforms"][0]["args"]["width"] = 999
        out1["segments"][0]["effects"][0]["apply"]["start"] = 999
        out2 = edit.to_dict()
        assert out2["segments"][0]["transforms"][0]["args"]["width"] == 320
        assert out2["segments"][0]["effects"][0]["apply"]["start"] == 0.1

    def test_step_record_snapshot_semantics(self):
        plan = {
            "segments": [
                _segment_plan(
                    transforms=[{"op": "resize", "args": {"width": 320, "height": 200}}],
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        # Mutate live object; to_dict should still emit parsed snapshot.
        resize_op = edit.segments[0].transform_records[0].operation
        resize_op.width = 123  # type: ignore[attr-defined]
        assert edit.to_dict()["segments"][0]["transforms"][0]["args"]["width"] == 320


class TestExecution:
    def test_from_dict_run(self):
        plan = {"segments": [_segment_plan(start=0.0, end=2.0)]}
        result = VideoEdit.from_dict(plan).run()
        assert isinstance(result, Video)
        assert result.total_seconds == pytest.approx(2.0, abs=0.25)

    def test_run_with_transform_and_effect(self):
        plan = {
            "segments": [
                _segment_plan(
                    start=0.0,
                    end=2.0,
                    transforms=[{"op": "resize", "args": {"width": 400, "height": 250}}],
                    effects=[
                        {
                            "op": "blur_effect",
                            "args": {"mode": "constant", "iterations": 1},
                            "apply": {"start": 0.0},
                        }
                    ],
                )
            ]
        }
        result = VideoEdit.from_dict(plan).run()
        assert result.frames.shape[2] == 400
        assert result.frames.shape[1] == 250

    def test_multi_segment_same_source_run(self):
        plan = {
            "segments": [
                _segment_plan(start=0.0, end=2.0),
                _segment_plan(start=4.0, end=6.0),
            ]
        }
        result = VideoEdit.from_dict(plan).run()
        assert result.total_seconds == pytest.approx(4.0, abs=0.3)


class TestValidation:
    def test_validate_from_dict(self):
        plan = {"segments": [_segment_plan(start=1.0, end=5.0)]}
        meta = VideoEdit.from_dict(plan).validate()
        assert isinstance(meta, VideoMetadata)
        assert meta.total_seconds == pytest.approx(4.0, abs=0.1)
        assert meta.width == SMALL_VIDEO_METADATA.width
        assert meta.height == SMALL_VIDEO_METADATA.height

    def test_validate_resize_and_crop_normalized(self):
        plan = {
            "segments": [
                _segment_plan(
                    start=0.0,
                    end=3.0,
                    transforms=[
                        {"op": "crop", "args": {"width": 0.5, "height": 0.5}},
                        {"op": "resize", "args": {"width": 320, "height": 200}},
                    ],
                )
            ]
        }
        meta = VideoEdit.from_dict(plan).validate()
        assert meta.width == 320
        assert meta.height == 200

    def test_validate_speed_change(self):
        plan = {
            "segments": [
                _segment_plan(
                    start=0.0,
                    end=4.0,
                    transforms=[{"op": "speed_change", "args": {"speed": 2.0}}],
                )
            ]
        }
        meta = VideoEdit.from_dict(plan).validate()
        assert meta.total_seconds == pytest.approx(2.0, abs=0.2)

    def test_validate_does_not_load_video_frames(self):
        plan = {"segments": [_segment_plan(start=0.0, end=3.0)]}
        edit = VideoEdit.from_dict(plan)
        with patch.object(Video, "from_path", side_effect=AssertionError("Video.from_path should not be called")):
            meta = edit.validate()
            assert isinstance(meta, VideoMetadata)

    def test_incompatible_fps_strict(self):
        plan = {
            "segments": [
                _segment_plan(start=0.0, end=2.0),
                _segment_plan(start=2.0, end=4.0, transforms=[{"op": "resample_fps", "args": {"fps": 23}}]),
            ]
        }
        with pytest.raises(ValueError, match="fps"):
            VideoEdit.from_dict(plan).validate()

    def test_effect_bounds_validated(self):
        plan = {
            "segments": [
                _segment_plan(
                    start=0.0,
                    end=3.0,
                    effects=[
                        {
                            "op": "blur_effect",
                            "args": {"mode": "constant", "iterations": 1},
                            "apply": {"start": 10},
                        }
                    ],
                )
            ]
        }
        with pytest.raises(ValueError, match="exceeds timeline duration"):
            VideoEdit.from_dict(plan).validate()

    def test_unsupported_transform_metadata_via_direct_record(self):
        dummy_overlay = _make_synthetic_video(100, 100, 24, 1.0)
        pip = PictureInPicture(overlay=dummy_overlay)
        segment = SegmentConfig(
            source_video=Path(SMALL_VIDEO_PATH),
            start_second=0.0,
            end_second=2.0,
            transform_records=(_StepRecord.create("picture_in_picture", {}, {}, pip),),
        )
        with pytest.raises(ValueError, match="Metadata prediction is not supported"):
            VideoEdit(segments=[segment]).validate()


class TestParsingErrors:
    def test_from_json_invalid_json_wrapped(self):
        with pytest.raises(ValueError, match="Invalid VideoEdit JSON"):
            VideoEdit.from_json("{bad json")

    def test_from_dict_non_dict_input(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            VideoEdit.from_dict([])  # type: ignore[arg-type]

    def test_missing_segments_key(self):
        with pytest.raises(ValueError, match="missing required key 'segments'"):
            VideoEdit.from_dict({})

    def test_empty_segments_list(self):
        with pytest.raises(ValueError, match="must not be empty"):
            VideoEdit.from_dict({"segments": []})

    def test_unknown_segment_key(self):
        with pytest.raises(ValueError, match="unknown keys"):
            VideoEdit.from_dict({"segments": [{"source": SMALL_VIDEO_PATH, "start": 0, "end": 1, "tranforms": []}]})

    def test_non_numeric_start(self):
        with pytest.raises(ValueError, match="must be a number"):
            VideoEdit.from_dict({"segments": [_segment_plan(start="0", end=1.0)]})  # type: ignore[arg-type]

    def test_missing_op_key(self):
        with pytest.raises(ValueError, match="missing required key 'op'"):
            VideoEdit.from_dict({"segments": [_segment_plan(transforms=[{"args": {}}])]})

    def test_unknown_step_key(self):
        with pytest.raises(ValueError, match="unknown keys"):
            VideoEdit.from_dict({"segments": [_segment_plan(transforms=[{"op": "resize", "foo": 1}])]})

    def test_transform_step_apply_rejected(self):
        with pytest.raises(ValueError, match="transforms do not accept apply params"):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment_plan(
                            transforms=[{"op": "resize", "args": {"width": 1}, "apply": {"start": 0}}]
                        )
                    ]
                }
            )

    def test_unknown_operation_has_ai_hint(self):
        with pytest.raises(ValueError, match="import videopython.ai"):
            VideoEdit.from_dict({"segments": [_segment_plan(transforms=[{"op": "face_crop"}])]})

    def test_category_mismatch_effect_in_transforms(self):
        with pytest.raises(ValueError, match="Expected transformation operation, got effect"):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment_plan(
                            transforms=[
                                {"op": "blur_effect", "args": {"mode": "constant", "iterations": 1}}
                            ]
                        )
                    ]
                }
            )

    def test_transition_rejected(self):
        with pytest.raises(ValueError, match="Expected transformation operation, got transition"):
            VideoEdit.from_dict({"segments": [_segment_plan(transforms=[{"op": "fade_transition"}])]})

    def test_multi_source_tag_rejected(self):
        with pytest.raises(ValueError, match="tag 'multi_source'"):
            VideoEdit.from_dict({"segments": [_segment_plan(transforms=[{"op": "picture_in_picture"}])]})

    def test_non_json_instantiable_rejected(self):
        with pytest.raises(ValueError, match="not JSON-instantiable"):
            VideoEdit.from_dict({"segments": [_segment_plan(effects=[{"op": "ken_burns"}])]})

    def test_non_json_instantiable_precedes_arg_validation(self):
        with pytest.raises(ValueError, match="not JSON-instantiable"):
            VideoEdit.from_dict(
                {"segments": [_segment_plan(effects=[{"op": "ken_burns", "args": {"bogus": 1}}])]}
            )

    def test_unknown_apply_arg_rejected(self):
        with pytest.raises(ValueError, match="unknown keys"):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment_plan(
                            effects=[
                                {
                                    "op": "blur_effect",
                                    "args": {"mode": "constant", "iterations": 1},
                                    "apply": {"foo": 1},
                                }
                            ]
                        )
                    ]
                }
            )

    def test_apply_arg_type_rejected_at_parse_time(self):
        with pytest.raises(ValueError, match=r"\.apply\.start must be a number"):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment_plan(
                            effects=[
                                {
                                    "op": "blur_effect",
                                    "args": {"mode": "constant", "iterations": 1},
                                    "apply": {"start": "abc"},
                                }
                            ]
                        )
                    ]
                }
            )


class TestRegistryAndMetadata:
    @pytest.mark.parametrize("op_id", ["cut", "cut_frames", "resize", "crop", "resample_fps", "speed_change"])
    def test_base_transforms_have_metadata_method(self, op_id):
        from videopython.base.registry import get_operation_spec

        spec = get_operation_spec(op_id)
        assert spec is not None
        assert spec.metadata_method == op_id

    def test_picture_in_picture_has_no_metadata_method(self):
        from videopython.base.registry import get_operation_spec

        spec = get_operation_spec("picture_in_picture")
        assert spec is not None
        assert spec.metadata_method is None

    def test_speed_change_metadata_runtime_semantics(self):
        meta = VideoMetadata(height=100, width=100, fps=24, frame_count=100, total_seconds=4.1667)
        result = meta.speed_change(3.0)
        assert result.frame_count == 33

    def test_speed_change_zero_frames_raises(self):
        meta = VideoMetadata(height=100, width=100, fps=24, frame_count=1, total_seconds=0.0417)
        with pytest.raises(ValueError, match="0 frames"):
            meta.speed_change(100.0)

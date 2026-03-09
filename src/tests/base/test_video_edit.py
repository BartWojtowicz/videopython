"""Tests for VideoEdit JSON plan parsing, execution, and validation."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.test_config import SMALL_VIDEO_METADATA, SMALL_VIDEO_PATH
from videopython.base.edit import SegmentConfig, VideoEdit, _StepRecord
from videopython.base.text.transcription import Transcription, TranscriptionWord
from videopython.base.transforms import CropMode, PictureInPicture
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


def _schema_step_ops(items_schema: dict) -> set[str]:
    return {step["properties"]["op"]["const"] for step in items_schema["oneOf"]}


class TestConstruction:
    def test_empty_segments_raises(self):
        with pytest.raises(ValueError, match="at least one segment"):
            VideoEdit(segments=[])

    def test_direct_record_based_construction(self):
        segment = SegmentConfig(source_video=Path(SMALL_VIDEO_PATH), start_second=0, end_second=1)
        edit = VideoEdit(segments=[segment])
        assert isinstance(edit.segments, tuple)
        assert len(edit.segments) == 1

    def test_step_record_apply_args_contract_rejects_non_numeric_start_stop(self):
        with pytest.raises(TypeError, match="must be numeric or None"):
            _StepRecord.create(
                "blur_effect",
                {},
                {"start": "bad"},
                object(),  # type: ignore[arg-type]
            )


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

    def test_constructor_enum_arg_is_normalized_for_crop(self):
        plan = {
            "segments": [
                _segment_plan(transforms=[{"op": "crop", "args": {"width": 0.5, "height": 0.5, "mode": "center"}}])
            ]
        }
        edit = VideoEdit.from_dict(plan)
        crop_op = edit.segments[0].transform_records[0].operation
        assert getattr(crop_op, "mode") is CropMode.CENTER
        # Snapshot stays JSON-native/canonical
        assert edit.to_dict()["segments"][0]["transforms"][0]["args"]["mode"] == "center"

    def test_constructor_tuple_arg_is_normalized_but_snapshot_stays_list(self):
        plan = {
            "segments": [
                _segment_plan(
                    effects=[
                        {
                            "op": "blur_effect",
                            "args": {"mode": "constant", "iterations": 1, "kernel_size": [5, 5]},
                        }
                    ]
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        blur_op = edit.segments[0].effect_records[0].operation
        assert getattr(blur_op, "kernel_size") == (5, 5)
        assert isinstance(getattr(blur_op, "kernel_size"), tuple)
        assert edit.to_dict()["segments"][0]["effects"][0]["args"]["kernel_size"] == [5, 5]


class TestJsonSchema:
    def test_json_schema_top_level_and_segment_shapes(self):
        schema = VideoEdit.json_schema()
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["type"] == "object"
        assert schema["required"] == ["segments"]
        assert "additionalProperties" not in schema

        segment_schema = schema["properties"]["segments"]["items"]
        assert segment_schema["type"] == "object"
        assert segment_schema["additionalProperties"] is False
        assert set(segment_schema["required"]) == {"source", "start", "end"}
        assert schema["properties"]["segments"]["minItems"] == 1

    def test_json_schema_uses_canonical_ops_and_excludes_unsupported(self):
        schema = VideoEdit.json_schema()
        transform_ops = _schema_step_ops(schema["properties"]["post_transforms"]["items"])
        effect_ops = _schema_step_ops(schema["properties"]["post_effects"]["items"])

        assert "resize" in transform_ops
        assert "cut" in transform_ops
        assert "blur_effect" in effect_ops
        assert "blur" not in effect_ops  # alias should not be emitted

        assert "fade_transition" not in transform_ops
        assert "picture_in_picture" not in transform_ops
        assert "ken_burns" not in effect_ops
        assert "full_image_overlay" not in effect_ops

    def test_json_schema_step_shapes_match_parser_rules(self):
        schema = VideoEdit.json_schema()
        transform_steps = schema["properties"]["post_transforms"]["items"]["oneOf"]
        effect_steps = schema["properties"]["post_effects"]["items"]["oneOf"]

        cut_step = next(step for step in transform_steps if step["properties"]["op"]["const"] == "cut")
        resize_step = next(step for step in transform_steps if step["properties"]["op"]["const"] == "resize")
        blur_step = next(step for step in effect_steps if step["properties"]["op"]["const"] == "blur_effect")

        assert cut_step["additionalProperties"] is False
        assert "apply" not in cut_step["properties"]
        assert "args" in cut_step["required"]  # cut requires start/end

        assert "apply" in blur_step["properties"]
        assert "apply" not in blur_step["required"]  # effect apply params are optional
        assert blur_step["properties"]["apply"]["properties"]["start"]["type"] == ["number", "null"]

        resize_step = next(step for step in transform_steps if step["properties"]["op"]["const"] == "resize")
        assert "args" in resize_step["required"]
        assert "anyOf" in resize_step["properties"]["args"]
        assert resize_step["properties"]["args"]["anyOf"][0]["required"] == ["width"]
        assert resize_step["properties"]["args"]["anyOf"][1]["required"] == ["height"]


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

    def test_validate_crop_matches_runtime_for_odd_center_crop(self):
        plan = {
            "segments": [
                _segment_plan(
                    start=0.0,
                    end=2.0,
                    transforms=[{"op": "crop", "args": {"width": 5, "height": 5, "mode": "center"}}],
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        meta = edit.validate()
        video = edit.run()
        assert meta.width == video.frame_shape[1]
        assert meta.height == video.frame_shape[0]

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

    @pytest.mark.parametrize("op_id", ["face_crop"])
    def test_validate_ai_aspect_crop_transforms(self, op_id):
        importlib.import_module("videopython.ai")
        importlib.reload(importlib.import_module("videopython.ai.registry"))

        plan = {
            "segments": [
                _segment_plan(
                    start=0.0,
                    end=3.0,
                    transforms=[{"op": op_id, "args": {"target_aspect": [9, 16]}}],
                )
            ]
        }

        meta = VideoEdit.from_dict(plan).validate()
        cut_meta = SMALL_VIDEO_METADATA.cut(0.0, 3.0)

        target_ratio = 9 / 16
        if target_ratio < cut_meta.width / cut_meta.height:
            expected_height = cut_meta.height - (cut_meta.height % 2)
            expected_width = int(expected_height * target_ratio)
            expected_width -= expected_width % 2
        else:
            expected_width = cut_meta.width - (cut_meta.width % 2)
            expected_height = int(expected_width / target_ratio)
            expected_height -= expected_height % 2

        assert meta.width == expected_width
        assert meta.height == expected_height
        assert meta.frame_count == cut_meta.frame_count
        assert meta.fps == cut_meta.fps

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


class TestValidateWithMetadata:
    def test_matches_validate_for_single_source(self):
        plan = {
            "segments": [
                _segment_plan(start=1.0, end=5.0),
                _segment_plan(start=0.0, end=3.0),
            ]
        }
        edit = VideoEdit.from_dict(plan)
        expected = edit.validate()
        result = edit.validate_with_metadata(SMALL_VIDEO_METADATA)
        assert result.total_seconds == pytest.approx(expected.total_seconds, abs=0.01)
        assert result.width == expected.width
        assert result.height == expected.height
        assert result.fps == expected.fps
        assert result.frame_count == expected.frame_count

    def test_with_transforms_and_effects(self):
        plan = {
            "segments": [
                _segment_plan(
                    start=0.0,
                    end=4.0,
                    transforms=[{"op": "speed_change", "args": {"speed": 2.0}}],
                    effects=[
                        {"op": "blur_effect", "args": {"mode": "constant", "iterations": 1}, "apply": {"stop": 1.5}}
                    ],
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        expected = edit.validate()
        result = edit.validate_with_metadata(SMALL_VIDEO_METADATA)
        assert result.total_seconds == pytest.approx(expected.total_seconds, abs=0.01)

    def test_effect_bounds_exceeded_after_speed_change(self):
        plan = {
            "segments": [
                _segment_plan(
                    start=0.0,
                    end=6.0,
                    transforms=[{"op": "speed_change", "args": {"speed": 1.5}}],
                    effects=[
                        {
                            "op": "volume_adjust",
                            "args": {"volume": 0.5},
                            "apply": {"stop": 6.0},
                        }
                    ],
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        with pytest.raises(ValueError, match="exceeds timeline duration"):
            edit.validate_with_metadata(SMALL_VIDEO_METADATA)

    def test_end_exceeds_source_duration(self):
        plan = {"segments": [_segment_plan(start=0.0, end=99.0)]}
        edit = VideoEdit.from_dict(plan)
        with pytest.raises(ValueError, match="exceeds source duration"):
            edit.validate_with_metadata(SMALL_VIDEO_METADATA)

    def test_multi_source_dict(self):
        plan = {
            "segments": [
                _segment_plan(source="video_a.mp4", start=0.0, end=2.0),
                _segment_plan(source="video_b.mp4", start=0.0, end=3.0),
            ]
        }
        meta_a = VideoMetadata(height=500, width=800, fps=24, frame_count=48, total_seconds=2.0)
        meta_b = VideoMetadata(height=500, width=800, fps=24, frame_count=72, total_seconds=3.0)
        edit = VideoEdit.from_dict(plan)
        result = edit.validate_with_metadata({"video_a.mp4": meta_a, "video_b.mp4": meta_b})
        assert result.total_seconds == pytest.approx(5.0, abs=0.01)

    def test_missing_source_in_dict_raises(self):
        plan = {
            "segments": [
                _segment_plan(source="video_a.mp4", start=0.0, end=2.0),
                _segment_plan(source="video_b.mp4", start=0.0, end=3.0),
            ]
        }
        meta_a = VideoMetadata(height=500, width=800, fps=24, frame_count=48, total_seconds=2.0)
        edit = VideoEdit.from_dict(plan)
        with pytest.raises(ValueError, match="no metadata provided for source"):
            edit.validate_with_metadata({"video_a.mp4": meta_a})

    def test_does_not_touch_disk(self):
        plan = {"segments": [_segment_plan(source="nonexistent_file.mp4", start=0.0, end=2.0)]}
        meta = VideoMetadata(height=500, width=800, fps=24, frame_count=48, total_seconds=5.0)
        edit = VideoEdit.from_dict(plan)
        result = edit.validate_with_metadata(meta)
        assert isinstance(result, VideoMetadata)


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
                        _segment_plan(transforms=[{"op": "resize", "args": {"width": 1}, "apply": {"start": 0}}])
                    ]
                }
            )

    def test_unknown_operation_has_ai_hint(self):
        for name in list(sys.modules):
            if name == "videopython.ai" or name.startswith("videopython.ai."):
                sys.modules.pop(name)
        importlib.reload(importlib.import_module("videopython.base.registry"))

        with pytest.raises(ValueError, match="import videopython.ai"):
            VideoEdit.from_dict({"segments": [_segment_plan(transforms=[{"op": "face_crop"}])]})

    def test_category_mismatch_effect_in_transforms(self):
        with pytest.raises(ValueError, match="Expected transformation operation, got effect"):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment_plan(transforms=[{"op": "blur_effect", "args": {"mode": "constant", "iterations": 1}}])
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
            VideoEdit.from_dict({"segments": [_segment_plan(effects=[{"op": "ken_burns", "args": {"bogus": 1}}])]})

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

    def test_nullable_apply_arg_accepts_none(self):
        edit = VideoEdit.from_dict(
            {
                "segments": [
                    _segment_plan(
                        effects=[
                            {
                                "op": "blur_effect",
                                "args": {"mode": "constant", "iterations": 1},
                                "apply": {"start": None},
                            }
                        ]
                    )
                ]
            }
        )
        assert edit.to_dict()["segments"][0]["effects"][0]["apply"]["start"] is None

    def test_bool_rejected_for_integer_param(self):
        with pytest.raises(ValueError, match=r"\.args\.width must be an integer"):
            VideoEdit.from_dict(
                {"segments": [_segment_plan(transforms=[{"op": "resize", "args": {"width": True, "height": 100}}])]}
            )

    def test_invalid_enum_value_rejected_at_parse_time(self):
        with pytest.raises(ValueError, match=r"\.args\.mode must be one of"):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment_plan(
                            transforms=[{"op": "crop", "args": {"width": 10, "height": 10, "mode": "not_a_mode"}}]
                        )
                    ]
                }
            )

    def test_crop_width_must_be_positive(self):
        with pytest.raises(ValueError, match=r"\.args\.width must be > 0"):
            VideoEdit.from_dict(
                {"segments": [_segment_plan(transforms=[{"op": "crop", "args": {"width": 0, "height": 10}}])]}
            )

    def test_resize_requires_width_or_height_at_parse_time(self):
        with pytest.raises(ValueError, match="must include at least one non-null value"):
            VideoEdit.from_dict({"segments": [_segment_plan(transforms=[{"op": "resize"}])]})

    def test_resize_rejects_all_null_dimensions_at_parse_time(self):
        with pytest.raises(ValueError, match="must include at least one non-null value"):
            VideoEdit.from_dict(
                {"segments": [_segment_plan(transforms=[{"op": "resize", "args": {"width": None, "height": None}}])]}
            )

    def test_speed_change_zero_rejected_by_exclusive_minimum(self):
        with pytest.raises(ValueError, match=r"\.args\.speed must be > 0"):
            VideoEdit.from_dict(
                {"segments": [_segment_plan(transforms=[{"op": "speed_change", "args": {"speed": 0}}])]}
            )

    def test_array_item_type_validation_for_blur_kernel_size(self):
        with pytest.raises(ValueError, match=r"\.args\.kernel_size\[1\] must be a number"):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment_plan(
                            effects=[
                                {
                                    "op": "blur_effect",
                                    "args": {
                                        "mode": "constant",
                                        "iterations": 1,
                                        "kernel_size": [5, "bad"],
                                    },
                                }
                            ]
                        )
                    ]
                }
            )

    def test_numeric_minimum_validation_for_blur_iterations(self):
        with pytest.raises(ValueError, match=r"\.args\.iterations must be >= 1"):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment_plan(
                            effects=[
                                {
                                    "op": "blur_effect",
                                    "args": {"mode": "constant", "iterations": 0},
                                }
                            ]
                        )
                    ]
                }
            )

    def test_numeric_minimum_validation_for_effect_apply_start(self):
        with pytest.raises(ValueError, match=r"\.apply\.start must be >= 0"):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment_plan(
                            effects=[
                                {
                                    "op": "blur_effect",
                                    "args": {"mode": "constant", "iterations": 1},
                                    "apply": {"start": -1},
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

    def test_cut_metadata_runtime_semantics_with_fractional_seconds(self):
        meta = VideoMetadata(height=100, width=100, fps=10, frame_count=10, total_seconds=1.0)
        result = meta.cut(0.05, 0.15)
        assert result.frame_count == 2
        assert result.total_seconds == pytest.approx(0.2)

    @pytest.mark.parametrize("op_id", ["reverse", "freeze_frame", "silence_removal"])
    def test_new_transforms_have_metadata_method(self, op_id):
        from videopython.base.registry import get_operation_spec

        spec = get_operation_spec(op_id)
        assert spec is not None
        assert spec.metadata_method == op_id


class TestReverseMetadata:
    def test_reverse_preserves_all_metadata(self):
        meta = VideoMetadata(height=200, width=300, fps=30, frame_count=150, total_seconds=5.0)
        result = meta.reverse()
        assert result.height == meta.height
        assert result.width == meta.width
        assert result.fps == meta.fps
        assert result.frame_count == meta.frame_count
        assert result.total_seconds == meta.total_seconds

    def test_validate_plan_with_reverse_in_segment(self):
        plan = {
            "segments": [
                _segment_plan(
                    start=0.0,
                    end=3.0,
                    transforms=[{"op": "reverse"}],
                )
            ]
        }
        meta = VideoEdit.from_dict(plan).validate()
        assert isinstance(meta, VideoMetadata)
        assert meta.total_seconds == pytest.approx(3.0, abs=0.1)

    def test_validate_plan_with_reverse_in_post_transforms(self):
        plan = {
            "segments": [_segment_plan(start=0.0, end=3.0)],
            "post_transforms": [{"op": "reverse"}],
        }
        meta = VideoEdit.from_dict(plan).validate()
        assert isinstance(meta, VideoMetadata)
        assert meta.total_seconds == pytest.approx(3.0, abs=0.1)

    def test_validate_with_metadata_reverse(self):
        plan = {
            "segments": [
                _segment_plan(
                    source="fake.mp4",
                    start=0.0,
                    end=2.0,
                    transforms=[{"op": "reverse"}],
                )
            ],
            "post_transforms": [{"op": "reverse"}],
        }
        source_meta = VideoMetadata(height=500, width=800, fps=24, frame_count=120, total_seconds=5.0)
        meta = VideoEdit.from_dict(plan).validate_with_metadata(source_meta)
        assert isinstance(meta, VideoMetadata)


class TestFreezeFrameMetadata:
    def test_freeze_frame_after(self):
        meta = VideoMetadata(height=100, width=200, fps=30, frame_count=300, total_seconds=10.0)
        result = meta.freeze_frame(timestamp=5.0, duration=2.0, position="after")
        expected_added = round(2.0 * 30)
        assert result.frame_count == 300 + expected_added

    def test_freeze_frame_before(self):
        meta = VideoMetadata(height=100, width=200, fps=30, frame_count=300, total_seconds=10.0)
        result = meta.freeze_frame(timestamp=5.0, duration=2.0, position="before")
        expected_added = round(2.0 * 30)
        assert result.frame_count == 300 + expected_added

    def test_freeze_frame_replace(self):
        meta = VideoMetadata(height=100, width=200, fps=30, frame_count=300, total_seconds=10.0)
        result = meta.freeze_frame(timestamp=5.0, duration=2.0, position="replace")
        # replace_end = min(150 + 60, 300) = 210, replaced = 60
        # new = 300 - 60 + 60 = 300
        assert result.frame_count == 300

    def test_freeze_frame_replace_at_end_clamps(self):
        meta = VideoMetadata(height=100, width=200, fps=30, frame_count=300, total_seconds=10.0)
        result = meta.freeze_frame(timestamp=9.0, duration=5.0, position="replace")
        # frame_idx = round(9.0 * 30) = 270, freeze_count = round(5.0 * 30) = 150
        # replace_end = min(270 + 150, 300) = 300, replaced = 30
        # new = 300 - 30 + 150 = 420
        assert result.frame_count == 420

    def test_freeze_frame_timestamp_validation(self):
        meta = VideoMetadata(height=100, width=200, fps=30, frame_count=300, total_seconds=10.0)
        with pytest.raises(ValueError, match="timestamp.*must be less than"):
            meta.freeze_frame(timestamp=10.0, duration=1.0)
        with pytest.raises(ValueError, match="timestamp must be >= 0"):
            meta.freeze_frame(timestamp=-1.0, duration=1.0)
        with pytest.raises(ValueError, match="duration must be > 0"):
            meta.freeze_frame(timestamp=1.0, duration=0.0)

    def test_validate_plan_with_freeze_frame(self):
        plan = {
            "segments": [
                _segment_plan(
                    start=0.0,
                    end=3.0,
                    transforms=[{"op": "freeze_frame", "args": {"timestamp": 1.0, "duration": 2.0}}],
                )
            ]
        }
        meta = VideoEdit.from_dict(plan).validate()
        assert isinstance(meta, VideoMetadata)
        assert meta.total_seconds > 3.0

    def test_validate_with_metadata_freeze_frame(self):
        plan = {
            "segments": [
                _segment_plan(
                    source="fake.mp4",
                    start=0.0,
                    end=3.0,
                    transforms=[
                        {"op": "freeze_frame", "args": {"timestamp": 1.0, "duration": 1.5, "position": "after"}}
                    ],
                )
            ]
        }
        source_meta = VideoMetadata(height=500, width=800, fps=24, frame_count=120, total_seconds=5.0)
        meta = VideoEdit.from_dict(plan).validate_with_metadata(source_meta)
        assert meta.total_seconds > 3.0


def _make_transcription(words_data: list[tuple[float, float, str]]) -> Transcription:
    words = [TranscriptionWord(start=s, end=e, word=w) for s, e, w in words_data]
    return Transcription(words=words)


class TestSilenceRemovalMetadata:
    def test_silence_removal_cut_mode(self):
        # 10s video at 30fps = 300 frames
        # Words at 2-3s and 7-8s with padding=0.15 -> speech: [1.85, 3.15] and [6.85, 8.15]
        # Silence gaps >= 1.0s: [0, 1.85] (1.85s), [3.15, 6.85] (3.7s), [8.15, 10.0] (1.85s)
        meta = VideoMetadata(height=100, width=200, fps=30, frame_count=300, total_seconds=10.0)
        transcription = _make_transcription([(2.0, 3.0, "hello"), (7.0, 8.0, "world")])
        result = meta.silence_removal(min_silence_duration=1.0, padding=0.15, mode="cut", transcription=transcription)
        assert result.frame_count < meta.frame_count
        assert result.height == meta.height
        assert result.width == meta.width
        assert result.fps == meta.fps

    def test_silence_removal_speed_up_mode(self):
        meta = VideoMetadata(height=100, width=200, fps=30, frame_count=300, total_seconds=10.0)
        transcription = _make_transcription([(2.0, 3.0, "hello"), (7.0, 8.0, "world")])
        result = meta.silence_removal(
            min_silence_duration=1.0, padding=0.15, mode="speed_up", speed_factor=3.0, transcription=transcription
        )
        assert result.frame_count < meta.frame_count
        # speed_up removes less than cut
        result_cut = meta.silence_removal(
            min_silence_duration=1.0, padding=0.15, mode="cut", transcription=transcription
        )
        assert result.frame_count > result_cut.frame_count

    def test_silence_removal_no_transcription_returns_self(self):
        meta = VideoMetadata(height=100, width=200, fps=30, frame_count=300, total_seconds=10.0)
        result = meta.silence_removal(transcription=None)
        assert result.frame_count == meta.frame_count

    def test_silence_removal_no_silence_gaps(self):
        # Continuous speech covering the whole video -> no gaps
        meta = VideoMetadata(height=100, width=200, fps=30, frame_count=300, total_seconds=10.0)
        transcription = _make_transcription([(0.0, 10.0, "continuous speech")])
        result = meta.silence_removal(min_silence_duration=1.0, padding=0.15, mode="cut", transcription=transcription)
        assert result.frame_count == meta.frame_count

    def test_validate_plan_with_silence_removal_and_context(self):
        transcription = _make_transcription([(1.0, 2.0, "hello"), (4.0, 5.0, "world")])
        plan = {
            "segments": [
                _segment_plan(
                    source="fake.mp4",
                    start=0.0,
                    end=6.0,
                    transforms=[{"op": "silence_removal", "args": {"min_silence_duration": 0.5}}],
                )
            ]
        }
        source_meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        meta = VideoEdit.from_dict(plan).validate_with_metadata(source_meta, context={"transcription": transcription})
        assert isinstance(meta, VideoMetadata)
        assert meta.frame_count < round(6.0 * 24)

    def test_validate_plan_silence_removal_no_context_raises(self):
        plan = {
            "segments": [
                _segment_plan(
                    source="fake.mp4",
                    start=0.0,
                    end=6.0,
                    transforms=[{"op": "silence_removal"}],
                )
            ]
        }
        source_meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        with pytest.raises(ValueError, match="requires transcription context"):
            VideoEdit.from_dict(plan).validate_with_metadata(source_meta)

    def test_validate_plan_silence_removal_in_post_transforms(self):
        transcription = _make_transcription([(1.0, 2.0, "hello")])
        plan = {
            "segments": [_segment_plan(source="fake.mp4", start=0.0, end=4.0)],
            "post_transforms": [{"op": "silence_removal"}],
        }
        source_meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        meta = VideoEdit.from_dict(plan).validate_with_metadata(source_meta, context={"transcription": transcription})
        assert isinstance(meta, VideoMetadata)

    def test_silence_removal_metadata_matches_apply_cut(self):
        fps = 24.0
        frame_count = 240
        total_seconds = frame_count / fps
        meta = VideoMetadata(height=100, width=200, fps=fps, frame_count=frame_count, total_seconds=total_seconds)
        transcription = _make_transcription([(1.0, 2.0, "hello"), (5.0, 6.0, "world")])

        from videopython.base.transforms import SilenceRemoval

        video = _make_synthetic_video(200, 100, fps, total_seconds)
        op = SilenceRemoval(min_silence_duration=1.0, padding=0.15, mode="cut")
        result_video = op.apply(video, transcription=transcription)
        predicted = meta.silence_removal(
            min_silence_duration=1.0, padding=0.15, mode="cut", transcription=transcription
        )
        assert predicted.frame_count == len(result_video.frames)

    def test_silence_removal_metadata_matches_apply_speed_up(self):
        fps = 24.0
        frame_count = 240
        total_seconds = frame_count / fps
        meta = VideoMetadata(height=100, width=200, fps=fps, frame_count=frame_count, total_seconds=total_seconds)
        transcription = _make_transcription([(1.0, 2.0, "hello"), (5.0, 6.0, "world")])

        from videopython.base.transforms import SilenceRemoval

        video = _make_synthetic_video(200, 100, fps, total_seconds)
        op = SilenceRemoval(min_silence_duration=1.0, padding=0.15, mode="speed_up", speed_factor=3.0)
        result_video = op.apply(video, transcription=transcription)
        predicted = meta.silence_removal(
            min_silence_duration=1.0, padding=0.15, mode="speed_up", speed_factor=3.0, transcription=transcription
        )
        assert predicted.frame_count == len(result_video.frames)

"""Tests for `VideoEdit` JSON plan parsing, execution, and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from tests.test_config import BIG_VIDEO_PATH, SMALL_VIDEO_METADATA, SMALL_VIDEO_PATH
from videopython.base.transcription import Transcription, TranscriptionWord
from videopython.base.video import Video, VideoMetadata
from videopython.editing.video_edit import SegmentConfig, VideoEdit


def _make_synthetic_video(width: int, height: int, fps: float, seconds: float) -> Video:
    frame_count = round(fps * seconds)
    frames = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    return Video(frames=frames, fps=fps)


def _segment(
    *,
    source: str = SMALL_VIDEO_PATH,
    start: float = 0.0,
    end: float = 2.0,
    operations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "start": start,
        "end": end,
        "operations": operations or [],
    }


def _make_transcription(words_data: list[tuple[float, float, str]]) -> Transcription:
    return Transcription(words=[TranscriptionWord(start=s, end=e, word=w) for s, e, w in words_data])


# ----------------------------------------------------------------- construction


class TestConstruction:
    def test_empty_segments_raises(self):
        with pytest.raises(ValidationError):
            VideoEdit(segments=[])

    def test_direct_construction(self):
        segment = SegmentConfig(source=Path(SMALL_VIDEO_PATH), start=0, end=1, operations=[])
        edit = VideoEdit(segments=[segment])
        assert len(edit.segments) == 1

    def test_segment_end_must_exceed_start(self):
        with pytest.raises(ValidationError, match="must be greater than start"):
            SegmentConfig(source=Path("a.mp4"), start=2.0, end=1.0)

    def test_segment_start_negative_rejected(self):
        with pytest.raises(ValidationError):
            SegmentConfig(source=Path("a.mp4"), start=-1.0, end=2.0)

    def test_extra_segment_key_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs"):
            SegmentConfig(source=Path("a.mp4"), start=0, end=1, weird=42)


# ----------------------------------------------------------- parsing / serialization


class TestParsingAndSerialization:
    def test_from_dict_single_segment(self):
        plan = {"segments": [_segment(start=0.0, end=2.0)]}
        edit = VideoEdit.from_dict(plan)
        out = edit.to_dict()
        assert out["segments"][0]["source"] == SMALL_VIDEO_PATH
        assert out["segments"][0]["start"] == 0.0
        assert out["segments"][0]["end"] == 2.0
        assert out["segments"][0]["operations"] == []

    def test_from_json(self):
        plan = {"segments": [_segment(start=0.0, end=1.0)]}
        edit = VideoEdit.from_json(json.dumps(plan))
        assert isinstance(edit, VideoEdit)
        assert edit.to_dict()["segments"][0]["end"] == 1.0

    def test_from_json_invalid_wraps_jsondecodeerror(self):
        with pytest.raises(ValueError, match="Invalid VideoEdit JSON"):
            VideoEdit.from_json("{bad json")

    def test_flat_op_shape_round_trip(self):
        plan = {
            "segments": [
                _segment(
                    operations=[
                        {"op": "resize", "width": 400, "height": 250},
                        {"op": "blur_effect", "mode": "constant", "iterations": 5},
                    ]
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        out = edit.to_dict()
        ops = out["segments"][0]["operations"]
        assert ops[0]["op"] == "resize"
        assert ops[0]["width"] == 400
        assert ops[1]["op"] == "blur_effect"
        assert ops[1]["mode"] == "constant"

    def test_post_operations_round_trip(self):
        plan = {
            "segments": [_segment()],
            "post_operations": [{"op": "reverse"}, {"op": "fade", "mode": "out", "duration": 0.5}],
        }
        edit = VideoEdit.from_dict(plan)
        out = edit.to_dict()
        assert [op["op"] for op in out["post_operations"]] == ["reverse", "fade"]

    def test_window_round_trip(self):
        plan = {
            "segments": [
                _segment(
                    operations=[
                        {
                            "op": "blur_effect",
                            "mode": "constant",
                            "iterations": 1,
                            "window": {"start": 0.5, "stop": 1.5},
                        }
                    ]
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        op = edit.to_dict()["segments"][0]["operations"][0]
        assert op["window"] == {"start": 0.5, "stop": 1.5}

    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ValidationError):
            VideoEdit.from_dict({"segments": [_segment()], "bogus": True})

    def test_unknown_segment_key_rejected(self):
        with pytest.raises(ValidationError):
            VideoEdit.from_dict({"segments": [{**_segment(), "transforms": []}]})


# ----------------------------------------------------------------- json schema


class TestJsonSchema:
    def test_schema_top_level_shape(self):
        schema = VideoEdit.json_schema()
        assert schema["type"] == "object"
        assert "segments" in schema["properties"]
        assert "post_operations" in schema["properties"]
        assert "match_to_lowest_fps" in schema["properties"]
        assert schema["required"] == ["segments"]

    def test_schema_segment_shape(self):
        schema = VideoEdit.json_schema()
        seg_schema = schema["properties"]["segments"]["items"]
        assert seg_schema["required"] == ["source", "start", "end"]
        assert seg_schema["additionalProperties"] is False
        assert "operations" in seg_schema["properties"]

    def test_schema_includes_operations_via_discriminator(self):
        """The op-union schema should cover every registered Operation."""
        schema = VideoEdit.json_schema()
        op_schema = schema["properties"]["segments"]["items"]["properties"]["operations"]["items"]
        # Pydantic produces a discriminated-union schema with `oneOf` (or similar);
        # the exact shape comes from Operation.json_schema(), so just check the
        # union mentions a known op_id.
        as_json = json.dumps(op_schema)
        for op_id in ("resize", "blur_effect", "fade", "reverse"):
            assert op_id in as_json, f"Operation {op_id!r} missing from schema"

    def test_schema_has_descriptions_for_every_top_level_field(self):
        """LLMs consume `description` for each field. Every slot must carry one."""
        schema = VideoEdit.json_schema()
        for name, prop in schema["properties"].items():
            assert prop.get("description"), f"Top-level field {name!r} missing description"

        seg = schema["properties"]["segments"]["items"]
        for name, prop in seg["properties"].items():
            assert prop.get("description"), f"Segment field {name!r} missing description"

    def test_schema_op_class_and_field_descriptions_flow_through(self):
        """Each op in the union must carry a class description plus per-field descriptions."""
        schema = VideoEdit.json_schema()
        op_schema = schema["properties"]["segments"]["items"]["properties"]["operations"]["items"]
        defs = op_schema.get("$defs") or op_schema.get("definitions") or {}
        assert defs, "Operation union schema missing $defs"

        # Only entries with an `op` discriminator field are Operation subclasses;
        # `$defs` also contains referenced models/enums (TimeRange, CropMode, ...)
        # which don't need descriptions. Also skip the underscore-prefixed
        # test-internal ops other modules register.
        op_entries = {
            name: s for name, s in defs.items() if "op" in s.get("properties", {}) and not name.startswith("_")
        }
        assert op_entries, "No Operation entries found in $defs"
        for cls_name, cls_schema in op_entries.items():
            assert cls_schema.get("description"), f"Op {cls_name!r} missing class description"
            for fname, fprop in cls_schema["properties"].items():
                if fname == "op":
                    continue
                assert fprop.get("description"), f"Op {cls_name!r} field {fname!r} missing description"


# ----------------------------------------------------------------- execution


class TestExecution:
    def test_run_no_ops(self):
        plan = {"segments": [_segment(start=0.0, end=2.0)]}
        video = VideoEdit.from_dict(plan).run()
        assert video.total_seconds == pytest.approx(2.0, abs=0.25)

    def test_run_with_transform_and_effect(self):
        plan = {
            "segments": [
                _segment(
                    operations=[
                        {"op": "resize", "width": 400, "height": 250},
                        {"op": "blur_effect", "mode": "constant", "iterations": 1},
                    ]
                )
            ]
        }
        result = VideoEdit.from_dict(plan).run()
        assert result.frames.shape[2] == 400
        assert result.frames.shape[1] == 250

    def test_run_multi_segment_same_source(self):
        plan = {
            "segments": [
                _segment(start=0.0, end=2.0),
                _segment(start=4.0, end=6.0),
            ]
        }
        result = VideoEdit.from_dict(plan).run()
        assert result.total_seconds == pytest.approx(4.0, abs=0.3)

    def test_run_with_post_operations(self):
        plan = {
            "segments": [_segment(start=0.0, end=2.0)],
            "post_operations": [{"op": "resize", "width": 200, "height": 100}],
        }
        result = VideoEdit.from_dict(plan).run()
        assert result.frames.shape[2] == 200
        assert result.frames.shape[1] == 100


# ----------------------------------------------------------------- validation


class TestValidation:
    def test_validate(self):
        plan = {"segments": [_segment(start=1.0, end=5.0)]}
        meta = VideoEdit.from_dict(plan).validate()
        assert meta.total_seconds == pytest.approx(4.0, abs=0.1)
        assert meta.width == SMALL_VIDEO_METADATA.width
        assert meta.height == SMALL_VIDEO_METADATA.height

    def test_validate_resize_and_crop_chained(self):
        plan = {
            "segments": [
                _segment(
                    operations=[
                        {"op": "crop", "width": 0.5, "height": 0.5},
                        {"op": "resize", "width": 320, "height": 200},
                    ]
                )
            ]
        }
        meta = VideoEdit.from_dict(plan).validate()
        assert meta.width == 320
        assert meta.height == 200

    def test_validate_crop_matches_runtime(self):
        plan = {"segments": [_segment(operations=[{"op": "crop", "width": 5, "height": 5, "mode": "center"}])]}
        edit = VideoEdit.from_dict(plan)
        meta = edit.validate()
        video = edit.run()
        assert meta.width == video.frame_shape[1]
        assert meta.height == video.frame_shape[0]

    def test_validate_speed_change(self):
        plan = {"segments": [_segment(start=0.0, end=4.0, operations=[{"op": "speed_change", "speed": 2.0}])]}
        meta = VideoEdit.from_dict(plan).validate()
        assert meta.total_seconds == pytest.approx(2.0, abs=0.2)

    def test_validate_does_not_load_frames(self):
        plan = {"segments": [_segment(start=0.0, end=3.0)]}
        edit = VideoEdit.from_dict(plan)
        with patch.object(Video, "from_path", side_effect=AssertionError("Video.from_path should not be called")):
            meta = edit.validate()
            assert isinstance(meta, VideoMetadata)

    def test_incompatible_fps_when_matching_disabled(self):
        plan = {
            "segments": [
                _segment(start=0.0, end=2.0),
                _segment(start=2.0, end=4.0, operations=[{"op": "resample_fps", "fps": 23}]),
            ],
            "match_to_lowest_fps": False,
        }
        with pytest.raises(ValueError, match="fps"):
            VideoEdit.from_dict(plan).validate()

    def test_effect_window_bounds_validated(self):
        plan = {
            "segments": [
                _segment(
                    end=3.0,
                    operations=[
                        {
                            "op": "blur_effect",
                            "mode": "constant",
                            "iterations": 1,
                            "window": {"start": 10},
                        }
                    ],
                )
            ]
        }
        with pytest.raises(ValueError, match="exceeds duration"):
            VideoEdit.from_dict(plan).validate()


# ------------------------------------------------------- validate_with_metadata


class TestValidateWithMetadata:
    def test_matches_validate_for_single_source(self):
        plan = {"segments": [_segment(start=1.0, end=5.0), _segment(start=0.0, end=3.0)]}
        edit = VideoEdit.from_dict(plan)
        expected = edit.validate()
        result = edit.validate_with_metadata(SMALL_VIDEO_METADATA)
        assert result.total_seconds == pytest.approx(expected.total_seconds, abs=0.01)
        assert (result.width, result.height, result.fps) == (expected.width, expected.height, expected.fps)

    def test_end_exceeds_source_duration(self):
        plan = {"segments": [_segment(start=0.0, end=99.0)]}
        edit = VideoEdit.from_dict(plan)
        with pytest.raises(ValueError, match="exceeds source duration"):
            edit.validate_with_metadata(SMALL_VIDEO_METADATA)

    def test_multi_source_dict(self):
        plan = {
            "segments": [
                _segment(source="a.mp4", start=0.0, end=2.0),
                _segment(source="b.mp4", start=0.0, end=3.0),
            ]
        }
        meta_a = VideoMetadata(height=500, width=800, fps=24, frame_count=48, total_seconds=2.0)
        meta_b = VideoMetadata(height=500, width=800, fps=24, frame_count=72, total_seconds=3.0)
        result = VideoEdit.from_dict(plan).validate_with_metadata({"a.mp4": meta_a, "b.mp4": meta_b})
        assert result.total_seconds == pytest.approx(5.0, abs=0.01)

    def test_missing_source_in_dict_raises(self):
        plan = {
            "segments": [
                _segment(source="a.mp4", start=0.0, end=2.0),
                _segment(source="b.mp4", start=0.0, end=3.0),
            ]
        }
        meta_a = VideoMetadata(height=500, width=800, fps=24, frame_count=48, total_seconds=2.0)
        with pytest.raises(ValueError, match="no metadata"):
            VideoEdit.from_dict(plan).validate_with_metadata({"a.mp4": meta_a})

    def test_does_not_touch_disk(self):
        plan = {"segments": [_segment(source="nonexistent.mp4", start=0.0, end=2.0)]}
        meta = VideoMetadata(height=500, width=800, fps=24, frame_count=48, total_seconds=5.0)
        result = VideoEdit.from_dict(plan).validate_with_metadata(meta)
        assert isinstance(result, VideoMetadata)


# ----------------------------------------------------------------- matching


class TestSegmentMatching:
    def test_match_to_lowest_fps(self):
        meta_a = VideoMetadata(width=320, height=240, fps=30.0, frame_count=60, total_seconds=2.0)
        meta_b = VideoMetadata(width=320, height=240, fps=24.0, frame_count=48, total_seconds=2.0)
        plan = {"segments": [_segment(source="a.mp4", end=2.0), _segment(source="b.mp4", end=2.0)]}
        meta = VideoEdit.from_dict(plan).validate_with_metadata({"a.mp4": meta_a, "b.mp4": meta_b})
        assert meta.fps == 24.0

    def test_match_to_lowest_fps_disabled_raises(self):
        meta_a = VideoMetadata(width=320, height=240, fps=30.0, frame_count=60, total_seconds=2.0)
        meta_b = VideoMetadata(width=320, height=240, fps=24.0, frame_count=48, total_seconds=2.0)
        plan = {
            "segments": [_segment(source="a.mp4", end=2.0), _segment(source="b.mp4", end=2.0)],
            "match_to_lowest_fps": False,
        }
        with pytest.raises(ValueError, match="fps"):
            VideoEdit.from_dict(plan).validate_with_metadata({"a.mp4": meta_a, "b.mp4": meta_b})

    def test_match_to_lowest_resolution(self):
        meta_a = VideoMetadata(width=640, height=480, fps=24.0, frame_count=48, total_seconds=2.0)
        meta_b = VideoMetadata(width=320, height=240, fps=24.0, frame_count=48, total_seconds=2.0)
        plan = {"segments": [_segment(source="a.mp4", end=2.0), _segment(source="b.mp4", end=2.0)]}
        meta = VideoEdit.from_dict(plan).validate_with_metadata({"a.mp4": meta_a, "b.mp4": meta_b})
        assert (meta.width, meta.height) == (320, 240)

    def test_match_resolution_disabled_raises(self):
        meta_a = VideoMetadata(width=640, height=480, fps=24.0, frame_count=48, total_seconds=2.0)
        meta_b = VideoMetadata(width=320, height=240, fps=24.0, frame_count=48, total_seconds=2.0)
        plan = {
            "segments": [_segment(source="a.mp4", end=2.0), _segment(source="b.mp4", end=2.0)],
            "match_to_lowest_resolution": False,
        }
        with pytest.raises(ValueError, match="dimensions"):
            VideoEdit.from_dict(plan).validate_with_metadata({"a.mp4": meta_a, "b.mp4": meta_b})

    def test_match_flags_default_true(self):
        edit = VideoEdit.from_dict({"segments": [_segment()]})
        assert edit.match_to_lowest_fps is True
        assert edit.match_to_lowest_resolution is True

    def test_match_real_videos_resolves_to_smaller(self):
        """small_video (800x500 @24fps) + big_video (1080x1920 @30fps)."""
        plan = {
            "segments": [
                {"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 2.0},
                {"source": BIG_VIDEO_PATH, "start": 0.0, "end": 2.0},
            ]
        }
        meta = VideoEdit.from_dict(plan).validate()
        assert meta.fps == 24.0
        assert (meta.width, meta.height) == (800, 500)

    def test_match_disabled_with_real_videos_raises(self):
        plan = {
            "segments": [
                {"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 2.0},
                {"source": BIG_VIDEO_PATH, "start": 0.0, "end": 2.0},
            ],
            "match_to_lowest_fps": False,
            "match_to_lowest_resolution": False,
        }
        with pytest.raises(ValueError):
            VideoEdit.from_dict(plan).validate()


# ----------------------------------------------------------------- parse errors


class TestParseErrors:
    def test_unknown_operation_id(self):
        with pytest.raises(ValidationError, match="Unknown op_id"):
            VideoEdit.from_dict({"segments": [_segment(operations=[{"op": "nope"}])]})

    def test_missing_op_field(self):
        with pytest.raises(ValidationError, match="missing required 'op'"):
            VideoEdit.from_dict({"segments": [_segment(operations=[{"width": 100}])]})

    def test_unknown_field_on_op(self):
        with pytest.raises(ValidationError):
            VideoEdit.from_dict({"segments": [_segment(operations=[{"op": "resize", "width": 100, "bogus": 1}])]})

    def test_resize_requires_at_least_one_dim(self):
        with pytest.raises(ValidationError, match="Resize requires"):
            VideoEdit.from_dict({"segments": [_segment(operations=[{"op": "resize"}])]})

    def test_resize_null_dims_rejected(self):
        with pytest.raises(ValidationError, match="Resize requires"):
            VideoEdit.from_dict({"segments": [_segment(operations=[{"op": "resize", "width": None, "height": None}])]})

    def test_speed_change_zero_rejected(self):
        with pytest.raises(ValidationError):
            VideoEdit.from_dict({"segments": [_segment(operations=[{"op": "speed_change", "speed": 0}])]})

    def test_blur_iterations_must_be_positive(self):
        with pytest.raises(ValidationError):
            VideoEdit.from_dict(
                {"segments": [_segment(operations=[{"op": "blur_effect", "mode": "constant", "iterations": 0}])]}
            )

    def test_window_stop_before_start_rejected(self):
        with pytest.raises(ValidationError, match="must be >="):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment(
                            operations=[
                                {
                                    "op": "blur_effect",
                                    "mode": "constant",
                                    "iterations": 1,
                                    "window": {"start": 2.0, "stop": 1.0},
                                }
                            ]
                        )
                    ]
                }
            )

    def test_window_negative_start_rejected(self):
        with pytest.raises(ValidationError):
            VideoEdit.from_dict(
                {
                    "segments": [
                        _segment(
                            operations=[
                                {
                                    "op": "blur_effect",
                                    "mode": "constant",
                                    "iterations": 1,
                                    "window": {"start": -1.0},
                                }
                            ]
                        )
                    ]
                }
            )


# --------------------------------------------------------------- context injection


class TestContextInjection:
    def test_silence_removal_with_context(self):
        transcription = _make_transcription([(1.0, 2.0, "hello"), (4.0, 5.0, "world")])
        plan = {
            "segments": [
                _segment(
                    source="fake.mp4",
                    start=0.0,
                    end=6.0,
                    operations=[{"op": "silence_removal", "min_silence_duration": 0.5}],
                )
            ]
        }
        source_meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        meta = VideoEdit.from_dict(plan).validate_with_metadata(source_meta, context={"transcription": transcription})
        assert meta.frame_count < round(6.0 * 24)

    def test_silence_removal_no_context_returns_identity(self):
        """Without context, SilenceRemoval.predict_metadata returns identity, not an error."""
        plan = {
            "segments": [
                _segment(
                    source="fake.mp4",
                    start=0.0,
                    end=6.0,
                    operations=[{"op": "silence_removal"}],
                )
            ]
        }
        source_meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        meta = VideoEdit.from_dict(plan).validate_with_metadata(source_meta)
        assert meta.frame_count == round(6.0 * 24)

    def test_silence_removal_in_post_operations(self):
        transcription = _make_transcription([(1.0, 2.0, "hello")])
        plan = {
            "segments": [_segment(source="fake.mp4", start=0.0, end=4.0)],
            "post_operations": [{"op": "silence_removal"}],
        }
        source_meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        meta = VideoEdit.from_dict(plan).validate_with_metadata(source_meta, context={"transcription": transcription})
        assert isinstance(meta, VideoMetadata)


# ---------------------------------------------------------------- metadata chain


class TestMetadataChain:
    def test_reverse_preserves_metadata(self):
        plan = {
            "segments": [_segment(start=0.0, end=3.0, operations=[{"op": "reverse"}])],
            "post_operations": [{"op": "reverse"}],
        }
        source_meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        meta = VideoEdit.from_dict(plan).validate_with_metadata(source_meta)
        cut_count = round(3.0 * 24) - round(0.0 * 24)
        assert meta.frame_count == cut_count

    def test_freeze_frame_extends_duration(self):
        plan = {
            "segments": [
                _segment(
                    source="fake.mp4",
                    start=0.0,
                    end=3.0,
                    operations=[{"op": "freeze_frame", "timestamp": 1.0, "duration": 2.0}],
                )
            ]
        }
        source_meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        meta = VideoEdit.from_dict(plan).validate_with_metadata(source_meta)
        assert meta.total_seconds > 3.0

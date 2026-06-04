"""Tests for `VideoEdit` JSON plan parsing, execution, and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from tests.test_config import BIG_VIDEO_PATH, SMALL_VIDEO_METADATA, SMALL_VIDEO_PATH
from videopython.base.exceptions import PlanErrorCode, PlanValidationError
from videopython.base.transcription import Transcription, TranscriptionWord
from videopython.base.video import Video, VideoMetadata
from videopython.editing.transforms import Resize, SpeedChange
from videopython.editing.video_edit import SegmentConfig, VideoEdit, _segment_context


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

    def test_schema_source_carries_path_format(self):
        # Derived from `model_json_schema()`, so `source` keeps its `format: path`
        # (the hand-rolled version dropped it).
        schema = VideoEdit.json_schema()
        source = schema["properties"]["segments"]["items"]["properties"]["source"]
        assert source["type"] == "string"
        assert source["format"] == "path"

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

    def test_run_with_image_overlay(self, tmp_path):
        logo = tmp_path / "logo.png"
        arr = np.zeros((50, 50, 4), dtype=np.uint8)
        arr[:, :, 2] = 255  # blue
        arr[:, :, 3] = 255
        Image.fromarray(arr, "RGBA").save(logo)
        plan = {
            "segments": [
                _segment(
                    start=0.0,
                    end=2.0,
                    operations=[{"op": "image_overlay", "source": str(logo), "scale": 0.2, "anchor": "bottom_right"}],
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        meta = edit.validate()
        result = edit.run()
        assert result.frame_shape[0] == meta.height
        assert result.frame_shape[1] == meta.width
        assert (result.frames[0][:, :, 2] == 255).any()

    def test_run_with_svg_overlay(self, tmp_path):
        svg = tmp_path / "logo.svg"
        svg.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40" width="100" '
            'height="40"><rect width="100" height="40" fill="rgb(0,200,120)"/></svg>'
        )
        plan = {
            "segments": [
                _segment(
                    start=0.0,
                    end=2.0,
                    operations=[{"op": "image_overlay", "source": str(svg), "scale": 0.2, "anchor": "bottom_right"}],
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        meta = edit.validate()
        result = edit.run()
        assert result.frame_shape[0] == meta.height
        assert result.frame_shape[1] == meta.width
        assert (result.frames[0][:, :, 1] > 150).any()  # the green logo is composited


# ----------------------------------------------------------------- validation


class TestValidation:
    def test_validate(self):
        plan = {"segments": [_segment(start=1.0, end=5.0)]}
        meta = VideoEdit.from_dict(plan).validate()
        assert meta.total_seconds == pytest.approx(4.0, abs=0.1)
        assert meta.width == SMALL_VIDEO_METADATA.width
        assert meta.height == SMALL_VIDEO_METADATA.height

    def test_validate_rejects_missing_image_overlay_source(self):
        plan = {"segments": [_segment(operations=[{"op": "image_overlay", "source": "/no/such/logo.png"}])]}
        with pytest.raises(ValueError, match="not a readable image"):
            VideoEdit.from_dict(plan).validate()

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


class TestWindowClamp:
    """Window-stop clamping at validate (clamp_windows) and its run() parity."""

    @staticmethod
    def _speed_then_blur_plan() -> dict[str, Any]:
        # 12s source @1.5x -> 8s; blur window.stop=10 overruns the post-speed length.
        return {
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

    def test_raises_by_default(self):
        edit = VideoEdit.from_dict(self._speed_then_blur_plan())
        with pytest.raises(PlanValidationError, match="window.stop") as exc:
            edit.validate_with_metadata(SMALL_VIDEO_METADATA)
        assert exc.value.errors[0].code == PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION

    def test_passes_with_clamp(self):
        edit = VideoEdit.from_dict(self._speed_then_blur_plan())
        meta = edit.validate_with_metadata(SMALL_VIDEO_METADATA, clamp_windows=True)
        # Effect metadata prediction is identity; the clamp only suppresses the
        # false raise and does not change the returned duration.
        assert meta.total_seconds == pytest.approx(8.0, abs=0.05)

    def test_clamped_stop_equals_post_op_duration(self):
        edit = VideoEdit.from_dict(self._speed_then_blur_plan())
        post_dur = SpeedChange(speed=1.5).predict_metadata(SMALL_VIDEO_METADATA).total_seconds
        repaired, clamps = edit.repair(SMALL_VIDEO_METADATA)
        assert [c.location for c in clamps] == ["segments[0].operations[1]"]
        assert clamps[0].field == "window.stop"
        assert clamps[0].old == 10.0
        assert clamps[0].new == pytest.approx(post_dur)
        # The repaired op carries the clamped stop; self is untouched.
        assert repaired.segments[0].operations[1].window.stop == pytest.approx(post_dur)
        assert edit.segments[0].operations[1].window.stop == 10.0
        # Clamped validate now passes for the repaired plan with no clamping.
        repaired.validate_with_metadata(SMALL_VIDEO_METADATA)

    def test_clamp_matches_run_resolved_window(self):
        # The clamped stop is exactly what Effect._resolved_window uses at run
        # time: min(stop, total_seconds) against the post-op running duration.
        post_dur = SpeedChange(speed=1.5).predict_metadata(SMALL_VIDEO_METADATA).total_seconds
        repaired, _ = VideoEdit.from_dict(self._speed_then_blur_plan()).repair(SMALL_VIDEO_METADATA)
        blur = repaired.segments[0].operations[1]
        _, run_stop = blur._resolved_window(post_dur)
        assert blur.window.stop == pytest.approx(run_stop)

    def test_run_to_file_produces_clamped_output(self, tmp_path):
        edit = VideoEdit.from_dict(self._speed_then_blur_plan())
        out = edit.run_to_file(tmp_path / "clamped")
        result = Video.from_path(str(out))
        post_dur = SpeedChange(speed=1.5).predict_metadata(SMALL_VIDEO_METADATA).total_seconds
        # run() silently clamps the same window, so the output is the post-speed length.
        assert result.total_seconds == pytest.approx(post_dur, abs=0.25)

    def test_cut_then_window_parity(self):
        # cut to 6s, then a blur with window.stop=9 overruns the post-cut length.
        plan = {
            "segments": [
                _segment(
                    start=0.0,
                    end=12.0,
                    operations=[
                        {"op": "cut", "start": 0.0, "end": 6.0},
                        {
                            "op": "blur_effect",
                            "mode": "constant",
                            "iterations": 1,
                            "window": {"start": 0, "stop": 9.0},
                        },
                    ],
                )
            ]
        }
        edit = VideoEdit.from_dict(plan)
        with pytest.raises(PlanValidationError, match="window.stop"):
            edit.validate_with_metadata(SMALL_VIDEO_METADATA)
        repaired, clamps = edit.repair(SMALL_VIDEO_METADATA)
        assert clamps[0].old == 9.0
        assert clamps[0].new == pytest.approx(6.0, abs=0.05)
        edit.validate_with_metadata(SMALL_VIDEO_METADATA, clamp_windows=True)

    def test_start_overrun_still_raises_with_clamp(self):
        # Clamping stop only -- a window.start past the duration must still raise.
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
                            "window": {"start": 10.0},
                        },
                    ],
                )
            ]
        }
        with pytest.raises(PlanValidationError, match="window.start"):
            VideoEdit.from_dict(plan).validate_with_metadata(SMALL_VIDEO_METADATA, clamp_windows=True)


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


# ----------------------------------------------------- per-segment context re-base


def _word_spans(transcription: Transcription) -> list[tuple[float, float, str]]:
    return [(w.start, w.end, w.word) for w in transcription.words]


class TestSegmentContextHelper:
    """Unit tests for ``_segment_context`` (the per-segment re-basing seam)."""

    def test_rebases_mid_video_cut_to_zero_based(self):
        tx = Transcription(
            words=[TranscriptionWord(12.0, 13.0, "a"), TranscriptionWord(14.5, 15.5, "b")],
            language="en",
        )
        out = _segment_context({"transcription": tx}, 12.0, 16.0)
        assert out is not None
        result = out["transcription"]
        assert _word_spans(result) == [(0.0, 1.0, "a"), (2.5, 3.5, "b")]
        assert result.language == "en"

    def test_slices_out_of_range_words_then_offsets(self):
        tx = _make_transcription([(1.0, 2.0, "x"), (5.0, 6.0, "keep"), (20.0, 21.0, "y")])
        out = _segment_context({"transcription": tx}, 4.0, 10.0)
        assert out is not None
        assert _word_spans(out["transcription"]) == [(1.0, 2.0, "keep")]

    def test_start_zero_still_clips_trailing_words(self):
        tx = _make_transcription([(1.0, 2.0, "in"), (8.0, 9.0, "out")])
        out = _segment_context({"transcription": tx}, 0.0, 5.0)
        assert out is not None
        assert _word_spans(out["transcription"]) == [(1.0, 2.0, "in")]

    def test_no_overlap_drops_transcription_key_keeps_others(self):
        tx = _make_transcription([(1.0, 2.0, "hello")])
        ctx = {"transcription": tx, "other": 42}
        out = _segment_context(ctx, 50.0, 60.0)
        assert out is not None
        assert "transcription" not in out
        assert out["other"] == 42

    def test_does_not_mutate_original_context(self):
        tx = _make_transcription([(12.0, 13.0, "a")])
        ctx = {"transcription": tx, "k": 1}
        out = _segment_context(ctx, 12.0, 14.0)
        assert ctx["transcription"] is tx
        assert _word_spans(ctx["transcription"]) == [(12.0, 13.0, "a")]
        assert out is not ctx

    def test_none_and_empty_context_pass_through(self):
        assert _segment_context(None, 0.0, 5.0) is None
        empty: dict[str, Any] = {}
        assert _segment_context(empty, 0.0, 5.0) is empty

    def test_non_transcription_value_passes_through_unchanged(self):
        ctx = {"transcription": "not-a-transcription", "other": 1}
        assert _segment_context(ctx, 10.0, 20.0) is ctx
        assert _segment_context({"other": 1}, 10.0, 20.0) == {"other": 1}


class TestSegmentContextWiring:
    """``run()`` and ``validate()`` re-base per segment but not post_operations."""

    def test_process_rebases_context_for_segment_ops(self):
        seg = SegmentConfig.model_validate(
            {"source": "fake.mp4", "start": 10.0, "end": 20.0, "operations": [{"op": "reverse"}]}
        )
        abs_tx = _make_transcription([(12.0, 13.0, "a"), (18.0, 19.0, "b")])
        video = _make_synthetic_video(32, 32, 24, 0.5)
        captured: dict[str, Any] = {}

        def fake_apply(op: Any, vid: Video, ctx: dict[str, Any] | None) -> Video:
            captured["ctx"] = ctx
            return vid

        with patch("videopython.editing.video_edit._apply_with_context", side_effect=fake_apply):
            out = seg.process(video, {"transcription": abs_tx, "other": 123})

        assert out is video
        assert captured["ctx"]["other"] == 123
        assert _word_spans(captured["ctx"]["transcription"]) == [(2.0, 3.0, "a"), (8.0, 9.0, "b")]

    def test_validate_rebases_segment_but_not_post_operations(self):
        plan = {
            "segments": [_segment(source="fake.mp4", start=10.0, end=20.0, operations=[{"op": "silence_removal"}])],
            "post_operations": [{"op": "silence_removal"}],
        }
        abs_tx = _make_transcription([(12.0, 13.0, "a"), (18.0, 19.0, "b")])
        source_meta = VideoMetadata(height=100, width=100, fps=24, frame_count=720, total_seconds=30.0)
        seen: list[Transcription | None] = []

        def fake_predict(op: Any, meta: VideoMetadata, ctx: dict[str, Any] | None) -> VideoMetadata:
            seen.append(ctx["transcription"] if ctx and "transcription" in ctx else None)
            return meta

        with patch("videopython.editing.video_edit._predict_with_context", side_effect=fake_predict):
            VideoEdit.from_dict(plan).validate_with_metadata(source_meta, context={"transcription": abs_tx})

        assert len(seen) == 2
        segment_tx, post_tx = seen
        assert segment_tx is not None and post_tx is not None
        assert _word_spans(segment_tx) == [(2.0, 3.0, "a"), (8.0, 9.0, "b")]
        assert _word_spans(post_tx) == [(12.0, 13.0, "a"), (18.0, 19.0, "b")]

    def test_mid_cut_predicts_same_metadata_as_equivalent_zero_based_cut(self):
        source_meta = VideoMetadata(height=100, width=100, fps=24, frame_count=720, total_seconds=30.0)

        mid_plan = {
            "segments": [_segment(source="fake.mp4", start=10.0, end=20.0, operations=[{"op": "silence_removal"}])]
        }
        mid_tx = _make_transcription([(10.5, 11.5, "hello"), (18.0, 19.0, "world")])
        mid_meta = VideoEdit.from_dict(mid_plan).validate_with_metadata(source_meta, context={"transcription": mid_tx})

        base_plan = {
            "segments": [_segment(source="fake.mp4", start=0.0, end=10.0, operations=[{"op": "silence_removal"}])]
        }
        base_tx = _make_transcription([(0.5, 1.5, "hello"), (8.0, 9.0, "world")])
        base_meta = VideoEdit.from_dict(base_plan).validate_with_metadata(
            source_meta, context={"transcription": base_tx}
        )

        assert mid_meta.frame_count == base_meta.frame_count
        assert mid_meta.frame_count < round(10.0 * 24)


class _FakeTimeline:
    """Minimal SegmentRebaseable: not a Transcription, just slice + offset."""

    def __init__(self, lo: float, hi: float) -> None:
        self.lo, self.hi = lo, hi

    def slice(self, start: float, end: float) -> _FakeTimeline | None:
        lo, hi = max(self.lo, start), min(self.hi, end)
        return _FakeTimeline(lo, hi) if lo < hi else None

    def offset(self, delta: float) -> _FakeTimeline:
        return _FakeTimeline(self.lo + delta, self.hi + delta)


class TestGenericContextRebasing:
    """_segment_context re-bases anything with slice+offset, not just Transcription."""

    def test_duck_typed_value_is_rebased(self):
        out = _segment_context({"clip": _FakeTimeline(12.0, 18.0), "k": 1}, 12.0, 16.0)
        assert out is not None
        clip = out["clip"]
        assert isinstance(clip, _FakeTimeline)
        assert (clip.lo, clip.hi) == (0.0, 4.0)
        assert out["k"] == 1

    def test_duck_typed_value_dropped_when_no_overlap(self):
        out = _segment_context({"clip": _FakeTimeline(50.0, 60.0)}, 0.0, 5.0)
        assert out == {}


class TestStreamingRequiresGuard:
    """An op that `requires` runtime context forces the eager path."""

    def test_build_streaming_plan_returns_none_for_requires_op(self, monkeypatch):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    _segment(
                        source=SMALL_VIDEO_PATH,
                        start=0.0,
                        end=1.0,
                        operations=[{"op": "resize", "width": 64, "height": 64}],
                    )
                ]
            }
        )
        seg = plan.segments[0]
        # resize is streamable, so without the guard a plan is built.
        assert plan._build_streaming_plan(seg, None, None, None) is not None
        # Same streamable op, now declaring a context requirement -> eager.
        monkeypatch.setattr(Resize, "requires", ("transcription",))
        assert plan._build_streaming_plan(seg, None, None, None) is None


class TestPostOpsGuard:
    """Multi-segment plans reject post_operations that need time-based context."""

    @staticmethod
    def _plan(n_segments: int, post: list[dict[str, Any]]) -> VideoEdit:
        segs = [_segment(source="fake.mp4", start=0.0, end=5.0) for _ in range(n_segments)]
        return VideoEdit.from_dict({"segments": segs, "post_operations": post})

    _META = VideoMetadata(height=100, width=100, fps=24, frame_count=240, total_seconds=10.0)

    def test_multi_segment_post_op_requiring_context_raises_on_validate(self):
        plan = self._plan(2, [{"op": "silence_removal"}])
        tx = _make_transcription([(1.0, 2.0, "hello")])
        with pytest.raises(ValueError, match="not re-based across a multi-segment concat"):
            plan.validate_with_metadata(self._META, context={"transcription": tx})

    def test_multi_segment_post_op_requiring_context_raises_on_run(self):
        plan = self._plan(2, [{"op": "silence_removal"}])
        tx = _make_transcription([(1.0, 2.0, "hello")])
        # Guard fires before any disk access, so "fake.mp4" is never read.
        with pytest.raises(ValueError, match="post_operation 'silence_removal' requires"):
            plan.run(context={"transcription": tx})

    def test_multi_segment_post_op_without_requires_is_allowed(self):
        plan = self._plan(2, [{"op": "reverse"}])
        tx = _make_transcription([(1.0, 2.0, "hello")])
        # reverse has no `requires`; the transcription is irrelevant to it.
        plan.validate_with_metadata(self._META, context={"transcription": tx})

    def test_single_segment_post_op_requiring_context_is_allowed(self):
        plan = self._plan(1, [{"op": "silence_removal"}])
        tx = _make_transcription([(1.0, 2.0, "hello")])
        # Single-segment concat == the one segment's timeline: documented-supported.
        plan.validate_with_metadata(self._META, context={"transcription": tx})


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


# --------------------------------------------------- typed PlanValidationError


class TestPlanValidationErrors:
    """Each validate raise site emits a typed `PlanValidationError` whose
    `.errors[0]` carries the right code/location, while `str(e)` stays
    byte-identical to the pre-change bare `ValueError` prose."""

    def test_segment_end_exceeds_source(self):
        plan = {"segments": [_segment(start=0.0, end=99.0)]}
        edit = VideoEdit.from_dict(plan)
        with pytest.raises(PlanValidationError) as exc:
            edit.validate_with_metadata(SMALL_VIDEO_METADATA)
        assert str(exc.value) == "Segment 0: end (99.0) exceeds source duration (12s)"
        err = exc.value.errors[0]
        assert err.code is PlanErrorCode.SEGMENT_END_EXCEEDS_SOURCE
        assert err.location == "segments[0]"
        assert err.field == "end"
        assert err.value == 99.0
        assert err.limit == 12

    def test_segment_end_equals_total_passes(self):
        # SMALL_VIDEO_METADATA.total_seconds == 12; exact boundary must pass.
        plan = {"segments": [_segment(start=0.0, end=12.0)]}
        out = VideoEdit.from_dict(plan).validate_with_metadata(SMALL_VIDEO_METADATA)
        assert out.total_seconds == pytest.approx(12.0, abs=1e-3)

    def test_segment_end_within_eps_passes(self):
        # total + 5e-4 is inside DURATION_EPS, so it must pass.
        plan = {"segments": [_segment(start=0.0, end=12.0 + 5e-4)]}
        VideoEdit.from_dict(plan).validate_with_metadata(SMALL_VIDEO_METADATA)

    def test_segment_end_beyond_eps_rejects_with_segment_index(self):
        # total + 2e-3 is beyond DURATION_EPS, so it must reject with the index.
        plan = {"segments": [_segment(start=0.0, end=12.0 + 2e-3)]}
        with pytest.raises(PlanValidationError) as exc:
            VideoEdit.from_dict(plan).validate_with_metadata(SMALL_VIDEO_METADATA)
        err = exc.value.errors[0]
        assert err.code is PlanErrorCode.SEGMENT_END_EXCEEDS_SOURCE
        assert err.location == "segments[0]"
        assert str(exc.value).startswith("Segment 0:")

    def test_effect_window_exceeds_duration(self):
        plan = {
            "segments": [
                _segment(
                    end=3.0,
                    operations=[
                        {"op": "blur_effect", "mode": "constant", "iterations": 1, "window": {"start": 10}},
                    ],
                )
            ]
        }
        with pytest.raises(PlanValidationError) as exc:
            VideoEdit.from_dict(plan).validate_with_metadata(SMALL_VIDEO_METADATA)
        assert str(exc.value) == "Effect 'blur_effect' window.start (10.0) exceeds duration (3.0s)"
        err = exc.value.errors[0]
        assert err.code is PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION
        assert err.location == "segments[0].operations[0]"
        assert err.op == "blur_effect"
        assert err.field == "window.start"

    def test_effect_window_stop_exceeds_duration(self):
        plan = {
            "segments": [
                _segment(
                    end=3.0,
                    operations=[
                        {"op": "blur_effect", "mode": "constant", "iterations": 1, "window": {"stop": 10}},
                    ],
                )
            ]
        }
        with pytest.raises(PlanValidationError) as exc:
            VideoEdit.from_dict(plan).validate_with_metadata(SMALL_VIDEO_METADATA)
        assert str(exc.value) == "Effect 'blur_effect' window.stop (10.0) exceeds duration (3.0s)"
        err = exc.value.errors[0]
        assert err.code is PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION
        assert err.location == "segments[0].operations[0]"
        assert err.field == "window.stop"

    def test_cut_exceeds_duration_enriched_with_segment_location(self):
        plan = {"segments": [_segment(start=0.0, end=2.0, operations=[{"op": "cut", "start": 0.0, "end": 99.0}])]}
        with pytest.raises(PlanValidationError) as exc:
            VideoEdit.from_dict(plan).validate_with_metadata(SMALL_VIDEO_METADATA)
        # The cut runs against the post-cut segment metadata (2.0s).
        assert str(exc.value) == "end time (99.0) exceeds video duration (2.0)"
        err = exc.value.errors[0]
        assert err.code is PlanErrorCode.CUT_EXCEEDS_DURATION
        assert err.location == "segments[0].operations[0]"
        assert err.op == "cut"

    def test_concat_mismatch_fps(self):
        meta_a = VideoMetadata(width=320, height=240, fps=30.0, frame_count=60, total_seconds=2.0)
        meta_b = VideoMetadata(width=320, height=240, fps=24.0, frame_count=48, total_seconds=2.0)
        plan = {
            "segments": [_segment(source="a.mp4", end=2.0), _segment(source="b.mp4", end=2.0)],
            "match_to_lowest_fps": False,
        }
        with pytest.raises(PlanValidationError) as exc:
            VideoEdit.from_dict(plan).validate_with_metadata({"a.mp4": meta_a, "b.mp4": meta_b})
        assert str(exc.value) == (
            "Segment 0 fps (30.0) != segment 1 fps (24.0); all segments must share fps for concatenation."
        )
        err = exc.value.errors[0]
        assert err.code is PlanErrorCode.CONCAT_MISMATCH
        assert err.location == "segments[1]"
        assert err.field == "fps"

    def test_concat_mismatch_dimensions(self):
        meta_a = VideoMetadata(width=640, height=480, fps=24.0, frame_count=48, total_seconds=2.0)
        meta_b = VideoMetadata(width=320, height=240, fps=24.0, frame_count=48, total_seconds=2.0)
        plan = {
            "segments": [_segment(source="a.mp4", end=2.0), _segment(source="b.mp4", end=2.0)],
            "match_to_lowest_resolution": False,
        }
        with pytest.raises(PlanValidationError) as exc:
            VideoEdit.from_dict(plan).validate_with_metadata({"a.mp4": meta_a, "b.mp4": meta_b})
        assert str(exc.value) == (
            "Segment 0 dimensions (640x480) != segment 1 (320x240); all segments must share dimensions."
        )
        err = exc.value.errors[0]
        assert err.code is PlanErrorCode.CONCAT_MISMATCH
        assert err.location == "segments[1]"
        assert err.field == "dimensions"

    def test_unknown_op_is_typed(self):
        from videopython.editing.video_edit import _resolve_operation

        with pytest.raises(PlanValidationError) as exc:
            _resolve_operation({"op": "nope"})
        assert "Unknown op_id 'nope'" in str(exc.value)
        err = exc.value.errors[0]
        assert err.code is PlanErrorCode.UNKNOWN_OP
        assert err.op == "nope"

    def test_unknown_op_via_from_dict_still_validation_error(self):
        # Raised inside a Pydantic BeforeValidator, so it surfaces as a
        # ValidationError; the message is preserved for substring matching.
        with pytest.raises(ValidationError, match="Unknown op_id"):
            VideoEdit.from_dict({"segments": [_segment(operations=[{"op": "nope"}])]})

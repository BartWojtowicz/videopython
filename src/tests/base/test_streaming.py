"""Tests for the streaming video processing pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.test_config import SMALL_VIDEO_PATH
from videopython.base.effects import ColorGrading, Fade
from videopython.base.streaming import EffectScheduleEntry, FrameEncoder, StreamingSegmentPlan, stream_segment
from videopython.base.video import Video, VideoMetadata
from videopython.editing import VideoEdit


@pytest.fixture
def small_meta():
    return VideoMetadata.from_path(SMALL_VIDEO_PATH)


class TestStreamSegment:
    """Test the low-level stream_segment function."""

    def test_no_effects_round_trip(self, small_meta):
        """Streaming with no effects should produce a valid video."""
        plan = StreamingSegmentPlan(
            source_path=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=2.0,
            output_fps=small_meta.fps,
            output_width=small_meta.width,
            output_height=small_meta.height,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            stream_segment(plan, out_path)
            assert out_path.exists()
            assert out_path.stat().st_size > 0
            result_meta = VideoMetadata.from_path(str(out_path))
            assert result_meta.width == small_meta.width
            assert result_meta.height == small_meta.height
        finally:
            out_path.unlink(missing_ok=True)

    def test_color_grading_streaming(self, small_meta):
        """ColorGrading via streaming should modify frames."""
        effect = ColorGrading(brightness=0.3)
        total_frames = round(2.0 * small_meta.fps)
        plan = StreamingSegmentPlan(
            source_path=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=2.0,
            output_fps=small_meta.fps,
            output_width=small_meta.width,
            output_height=small_meta.height,
            effect_schedule=[EffectScheduleEntry(effect, 0, total_frames)],
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            stream_segment(plan, out_path)
            assert out_path.exists()
            # Load both and compare -- streaming should differ from original
            original = Video.from_path(SMALL_VIDEO_PATH, end_second=2.0)
            streamed = Video.from_path(str(out_path))
            # Brightened frames should have higher mean
            assert streamed.frames.mean() > original.frames.mean()
        finally:
            out_path.unlink(missing_ok=True)

    def test_fade_streaming(self, small_meta):
        """Fade effect via streaming should produce dark first frame."""
        effect = Fade(mode="in", duration=1.0)
        total_frames = round(2.0 * small_meta.fps)
        plan = StreamingSegmentPlan(
            source_path=Path(SMALL_VIDEO_PATH),
            start_second=0,
            end_second=2.0,
            output_fps=small_meta.fps,
            output_width=small_meta.width,
            output_height=small_meta.height,
            effect_schedule=[EffectScheduleEntry(effect, 0, total_frames)],
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            stream_segment(plan, out_path)
            result = Video.from_path(str(out_path))
            # First frame should be very dark (faded in from black)
            assert result.frames[0].mean() < 5
        finally:
            out_path.unlink(missing_ok=True)


class TestVideoEditRunToFile:
    """Test VideoEdit.run_to_file() integration."""

    def test_single_segment_streaming(self):
        """Single segment with streamable effects should stream."""
        plan = {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": 0,
                    "end": 2.0,
                    "transforms": [],
                    "effects": [
                        {"op": "color_adjust", "args": {"brightness": 0.2}},
                    ],
                }
            ],
            "post_transforms": [],
            "post_effects": [],
        }
        edit = VideoEdit.from_dict(plan)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            result_path = edit.run_to_file(out_path)
            assert result_path.exists()
            assert result_path.stat().st_size > 0
            meta = VideoMetadata.from_path(str(result_path))
            assert meta.width == 800
            assert meta.height == 500
        finally:
            out_path.unlink(missing_ok=True)

    def test_streaming_matches_eager(self):
        """Streaming and eager paths should produce similar output."""
        plan_dict = {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": 0,
                    "end": 2.0,
                    "transforms": [],
                    "effects": [
                        {"op": "color_adjust", "args": {"saturation": 0, "contrast": 1.15}},
                    ],
                }
            ],
            "post_transforms": [],
            "post_effects": [],
        }
        # Eager path
        edit_eager = VideoEdit.from_dict(plan_dict)
        eager_video = edit_eager.run()

        # Streaming path
        edit_stream = VideoEdit.from_dict(plan_dict)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            edit_stream.run_to_file(out_path)
            streamed_video = Video.from_path(str(out_path))

            # Frame counts should match (within 1 due to codec rounding)
            assert abs(len(eager_video.frames) - len(streamed_video.frames)) <= 1

            # Compare pixel values -- lossy encode means not exact, but should be close
            min_frames = min(len(eager_video.frames), len(streamed_video.frames))
            # Sample middle frame
            mid = min_frames // 2
            eager_frame = eager_video.frames[mid].astype(np.float32)
            stream_frame = streamed_video.frames[mid].astype(np.float32)
            # Mean absolute error should be small (re-encoding introduces some loss)
            mae = np.abs(eager_frame - stream_frame).mean()
            assert mae < 15, f"Mean absolute error too high: {mae}"
        finally:
            out_path.unlink(missing_ok=True)

    def test_volume_adjust_streaming(self):
        """Audio-only effects should work in streaming mode."""
        plan = {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": 0,
                    "end": 2.0,
                    "transforms": [],
                    "effects": [
                        {"op": "volume_adjust", "args": {"volume": 1.5}},
                    ],
                }
            ],
        }
        edit = VideoEdit.from_dict(plan)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            result_path = edit.run_to_file(out_path)
            assert result_path.exists()
        finally:
            out_path.unlink(missing_ok=True)

    def test_multiple_effects_streaming(self):
        """Multiple effects should chain correctly in streaming mode."""
        plan = {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": 0,
                    "end": 2.0,
                    "transforms": [],
                    "effects": [
                        {"op": "color_adjust", "args": {"saturation": 0}},
                        {"op": "volume_adjust", "args": {"volume": 1.6}},
                        {"op": "fade", "args": {"mode": "in_out", "duration": 0.5}},
                    ],
                }
            ],
        }
        edit = VideoEdit.from_dict(plan)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            result_path = edit.run_to_file(out_path)
            assert result_path.exists()
            result = Video.from_path(str(result_path))
            # First frame should be black (fade in)
            assert result.frames[0].mean() < 5
        finally:
            out_path.unlink(missing_ok=True)


class TestStreamableTransforms:
    """Verify all transforms tagged streamable work via run_to_file."""

    def _run_plan(self, plan_dict):
        edit = VideoEdit.from_dict(plan_dict)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            result_path = edit.run_to_file(out_path)
            assert result_path.exists()
            assert result_path.stat().st_size > 0
            return VideoMetadata.from_path(str(result_path))
        finally:
            out_path.unlink(missing_ok=True)

    def test_resize(self):
        meta = self._run_plan(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 0,
                        "end": 2.0,
                        "transforms": [{"op": "resize", "args": {"width": 400, "height": 250}}],
                        "effects": [],
                    }
                ]
            }
        )
        assert meta.width == 400
        assert meta.height == 250

    def test_crop(self):
        meta = self._run_plan(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 0,
                        "end": 2.0,
                        "transforms": [{"op": "crop", "args": {"width": 400, "height": 300, "mode": "center"}}],
                        "effects": [],
                    }
                ]
            }
        )
        assert meta.width == 400
        assert meta.height == 300

    def test_resample_fps(self):
        meta = self._run_plan(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 0,
                        "end": 2.0,
                        "transforms": [{"op": "resample_fps", "args": {"fps": 12}}],
                        "effects": [],
                    }
                ]
            }
        )
        assert abs(meta.fps - 12) < 1

    def test_speed_change_falls_back_to_eager(self):
        """speed_change is not streamable -- should fall back to eager and still work."""
        meta = self._run_plan(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 0,
                        "end": 4.0,
                        "transforms": [{"op": "speed_change", "args": {"speed": 2.0}}],
                        "effects": [],
                    }
                ]
            }
        )
        # 4s at 2x speed = ~2s output (via eager fallback)
        assert meta.total_seconds < 3.0

    def test_transforms_plus_effects(self):
        meta = self._run_plan(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 0,
                        "end": 2.0,
                        "transforms": [
                            {"op": "resize", "args": {"width": 400, "height": 250}},
                            {"op": "crop", "args": {"width": 380, "height": 230, "mode": "center"}},
                        ],
                        "effects": [
                            {"op": "color_adjust", "args": {"brightness": 0.1}},
                            {"op": "fade", "args": {"mode": "in", "duration": 0.5}},
                        ],
                    }
                ]
            }
        )
        assert meta.width == 380
        assert meta.height == 230


class TestStreamableEffects:
    """Verify all effects tagged streamable work via run_to_file."""

    def _run_effect(self, effect_dict, start=0, end=2.0):
        plan = {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": start,
                    "end": end,
                    "transforms": [],
                    "effects": [effect_dict],
                }
            ]
        }
        edit = VideoEdit.from_dict(plan)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            result_path = edit.run_to_file(out_path)
            assert result_path.exists()
            return Video.from_path(str(result_path))
        finally:
            out_path.unlink(missing_ok=True)

    def test_color_adjust(self):
        result = self._run_effect({"op": "color_adjust", "args": {"saturation": 0, "contrast": 1.2}})
        assert result.frames.shape[0] > 0

    def test_blur_constant(self):
        result = self._run_effect({"op": "blur_effect", "args": {"mode": "constant", "iterations": 5}})
        assert result.frames.shape[0] > 0

    def test_blur_ascending(self):
        result = self._run_effect({"op": "blur_effect", "args": {"mode": "ascending", "iterations": 10}})
        assert result.frames.shape[0] > 0

    def test_zoom_in(self):
        result = self._run_effect({"op": "zoom_effect", "args": {"zoom_factor": 1.5, "mode": "in"}})
        assert result.frames.shape[0] > 0

    def test_zoom_out(self):
        result = self._run_effect({"op": "zoom_effect", "args": {"zoom_factor": 1.5, "mode": "out"}})
        assert result.frames.shape[0] > 0

    def test_vignette(self):
        result = self._run_effect({"op": "vignette", "args": {"strength": 0.8, "radius": 1.0}})
        assert result.frames.shape[0] > 0

    def test_fade_in(self):
        result = self._run_effect({"op": "fade", "args": {"mode": "in", "duration": 0.5}})
        assert result.frames[0].mean() < 5

    def test_fade_out(self):
        result = self._run_effect({"op": "fade", "args": {"mode": "out", "duration": 0.5}})
        assert result.frames[-1].mean() < 5

    def test_fade_in_out_all_curves(self):
        for curve in ("sqrt", "linear", "exponential"):
            result = self._run_effect({"op": "fade", "args": {"mode": "in_out", "duration": 0.5, "curve": curve}})
            assert result.frames[0].mean() < 5

    def test_volume_adjust(self):
        result = self._run_effect({"op": "volume_adjust", "args": {"volume": 0.5}})
        assert result.frames.shape[0] > 0

    def test_text_overlay(self):
        result = self._run_effect({"op": "text_overlay", "args": {"text": "Test", "font_size": 24}})
        assert result.frames.shape[0] > 0

    def test_effect_with_time_range(self):
        """Effect with start/stop apply args should work in streaming."""
        plan = {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": 0,
                    "end": 4.0,
                    "transforms": [],
                    "effects": [
                        {"op": "fade", "args": {"mode": "in", "duration": 1.0}, "apply": {"start": 0, "stop": 2.0}},
                    ],
                }
            ]
        }
        edit = VideoEdit.from_dict(plan)
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            result_path = edit.run_to_file(out_path)
            assert result_path.exists()
        finally:
            out_path.unlink(missing_ok=True)


class TestFrameEncoder:
    """Test the FrameEncoder class."""

    def test_encode_synthetic_frames(self):
        """Encode a few synthetic frames and verify the output is valid."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            out_path = Path(f.name)
        try:
            with FrameEncoder(out_path, width=64, height=64, fps=10) as encoder:
                for i in range(10):
                    frame = np.full((64, 64, 3), i * 25, dtype=np.uint8)
                    encoder.write_frame(frame)
            assert out_path.exists()
            meta = VideoMetadata.from_path(str(out_path))
            assert meta.width == 64
            assert meta.height == 64
        finally:
            out_path.unlink(missing_ok=True)

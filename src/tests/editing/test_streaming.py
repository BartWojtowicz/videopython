"""Tests for the streaming video processing pipeline."""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.test_config import BIG_VIDEO_PATH, SMALL_VIDEO_PATH
from videopython.base.transcription import Transcription, TranscriptionWord
from videopython.base.video import Video, VideoMetadata
from videopython.editing import VideoEdit
from videopython.editing.effects import ColorGrading, Fade
from videopython.editing.streaming import EffectScheduleEntry, FrameEncoder, StreamingSegmentPlan, stream_segment


@pytest.fixture
def small_meta():
    return VideoMetadata.from_path(SMALL_VIDEO_PATH)


def _assert_op_changes_frame(render, operations, *, name):
    """Render ``operations`` over a SMALL_VIDEO cut and a no-op cut of the same
    source, then assert the op actually altered the middle frame end-to-end
    (guarding the silent-drop failure mode) without changing the frame count.
    Returns the edited :class:`Video` for any op-specific follow-up checks."""
    seg = {"source": SMALL_VIDEO_PATH, "start": 0, "end": 2.0}
    edited = render(VideoEdit.from_dict({"segments": [{**seg, "operations": operations}]}), name=name)
    plain = render(VideoEdit.from_dict({"segments": [seg]}), name="plain_" + name)
    assert abs(len(edited.frames) - len(plain.frames)) <= 1
    mid = min(len(edited.frames), len(plain.frames)) // 2
    mae = np.abs(edited.frames[mid].astype(np.float32) - plain.frames[mid].astype(np.float32)).mean()
    assert mae > 2.0, f"op did not change the rendered frame (mae={mae})"
    return edited


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
                    "operations": [
                        {"op": "color_adjust", "brightness": 0.2},
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
            assert result_path.stat().st_size > 0
            meta = VideoMetadata.from_path(str(result_path))
            assert meta.width == 800
            assert meta.height == 500
        finally:
            out_path.unlink(missing_ok=True)

    def test_color_adjust_changes_pixels(self, render):
        """color_adjust streams through run_to_file and actually alters the frame."""
        _assert_op_changes_frame(
            render, [{"op": "color_adjust", "saturation": 0, "contrast": 1.15}], name="coloradj.mp4"
        )

    def test_image_overlay_renders(self, render):
        """A PNG image_overlay streams and composites onto the frame."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            logo_path = Path(f.name)
        arr = np.zeros((80, 120, 4), dtype=np.uint8)
        arr[:, :, 1] = 255  # green
        arr[:, :, 3] = 180
        Image.fromarray(arr, "RGBA").save(logo_path)
        try:
            _assert_op_changes_frame(
                render,
                [{"op": "image_overlay", "source": str(logo_path), "scale": 0.25, "anchor": "bottom_right"}],
                name="overlay.mp4",
            )
        finally:
            logo_path.unlink(missing_ok=True)

    def test_svg_overlay_renders(self, render):
        """An SVG image_overlay rasterises and composites onto the frame."""
        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w") as f:
            f.write(
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 60" width="120" '
                'height="60"><rect width="120" height="60" fill="rgb(0,200,120)"/></svg>'
            )
            svg_path = Path(f.name)
        try:
            _assert_op_changes_frame(
                render,
                [{"op": "image_overlay", "source": str(svg_path), "scale": 0.25, "anchor": "bottom_right"}],
                name="svg_overlay.mp4",
            )
        finally:
            svg_path.unlink(missing_ok=True)

    def test_volume_adjust_streaming(self):
        """Audio-only effects should work in streaming mode."""
        plan = {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": 0,
                    "end": 2.0,
                    "operations": [
                        {"op": "volume_adjust", "volume": 1.5},
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
                    "operations": [
                        {"op": "color_adjust", "saturation": 0},
                        {"op": "volume_adjust", "volume": 1.6},
                        {"op": "fade", "mode": "in_out", "duration": 0.5},
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
                        "operations": [{"op": "resize", "width": 400, "height": 250}],
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
                        "operations": [{"op": "crop", "width": 400, "height": 300, "mode": "center"}],
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
                        "operations": [{"op": "resample_fps", "fps": 12}],
                    }
                ]
            }
        )
        assert abs(meta.fps - 12) < 1

    def test_speed_change_streams_natively(self):
        """speed_change compiles to setpts+fps (0.42.0) and streams natively."""
        meta = self._run_plan(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 0,
                        "end": 4.0,
                        "operations": [{"op": "speed_change", "speed": 2.0}],
                    }
                ]
            }
        )
        # 4s at 2x speed = ~2s output, compiled to setpts+fps (0.42.0)
        assert meta.total_seconds < 3.0

    def test_transforms_plus_effects(self):
        meta = self._run_plan(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 0,
                        "end": 2.0,
                        "operations": [
                            {"op": "resize", "width": 400, "height": 250},
                            {"op": "crop", "width": 380, "height": 230, "mode": "center"},
                            {"op": "color_adjust", "brightness": 0.1},
                            {"op": "fade", "mode": "in", "duration": 0.5},
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
                    "operations": [effect_dict],
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
        result = self._run_effect({"op": "color_adjust", "saturation": 0, "contrast": 1.2})
        assert result.frames.shape[0] > 0

    def test_blur_constant(self):
        result = self._run_effect({"op": "blur_effect", "mode": "constant", "iterations": 5})
        assert result.frames.shape[0] > 0

    def test_blur_ascending(self):
        result = self._run_effect({"op": "blur_effect", "mode": "ascending", "iterations": 10})
        assert result.frames.shape[0] > 0

    def test_zoom_in(self):
        result = self._run_effect({"op": "zoom_effect", "zoom_factor": 1.5, "mode": "in"})
        assert result.frames.shape[0] > 0

    def test_zoom_out(self):
        result = self._run_effect({"op": "zoom_effect", "zoom_factor": 1.5, "mode": "out"})
        assert result.frames.shape[0] > 0

    def test_vignette(self):
        result = self._run_effect({"op": "vignette", "strength": 0.8, "radius": 1.0})
        assert result.frames.shape[0] > 0

    def test_fade_in(self):
        result = self._run_effect({"op": "fade", "mode": "in", "duration": 0.5})
        assert result.frames[0].mean() < 5

    def test_fade_out(self):
        result = self._run_effect({"op": "fade", "mode": "out", "duration": 0.5})
        assert result.frames[-1].mean() < 5

    def test_fade_in_out_all_curves(self):
        for curve in ("sqrt", "linear", "exponential"):
            result = self._run_effect({"op": "fade", "mode": "in_out", "duration": 0.5, "curve": curve})
            assert result.frames[0].mean() < 5

    def test_volume_adjust(self):
        result = self._run_effect({"op": "volume_adjust", "volume": 0.5})
        assert result.frames.shape[0] > 0

    def test_text_overlay(self):
        result = self._run_effect({"op": "text_overlay", "text": "Test", "font_size": 24})
        assert result.frames.shape[0] > 0

    def test_image_overlay(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            logo_path = Path(f.name)
        try:
            arr = np.zeros((60, 60, 4), dtype=np.uint8)
            arr[:, :, 0] = 255
            arr[:, :, 3] = 200
            Image.fromarray(arr, "RGBA").save(logo_path)
            result = self._run_effect(
                {"op": "image_overlay", "source": str(logo_path), "scale": 0.2, "anchor": "bottom_right"}
            )
            assert result.frames.shape[0] > 0
        finally:
            logo_path.unlink(missing_ok=True)

    def test_shake(self):
        result = self._run_effect({"op": "shake", "intensity_px": 4.0, "mode": "random", "seed": 1})
        assert result.frames.shape[0] > 0

    def test_punch_in(self):
        result = self._run_effect({"op": "punch_in", "zoom_factor": 1.5, "attack_frames": 3})
        assert result.frames.shape[0] > 0

    def test_flash(self):
        result = self._run_effect(
            {"op": "flash", "color": [255, 255, 255], "peak_alpha": 0.8, "attack_frames": 2, "decay_frames": 4}
        )
        assert result.frames.shape[0] > 0

    def test_chromatic_aberration(self):
        result = self._run_effect({"op": "chromatic_aberration", "shift_px": 4, "mode": "horizontal"})
        assert result.frames.shape[0] > 0

    def test_glitch(self):
        result = self._run_effect({"op": "glitch", "intensity": 0.5, "seed": 7})
        assert result.frames.shape[0] > 0

    def test_film_grain(self):
        result = self._run_effect({"op": "film_grain", "intensity": 0.1, "seed": 0})
        assert result.frames.shape[0] > 0

    def test_sharpen(self):
        result = self._run_effect({"op": "sharpen", "amount": 1.0, "kernel_size": 5})
        assert result.frames.shape[0] > 0

    def test_pixelate(self):
        result = self._run_effect({"op": "pixelate", "block_size": 16})
        assert result.frames.shape[0] > 0

    def test_mirror_flip(self):
        result = self._run_effect({"op": "mirror_flip", "mode": "horizontal"})
        assert result.frames.shape[0] > 0

    def test_kaleidoscope(self):
        result = self._run_effect({"op": "kaleidoscope", "segments": 6})
        assert result.frames.shape[0] > 0

    def test_effect_with_window(self):
        """Effect with `window` should work in streaming."""
        plan = {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": 0,
                    "end": 4.0,
                    "operations": [
                        {
                            "op": "fade",
                            "mode": "in",
                            "duration": 1.0,
                            "window": {"start": 0, "stop": 2.0},
                        },
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


class TestContextStreaming:
    """Context-requiring ops (add_subtitles) run on the streaming path.

    The plan builder resolves `requires` context onto the segment-local
    timeline and delivers it at compile time (the subtitles op bakes it into
    its ASS document); the segment uses a mid-video cut (start > 0) so these
    tests fail if re-basing is skipped.
    """

    _PLAN = {
        "segments": [
            {
                "source": SMALL_VIDEO_PATH,
                "start": 4.0,
                "end": 8.0,
                "operations": [{"op": "add_subtitles", "font_scale": 0.1}],
            }
        ],
    }

    @staticmethod
    def _absolute_transcription() -> Transcription:
        # Source-absolute words overlapping the [4.0, 8.0) cut.
        return Transcription(
            words=[
                TranscriptionWord(start=4.5, end=5.5, word="hello"),
                TranscriptionWord(start=5.5, end=6.5, word="streaming"),
                TranscriptionWord(start=6.5, end=7.5, word="world"),
            ]
        )

    def test_add_subtitles_burns_in_and_rebases(self, render):
        """run_to_file burns subtitles in (not a silent context drop) and re-bases
        the transcription onto the cut segment's local timeline."""
        context = {"transcription": self._absolute_transcription()}
        subtitled = render(VideoEdit.from_dict(self._PLAN), name="subtitled.mp4", context=context)

        # Frame inside the first cue (segment-local t=1.0s == source t=5.0s).
        active = round(1.0 * 24)
        stream_frame = subtitled.frames[active].astype(np.float32)

        # The subtitle was actually drawn: against the raw (no-subtitle) cut,
        # a meaningful share of pixels changed by far more than codec noise.
        # Guards the failure mode where the context is silently dropped.
        baseline = Video.from_path(SMALL_VIDEO_PATH, start_second=4.0, end_second=8.0)
        drawn_diff = np.abs(baseline.frames[active].astype(np.float32) - stream_frame)
        drawn_fraction = (drawn_diff > 50).mean()
        assert drawn_fraction > 0.005, f"No subtitle pixels detected: {drawn_fraction}"

        # Re-base proof: before the first re-based cue (local t=0.1s; words
        # start at local 0.5s) the frame is the unmodified source. Absolute
        # (un-re-based) timestamps would draw here (4.5-7.5s overlaps 0.1s
        # only if timestamps were kept source-absolute -- they must not be).
        quiet = round(0.1 * 24)
        quiet_diff = np.abs(
            baseline.frames[quiet].astype(np.float32) - subtitled.frames[quiet].astype(np.float32)
        ).mean()
        assert quiet_diff < 15, f"Frame before first cue was modified: {quiet_diff}"

    def test_missing_context_raises_before_decode(self, tmp_path):
        """No transcription in context -> the op's own clear error, pre-decode."""
        edit = VideoEdit.from_dict(self._PLAN)
        with pytest.raises(ValueError, match="requires transcription data"):
            edit.run_to_file(tmp_path / "out.mp4")

    def test_no_overlap_context_raises(self, tmp_path):
        """Words entirely outside the cut are dropped by re-basing -> same error."""
        context = {
            "transcription": Transcription(words=[TranscriptionWord(start=50.0, end=51.0, word="late")]),
        }
        edit = VideoEdit.from_dict(self._PLAN)
        with pytest.raises(ValueError, match="requires transcription data"):
            edit.run_to_file(tmp_path / "out.mp4", context=context)

    def test_subtitles_then_transform_streams_in_plan_order(self, render):
        """A transform after add_subtitles streams: both join the decode filter
        chain in plan order (subtitles burn at pre-crop dims, then the crop
        applies), so the rendered output is the cropped size.
        """
        plan = {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": 4.0,
                    "end": 8.0,
                    "operations": [
                        {"op": "add_subtitles", "font_scale": 0.1},
                        {"op": "crop", "width": 400, "height": 300},
                    ],
                }
            ],
        }
        context = {"transcription": self._absolute_transcription()}
        rendered = render(VideoEdit.from_dict(plan), name="subs_then_crop.mp4", context=context)
        assert rendered.frames.shape[1:3] == (300, 400)


class TestPerSourceContextStreaming:
    """A per-source transcription map feeds each segment its OWN transcription.

    The two segments cut from different sources; the pre-0.43 global context
    would apply one transcription to both. These tests pin per-source keying
    end-to-end through the streaming engine.
    """

    @staticmethod
    def _plan():
        return {
            "segments": [
                {
                    "source": SMALL_VIDEO_PATH,
                    "start": 4.0,
                    "end": 8.0,
                    "operations": [{"op": "add_subtitles", "font_scale": 0.1}],
                },
                {
                    "source": BIG_VIDEO_PATH,
                    "start": 2.0,
                    "end": 6.0,
                    "operations": [{"op": "add_subtitles", "font_scale": 0.1}],
                },
            ],
        }

    def test_missing_source_in_map_raises_for_that_segment(self, tmp_path):
        """The map supplies only segment 0's source; segment 1 must not inherit it."""
        tx = Transcription(words=[TranscriptionWord(start=4.5, end=5.5, word="hi")])
        context = {"transcription": {SMALL_VIDEO_PATH: tx}}
        with pytest.raises(ValueError, match="requires transcription data"):
            VideoEdit.from_dict(self._plan()).run_to_file(tmp_path / "out.mp4", context=context)

    def test_per_source_map_renders_both_segments(self, tmp_path):
        """Both sources keyed -> the multi-clip plan streams end to end."""
        context = {
            "transcription": {
                SMALL_VIDEO_PATH: Transcription(words=[TranscriptionWord(start=4.5, end=5.5, word="hi")]),
                BIG_VIDEO_PATH: Transcription(words=[TranscriptionWord(start=2.5, end=3.5, word="yo")]),
            }
        }
        out = tmp_path / "out.mp4"
        VideoEdit.from_dict(self._plan()).run_to_file(out, context=context)
        assert out.exists()
        assert Video.from_path(str(out)).frames.shape[0] > 0

    def test_broadcast_value_applies_to_all_sources(self, tmp_path):
        """A bare (non-dict) transcription still broadcasts to every segment (back-compat)."""
        tx = Transcription(
            words=[
                TranscriptionWord(start=4.5, end=5.5, word="hi"),
                TranscriptionWord(start=2.5, end=3.5, word="yo"),
            ]
        )
        out = tmp_path / "out.mp4"
        VideoEdit.from_dict(self._plan()).run_to_file(out, context={"transcription": tx})
        assert out.exists()
        assert Video.from_path(str(out)).frames.shape[0] > 0


class TestTrailingFrameSchedule:
    """Frames past the round(duration*fps) estimate still get full-range effects."""

    def test_fade_out_covers_rounding_tie_extra_frame(self, tmp_path):
        # At 10 fps, the cut [1.0, 4.05] schedules round(3.05 * 10) = 30
        # frames while ffmpeg can emit 31 on the rounding tie. Without
        # open-ended schedule entries the extra frame escaped the fade-out
        # and popped back to full brightness on the last frame.
        src_10fps = tmp_path / "src_10fps.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", SMALL_VIDEO_PATH, "-r", "10", "-an", str(src_10fps)],
            check=True,
        )
        plan = {
            "segments": [
                {
                    "source": str(src_10fps),
                    "start": 1.0,
                    "end": 4.05,
                    "operations": [{"op": "fade", "mode": "out", "duration": 1.0}],
                }
            ],
        }
        out_path = tmp_path / "faded.mp4"
        VideoEdit.from_dict(plan).run_to_file(out_path)
        result = Video.from_path(str(out_path))
        assert result.frames[-1].mean() < 5, f"Last frame escaped the fade: {result.frames[-1].mean()}"

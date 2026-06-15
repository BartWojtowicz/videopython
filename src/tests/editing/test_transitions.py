"""Tests for segment transitions (xfade/acrossfade) -- P1.10.

A transition is a native ffmpeg ``xfade`` (video) + ``acrossfade``/butt-join
(audio) pass over two realized segment files. The in-memory ``run()`` twin and
the ``run_to_file`` file path share one xfade filter-string builder, so the
seam pixels match (lossy encode aside). These tests pin the duration math, the
seam blend, the audio length, the hard-cut/transition grouping, and the
``TRANSITION_TOO_LONG`` reporting/raising/repair.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from videopython.audio import Audio
from videopython.audio.audio import AudioMetadata
from videopython.base.exceptions import PlanErrorCode, PlanValidationError
from videopython.base.video import Video, VideoMetadata
from videopython.editing import StreamingClass
from videopython.editing.streaming import TRANSITION_TYPES, stream_transition_frames, xfade_filter
from videopython.editing.video_edit import TransitionSpec, VideoEdit

FPS = 24.0
W = H = 64


def _solid(color: tuple[int, int, int], seconds: float, *, tone_hz: float | None = None) -> Video:
    """A solid-color clip; silent unless ``tone_hz`` is given (then a stereo sine)."""
    n = round(FPS * seconds)
    frames = np.zeros((n, H, W, 3), dtype=np.uint8)
    frames[:] = color
    video = Video.from_frames(frames, fps=FPS)
    if tone_hz is None:
        video.audio = Audio.create_silent(seconds, stereo=True, sample_rate=44100)
    else:
        sr = 44100
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        wave = (0.3 * np.sin(2 * np.pi * tone_hz * t)).astype(np.float32)
        data = np.stack([wave, wave], axis=1)
        meta = AudioMetadata(
            sample_rate=sr, channels=2, sample_width=2, duration_seconds=seconds, frame_count=len(data)
        )
        video.audio = Audio(data, meta)
    return video


@pytest.fixture
def clips(tmp_path: Path) -> dict[str, str]:
    """Three saved solid clips (silent) keyed by name -> path string."""
    out: dict[str, str] = {}
    for name, color in (("red", (255, 0, 0)), ("green", (0, 255, 0)), ("blue", (0, 0, 255))):
        p = tmp_path / f"{name}.mp4"
        _solid(color, 1.5).save(str(p))
        out[name] = str(p)
    return out


def _metas(clips: dict[str, str]) -> dict[str, VideoMetadata]:
    return {p: VideoMetadata.from_path(p) for p in clips.values()}


# ----------------------------------------------------------------- plan surface


class TestPlanSurface:
    def test_transition_spec_is_frozen_and_closed(self):
        spec = TransitionSpec(type="fade", duration=0.5)
        assert spec.audio is True
        with pytest.raises(Exception):
            spec.duration = 1.0  # frozen
        with pytest.raises(Exception):
            TransitionSpec(type="fade", duration=0.5, bogus=1)  # extra forbidden

    def test_duration_must_be_positive(self):
        with pytest.raises(Exception):
            TransitionSpec(type="fade", duration=0.0)

    def test_type_constrained_to_catalog(self):
        with pytest.raises(Exception):
            TransitionSpec(type="not_a_transition", duration=0.5)

    def test_literal_matches_catalog(self):
        # Every exposed type round-trips through the model.
        for t in TRANSITION_TYPES:
            assert TransitionSpec(type=t, duration=0.3).type == t

    def test_transition_in_defaults_none(self, clips):
        edit = VideoEdit.from_dict({"segments": [{"source": clips["red"], "start": 0, "end": 1.5}]})
        assert edit.segments[0].transition_in is None

    def test_schema_includes_transition_object(self):
        schema = VideoEdit.json_schema()
        seg = schema["properties"]["segments"]["items"]
        assert "transition_in" in seg["properties"]
        any_of = seg["properties"]["transition_in"]["anyOf"]
        obj = next(o for o in any_of if o.get("type") == "object")
        assert set(obj["properties"]["type"]["enum"]) == set(TRANSITION_TYPES)
        assert obj["properties"]["duration"]["exclusiveMinimum"] == 0
        assert obj["additionalProperties"] is False

    def test_strict_schema_round_trips(self):
        # The strict closing pass must not choke on the inlined transition object.
        schema = VideoEdit.json_schema(strict=True)
        seg = schema["properties"]["segments"]["items"]
        assert "transition_in" in seg["properties"]


# ----------------------------------------------------------------- duration math


class TestDurationMath:
    def test_run_frame_count_subtracts_overlap(self, clips):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5},
                    },
                ]
            }
        )
        result = plan.run()
        overlap = round(0.5 * FPS)
        assert result.frames.shape[0] == 36 + 36 - overlap

    def test_run_to_file_frame_count_matches(self, clips, tmp_path):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "dissolve", "duration": 0.5},
                    },
                ]
            }
        )
        out = plan.run_to_file(tmp_path / "out.mp4")
        meta = VideoMetadata.from_path(str(out))
        overlap = round(0.5 * FPS)
        assert meta.frame_count == 36 + 36 - overlap

    def test_predicted_assembled_timeline_matches_realized(self, clips, tmp_path):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5},
                    },
                ]
            }
        )
        predicted = plan.validate_with_metadata(_metas(clips)).frame_count
        out = plan.run_to_file(tmp_path / "out.mp4")
        assert VideoMetadata.from_path(str(out)).frame_count == predicted


# ------------------------------------------------------------------- seam pixels


class TestSeamPixels:
    @pytest.mark.parametrize("ttype", list(TRANSITION_TYPES))
    def test_endpoints_pure_overlap_mixed_in_memory(self, ttype):
        left = _solid((255, 0, 0), 1.5)
        right = _solid((0, 0, 255), 1.5)
        blended = stream_transition_frames(left.frames, right.frames, ttype, 0.5, FPS)
        assert blended.shape[0] == 36 + 36 - 12
        # The pre- and post-overlap endpoints stay pure source.
        assert tuple(blended[0, 0, 0]) == (255, 0, 0)
        assert tuple(blended[-1, 0, 0]) == (0, 0, 255)
        # A mid-overlap frame is neither uniformly the left nor the right
        # source: a fade/dissolve alpha-blends every pixel, a wipe/slide shows
        # part of each source. Either way the frame is a genuine mix.
        mid_frame = blended[(36 - 12) + 6]
        pure_red = np.all(mid_frame == np.array([255, 0, 0], dtype=np.uint8))
        pure_blue = np.all(mid_frame == np.array([0, 0, 255], dtype=np.uint8))
        assert not pure_red and not pure_blue

    def test_fade_midpoint_is_a_blend(self):
        left = _solid((255, 0, 0), 1.5)
        right = _solid((0, 0, 255), 1.5)
        blended = stream_transition_frames(left.frames, right.frames, "fade", 0.5, FPS)
        mid = (36 - 12) + 6  # halfway through the 12-frame overlap
        r, g, b = (int(x) for x in blended[mid, H // 2, W // 2])
        assert 90 < r < 165 and 90 < b < 165 and g < 20

    def test_run_and_run_to_file_share_builder(self, clips, tmp_path):
        # The in-memory twin (lossless) and the file path (lossy encode) come
        # from the same xfade filter, so the seam blend is close, not identical.
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5},
                    },
                ]
            }
        )
        in_mem = plan.run()
        out = plan.run_to_file(tmp_path / "out.mp4")
        on_disk = Video.from_path(str(out))
        assert in_mem.frames.shape == on_disk.frames.shape
        mid = (36 - 12) + 6
        a = in_mem.frames[mid, H // 2, W // 2].astype(int)
        b = on_disk.frames[mid, H // 2, W // 2].astype(int)
        assert np.max(np.abs(a - b)) <= 12  # within x264 quantization at the seam

    def test_builder_is_shared_string(self):
        # One builder produces the filter both paths route through.
        assert xfade_filter("fade", 0.5, 1.0) == "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=1.0"
        assert xfade_filter("wipeleft", 0.5, 2.5, in_a="lv", in_b="rv").startswith("[lv][rv]xfade=transition=wipeleft")


# ------------------------------------------------------------------------- audio


class TestAudio:
    def test_crossfade_audio_tracks_shortened_timeline(self, tmp_path):
        red = tmp_path / "red.mp4"
        blue = tmp_path / "blue.mp4"
        _solid((255, 0, 0), 1.5, tone_hz=440).save(str(red))
        _solid((0, 0, 255), 1.5, tone_hz=880).save(str(blue))
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": str(red), "start": 0, "end": 1.5},
                    {
                        "source": str(blue),
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5, "audio": True},
                    },
                ]
            }
        )
        out = plan.run_to_file(tmp_path / "out.mp4")
        vmeta = VideoMetadata.from_path(str(out))
        audio = Audio.from_path(str(out))
        # 2.5s shortened timeline; AAC priming/padding keeps this within ~50ms.
        assert abs(audio.metadata.duration_seconds - vmeta.total_seconds) < 0.1
        assert abs(vmeta.total_seconds - 2.5) < 0.05

    def test_run_audio_fits_shortened_video(self, tmp_path):
        red = tmp_path / "red.mp4"
        blue = tmp_path / "blue.mp4"
        _solid((255, 0, 0), 1.5, tone_hz=440).save(str(red))
        _solid((0, 0, 255), 1.5, tone_hz=880).save(str(blue))
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": str(red), "start": 0, "end": 1.5},
                    {
                        "source": str(blue),
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5},
                    },
                ]
            }
        )
        result = plan.run()
        assert abs(result.audio.metadata.duration_seconds - result.total_seconds) < 1e-3

    def test_audio_false_hard_butt_join_same_length(self, tmp_path):
        red = tmp_path / "red.mp4"
        blue = tmp_path / "blue.mp4"
        _solid((255, 0, 0), 1.5, tone_hz=440).save(str(red))
        _solid((0, 0, 255), 1.5, tone_hz=880).save(str(blue))
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": str(red), "start": 0, "end": 1.5},
                    {
                        "source": str(blue),
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5, "audio": False},
                    },
                ]
            }
        )
        out = plan.run_to_file(tmp_path / "out.mp4")
        vmeta = VideoMetadata.from_path(str(out))
        audio = Audio.from_path(str(out))
        assert abs(audio.metadata.duration_seconds - vmeta.total_seconds) < 0.1

    def test_silent_segment_butt_joins(self, tmp_path):
        # One silent segment -> no crossfade even with audio=True; still fits.
        red = tmp_path / "red.mp4"
        blue = tmp_path / "blue.mp4"
        _solid((255, 0, 0), 1.5).save(str(red))  # silent
        _solid((0, 0, 255), 1.5, tone_hz=880).save(str(blue))
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": str(red), "start": 0, "end": 1.5},
                    {
                        "source": str(blue),
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5, "audio": True},
                    },
                ]
            }
        )
        result = plan.run()
        assert abs(result.audio.metadata.duration_seconds - result.total_seconds) < 1e-3


# --------------------------------------------------------------------- assembly


class TestMixedAssembly:
    def test_mixed_plan_hardcut_and_transition(self, clips, tmp_path):
        # [seg0, seg1 hard-cut, seg2 transition]: only the seam re-encodes; the
        # hard-cut run concat-copies. Frame count = 36*3 - overlap.
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {"source": clips["green"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "wipeleft", "duration": 0.5},
                    },
                ]
            }
        )
        predicted = plan.validate_with_metadata(_metas(clips)).frame_count
        out = plan.run_to_file(tmp_path / "out.mp4")
        realized = VideoMetadata.from_path(str(out)).frame_count
        assert realized == predicted == 36 * 3 - round(0.5 * FPS)

    def test_all_transition_reel(self, clips, tmp_path):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["green"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5},
                    },
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5},
                    },
                ]
            }
        )
        predicted = plan.validate_with_metadata(_metas(clips)).frame_count
        out = plan.run_to_file(tmp_path / "out.mp4")
        assert VideoMetadata.from_path(str(out)).frame_count == predicted == 36 * 3 - 2 * round(0.5 * FPS)


# ------------------------------------------------------------- errors and repair


class TestTransitionTooLong:
    def test_check_reports_too_long(self, clips):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 2.0},
                    },
                ]
            }
        )
        codes = [e.code for e in plan.check(_metas(clips))]
        assert PlanErrorCode.TRANSITION_TOO_LONG in codes

    def test_run_to_file_raises_before_decode(self, clips, tmp_path):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 2.0},
                    },
                ]
            }
        )
        with pytest.raises(PlanValidationError) as exc:
            plan.run_to_file(tmp_path / "out.mp4")
        assert exc.value.errors[0].code is PlanErrorCode.TRANSITION_TOO_LONG

    def test_run_raises(self, clips):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 2.0},
                    },
                ]
            }
        )
        with pytest.raises(PlanValidationError) as exc:
            plan.run()
        assert exc.value.errors[0].code is PlanErrorCode.TRANSITION_TOO_LONG

    def test_first_segment_transition_rejected(self, clips):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {
                        "source": clips["red"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.3},
                    },
                    {"source": clips["blue"], "start": 0, "end": 1.5},
                ]
            }
        )
        codes = [e.code for e in plan.check(_metas(clips))]
        assert PlanErrorCode.TRANSITION_TOO_LONG in codes
        with pytest.raises(PlanValidationError):
            plan.run()

    def test_repair_clamps_duration(self, clips):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 2.0},
                    },
                ]
            }
        )
        metas = _metas(clips)
        repaired, repairs = plan.repair(metas)
        spec = repaired.segments[1].transition_in
        assert any(r.code is PlanErrorCode.TRANSITION_TOO_LONG for r in repairs)
        # Frame-safe clamp: strictly fewer overlap frames than the 36-frame
        # segment, NOT the segment length itself (which rounds to a full overlap).
        assert round(spec.duration * FPS) == 35
        assert spec.duration < 1.5
        # The clamped plan is clean AND actually runs -- the regression: a
        # seconds clamp to the segment length passed check() but crashed run().
        assert repaired.check(metas) == []
        result = repaired.run()
        assert result.frames.shape[0] == 36 + 36 - 35

    def test_near_full_overlap_band_is_rejected(self, clips, tmp_path):
        """A duration just under the segment length but rounding to a full-frame
        overlap (round(D*fps) == segment_frames) must be rejected by every path.

        Regression: the guard used to compare seconds, so duration=1.49s
        (round(1.49*24)=36 == the 36-frame segment) slipped through check while
        run() crashed with a bare ValueError and run_to_file silently produced a
        degenerate full-overlap clip with mismatched audio/video length.
        """
        assert round(1.49 * FPS) == 36  # full overlap of a 36-frame segment
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 1.49},
                    },
                ]
            }
        )
        metas = _metas(clips)
        assert any(e.code is PlanErrorCode.TRANSITION_TOO_LONG for e in plan.check(metas))
        with pytest.raises(PlanValidationError) as run_exc:
            plan.run()
        assert run_exc.value.errors[0].code is PlanErrorCode.TRANSITION_TOO_LONG
        with pytest.raises(PlanValidationError) as file_exc:
            plan.run_to_file(tmp_path / "out.mp4")
        assert file_exc.value.errors[0].code is PlanErrorCode.TRANSITION_TOO_LONG

    def test_repair_leaves_first_segment_transition_for_check(self, clips):
        # A first-segment transition is structural, not a mechanical clamp.
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {
                        "source": clips["red"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.3},
                    },
                    {"source": clips["blue"], "start": 0, "end": 1.5},
                ]
            }
        )
        repaired, _ = plan.repair(_metas(clips))
        assert repaired.segments[0].transition_in is not None
        assert any(e.code is PlanErrorCode.TRANSITION_TOO_LONG for e in repaired.check(_metas(clips)))


# ------------------------------------------------------------------ streamability


class TestStreamability:
    def test_transition_plan_is_streamable(self, clips):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5},
                    },
                ]
            }
        )
        report = plan.streamability()
        assert report.streamable is True

    def test_post_op_with_transition_is_unstreamable(self, clips):
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {"source": clips["red"], "start": 0, "end": 1.5},
                    {
                        "source": clips["blue"],
                        "start": 0,
                        "end": 1.5,
                        "transition_in": {"type": "fade", "duration": 0.5},
                    },
                ],
                "post_operations": [{"op": "fade", "mode": "in", "duration": 0.3}],
            }
        )
        report = plan.streamability()
        assert report.streamable is False
        assert report.fallbacks[0].streaming_class is StreamingClass.UNSTREAMABLE
        assert "transition" in report.fallbacks[0].reason

"""Tests for natively compiled duration-changing transforms (P0.3).

``speed_change`` and ``freeze_frame`` compile to ffmpeg filter chains
(``setpts``/``fps``/``framerate``, ``loop``/``select``) instead of forcing
the whole-plan eager fallback; the plan builder folds real metadata through
the chain so frame counts, effect ranges, and the in-memory audio stay in
sync with the filtered output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.test_config import SMALL_VIDEO_PATH, TEST_AUDIO_PATH
from videopython.base.exceptions import PlanValidationError
from videopython.base.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video
from videopython.editing import StreamingClass, VideoEdit

FPS = 24
SEGMENT = {"start": 2.0, "end": 8.0}  # 6 s cut -> 144 frames at 24 fps


def _plan(operations: list[dict[str, Any]], source: str = SMALL_VIDEO_PATH) -> VideoEdit:
    return VideoEdit.model_validate({"segments": [{"source": source, **SEGMENT, "operations": operations}]})


def _stream(plan: VideoEdit, tmp_path, name: str = "out.mp4") -> Video:
    """run_to_file (streaming is the only engine since 0.42.0)."""
    out = plan.run_to_file(tmp_path / name)
    return Video.from_path(str(out))


@pytest.fixture(scope="module")
def audio_source(tmp_path_factory) -> str:
    """The small test video with a real audio track muxed in."""
    video = Video.from_path(SMALL_VIDEO_PATH).add_audio_from_file(TEST_AUDIO_PATH)
    out = tmp_path_factory.mktemp("native_transforms") / "with_audio.mp4"
    return str(video.save(out))


class TestSpeedChangeStreaming:
    def test_constant_speedup_streams_with_exact_frame_count(self, tmp_path):
        plan = _plan([{"op": "speed_change", "speed": 2.0}])
        assert plan.streamability().entries[0].streaming_class is StreamingClass.FILTER

        video = _stream(plan, tmp_path)
        assert len(video.frames) == 72  # int(144 / 2.0), exact

    def test_speedup_samples_the_right_source_frames(self, tmp_path):
        """Output frame k must show source content from around frame 2k."""
        video = _stream(_plan([{"op": "speed_change", "speed": 2.0}]), tmp_path)
        source = Video.from_path(SMALL_VIDEO_PATH, start_second=2.0, end_second=8.0)

        k = 30  # output frame 30 should match source frame ~60, not ~30
        out_frame = video.frames[k].astype(np.float32)
        right = np.abs(out_frame - source.frames[60].astype(np.float32)).mean()
        wrong = np.abs(out_frame - source.frames[30].astype(np.float32)).mean()
        assert right < wrong, f"speed mapping off: MAE to 2k={right:.1f}, to k={wrong:.1f}"

    def test_slowdown_streams_with_interpolation(self, tmp_path):
        plan = _plan([{"op": "speed_change", "speed": 0.5}])
        video = _stream(plan, tmp_path)
        assert abs(len(video.frames) - 288) <= 1  # int(144 / 0.5), +-1 EOF rounding

    def test_ramp_streams_and_matches_predicted_count(self, tmp_path):
        plan = _plan([{"op": "speed_change", "speed": 1.0, "end_speed": 3.0}])
        video = _stream(plan, tmp_path)
        assert abs(len(video.frames) - 72) <= 1  # int(144 / avg(1,3))

    def test_ramp_accelerates(self, tmp_path):
        """Early output advances the source slower than late output.

        Uses a synthetic source whose frame i is the uniform gray level i, so
        each output frame's mean intensity reads back the source index it
        sampled -- the time-warp curve measured directly.
        """
        frames = np.zeros((144, 64, 64, 3), dtype=np.uint8)
        for i in range(144):
            frames[i] = i
        src = str(Video.from_frames(frames, fps=24).save(tmp_path / "ramp_src.mp4", crf=0))

        plan = VideoEdit.model_validate(
            {
                "segments": [
                    {
                        "source": src,
                        "start": 0.0,
                        "end": 6.0,
                        "operations": [{"op": "speed_change", "speed": 1.0, "end_speed": 3.0, "interpolate": False}],
                    }
                ]
            }
        )
        video = _stream(plan, tmp_path, "ramp_idx.mp4")

        sampled_index = video.frames.reshape(len(video.frames), -1).mean(axis=1)
        early_advance = sampled_index[20] - sampled_index[10]
        late_advance = sampled_index[65] - sampled_index[55]
        assert late_advance > early_advance + 5, (early_advance, late_advance)
        assert sampled_index[-1] > 130, "ramp did not span the full source"

    def test_strong_slowdown_matches_predicted_exactly(self, tmp_path):
        """Regression: a negative setpts bias dropped head frames (k<1)."""
        plan = VideoEdit.model_validate(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 2.0,
                        "end": 4.0,
                        "operations": [{"op": "speed_change", "speed": 0.05, "interpolate": False}],
                    }
                ]
            }
        )
        video = _stream(plan, tmp_path)
        assert len(video.frames) == 960  # int(48 / 0.05), exact

    def test_zero_frame_speed_raises_before_decode(self, tmp_path):
        plan = _plan([{"op": "speed_change", "speed": 1000.0}])
        with pytest.raises(ValueError, match="0 frames"):
            plan.run_to_file(tmp_path / "out.mp4")


class TestFreezeFrameStreaming:
    def test_insert_freeze_extends_and_holds(self, tmp_path):
        plan = _plan([{"op": "freeze_frame", "timestamp": 1.0, "duration": 0.5}])
        assert plan.streamability().entries[0].streaming_class is StreamingClass.FILTER

        video = _stream(plan, tmp_path)
        assert abs(len(video.frames) - 156) <= 1  # 144 + round(0.5 * 24)

        # Frames 24..36 hold frame 24's content (codec noise only).
        held = video.frames[24:36].astype(np.float32)
        spread = np.abs(held - held[0]).mean()
        assert spread < 3.0, f"freeze region not held: {spread}"
        # And the video did not stall: content after the freeze differs.
        after = np.abs(video.frames[40].astype(np.float32) - held[0]).mean()
        assert after > spread

    def test_replace_freeze_keeps_duration(self, tmp_path):
        plan = _plan([{"op": "freeze_frame", "timestamp": 1.0, "duration": 0.5, "position": "replace"}])
        video = _stream(plan, tmp_path)
        assert abs(len(video.frames) - 144) <= 1

        held = video.frames[24:36].astype(np.float32)
        assert np.abs(held - held[0]).mean() < 3.0
        # Replace skips the covered originals: playback resumes past them.
        source = Video.from_path(SMALL_VIDEO_PATH, start_second=2.0, end_second=8.0)
        resumed = video.frames[40].astype(np.float32)
        ahead = np.abs(resumed - source.frames[40].astype(np.float32)).mean()
        behind = np.abs(resumed - source.frames[28].astype(np.float32)).mean()
        assert ahead < behind, "replace mode did not skip the replaced originals"

    def test_replace_freeze_after_resample_fps(self, tmp_path):
        """Regression: without its own trailing resampler the replace chain
        re-duplicated every post-freeze frame whenever an fps= filter ran
        earlier (FrameIterator suppresses its trailing fps= append then)."""
        plan = _plan(
            [
                {"op": "resample_fps", "fps": 12},
                {"op": "freeze_frame", "timestamp": 1.0, "duration": 0.5, "position": "replace"},
            ]
        )
        video = _stream(plan, tmp_path)
        assert abs(len(video.frames) - 72) <= 1  # 6s at 12fps, duration unchanged

    def test_boundary_freeze_agrees_across_paths(self, tmp_path):
        """Regression: a timestamp in the last half-frame window rounded to
        frame_count, where eager silently dropped the freeze while streaming
        held the last frame -- all paths now clamp to the last frame."""
        ops = [{"op": "freeze_frame", "timestamp": 5.99, "duration": 0.5}]
        eager = _plan(ops).run()
        video = _stream(_plan(ops), tmp_path)
        assert len(eager.frames) == 156
        assert abs(len(video.frames) - 156) <= 1

    def test_sub_frame_segment_raises_structured_error(self, tmp_path):
        plan = VideoEdit.model_validate(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 1.0,
                        "end": 1.01,
                        "operations": [{"op": "freeze_frame", "timestamp": 0.0, "duration": 0.5}],
                    }
                ]
            }
        )
        with pytest.raises(ValueError, match="shorter than one frame"):
            plan.run_to_file(tmp_path / "out.mp4")

    def test_timestamp_out_of_range_raises_before_decode(self, tmp_path):
        plan = _plan([{"op": "freeze_frame", "timestamp": 30.0, "duration": 0.5}])
        with pytest.raises(ValueError, match="must be less than video duration"):
            plan.run_to_file(tmp_path / "out.mp4")


class TestDurationFoldIntegration:
    def test_fade_out_after_speedup_covers_the_new_end(self, tmp_path):
        """The fade envelope must be sized to the folded frame count."""
        plan = _plan(
            [
                {"op": "speed_change", "speed": 2.0},
                {"op": "fade", "mode": "out", "duration": 1.0},
            ]
        )
        video = _stream(plan, tmp_path)
        assert video.frames[-1].mean() < 5, "fade-out did not reach black at the sped-up end"

    def test_subtitles_before_speed_streams(self, tmp_path):
        words = [
            TranscriptionWord(word="hello", start=3.0, end=4.0),
            TranscriptionWord(word="world", start=4.0, end=5.0),
        ]
        tr = Transcription(
            segments=[TranscriptionSegment(text="hello world", start=3.0, end=5.0, words=words)],
            language="en",
        )
        plan = _plan([{"op": "add_subtitles", "font_scale": 0.1}, {"op": "speed_change", "speed": 2.0}])
        report = plan.streamability()
        assert [e.streaming_class for e in report.entries] == [StreamingClass.FILTER, StreamingClass.FILTER]

        out = plan.run_to_file(tmp_path / "subs_speed.mp4", context={"transcription": tr})
        assert Video.from_path(str(out)).frames.shape[0] == 72

    def test_subtitles_after_speed_is_rejected(self, tmp_path):
        plan = _plan([{"op": "speed_change", "speed": 2.0}, {"op": "add_subtitles", "font_scale": 0.1}])
        report = plan.streamability()
        subtitles = report.entries[1]
        assert subtitles.streaming_class is StreamingClass.UNSTREAMABLE
        assert subtitles.reason is not None and "duration-changing" in subtitles.reason
        assert not report.streamable
        with pytest.raises(PlanValidationError, match="cannot stream"):
            plan.run_to_file(tmp_path / "out.mp4")


class TestAudioSync:
    def test_speedup_audio_matches_video_duration(self, tmp_path, audio_source):
        plan = _plan([{"op": "speed_change", "speed": 2.0}], source=audio_source)
        video = _stream(plan, tmp_path)
        assert video.audio is not None
        video_duration = len(video.frames) / video.fps
        assert abs(video.audio.metadata.duration_seconds - video_duration) < 0.15

    def test_freeze_audio_matches_video_duration_and_goes_silent(self, tmp_path, audio_source):
        plan = _plan([{"op": "freeze_frame", "timestamp": 1.0, "duration": 1.0}], source=audio_source)
        video = _stream(plan, tmp_path)
        assert video.audio is not None
        video_duration = len(video.frames) / video.fps
        assert abs(video.audio.metadata.duration_seconds - video_duration) < 0.15

        # The freeze window holds silence; just after it the track resumes.
        sr = video.audio.metadata.sample_rate
        data = np.abs(video.audio.data)
        inside = data[int(1.3 * sr) : int(1.7 * sr)].mean()
        outside = data[int(0.2 * sr) : int(0.8 * sr)].mean()
        assert inside < outside * 0.2, f"freeze window not silent: {inside} vs {outside}"


class TestSilenceRemovalStreaming:
    @staticmethod
    def _gapped_transcription() -> Transcription:
        """Speech in [2.5, 4.5] and [7.0, 8.0] source time; silence between."""
        return Transcription(
            words=[
                TranscriptionWord(word="a", start=2.5, end=3.5),
                TranscriptionWord(word="b", start=3.5, end=4.5),
                TranscriptionWord(word="c", start=7.0, end=8.0),
            ]
        )

    def test_streams_and_cuts_the_gap(self, tmp_path):
        plan = _plan([{"op": "silence_removal", "min_silence_duration": 1.0, "padding": 0.0}])
        assert plan.streamability().entries[0].streaming_class is StreamingClass.FILTER

        context = {"transcription": self._gapped_transcription()}
        eager = _plan([{"op": "silence_removal", "min_silence_duration": 1.0, "padding": 0.0}]).run(context=context)
        out = plan.run_to_file(tmp_path / "out.mp4", context=context)
        video = Video.from_path(str(out))

        # Keep [0, 2.5) + [5.0, 6.0) of the 6s cut = 3.5s = 84 frames.
        assert len(video.frames) == 84
        assert len(eager.frames) == 84

    def test_audio_is_cut_in_sync(self, tmp_path, audio_source):
        plan = _plan(
            [{"op": "silence_removal", "min_silence_duration": 1.0, "padding": 0.0}],
            source=audio_source,
        )
        out = plan.run_to_file(tmp_path / "out.mp4", context={"transcription": self._gapped_transcription()})
        video = Video.from_path(str(out))
        assert video.audio is not None
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < 0.15

    def test_missing_context_raises_at_compile(self, tmp_path):
        plan = _plan([{"op": "silence_removal"}])
        with pytest.raises(ValueError, match="requires transcription data"):
            plan.run_to_file(tmp_path / "out.mp4")

    def test_after_duration_change_is_rejected(self):
        report = _plan([{"op": "speed_change", "speed": 2.0}, {"op": "silence_removal"}]).streamability()
        silence = report.entries[1]
        assert silence.streaming_class is StreamingClass.UNSTREAMABLE
        assert silence.reason is not None and "duration-changing" in silence.reason

    def test_no_silence_compiles_to_noop(self, tmp_path):
        dense = Transcription(words=[TranscriptionWord(word=str(i), start=2.0 + i, end=3.0 + i) for i in range(6)])
        plan = _plan([{"op": "silence_removal", "min_silence_duration": 1.0, "padding": 0.0}])
        out = plan.run_to_file(tmp_path / "out.mp4", context={"transcription": dense})
        assert abs(len(Video.from_path(str(out)).frames) - 144) <= 1


class TestEncodeStageTransforms:
    """Transforms ordered after frame effects stream via the encoder's -vf."""

    def test_fade_then_crop_streams_and_matches_eager(self, tmp_path):
        ops = [{"op": "fade", "mode": "out", "duration": 1.0}, {"op": "crop", "width": 400, "height": 300}]
        eager = _plan(ops).run()
        video = _stream(_plan(ops), tmp_path)

        assert video.frames.shape[1:3] == (300, 400)
        assert abs(len(video.frames) - len(eager.frames)) <= 1
        assert video.frames[-1].mean() < 5, "fade-out lost in the encode-stage crop"
        active = 60
        mae = np.abs(video.frames[active].astype(np.float32) - eager.frames[active].astype(np.float32)).mean()
        assert mae < 15, f"encode-stage crop diverged from eager: {mae}"

    def test_fade_then_speed_streams_with_audio_sync(self, tmp_path, audio_source):
        ops = [{"op": "fade", "mode": "out", "duration": 1.0}, {"op": "speed_change", "speed": 2.0}]
        video = _stream(_plan(ops, source=audio_source), tmp_path)

        assert abs(len(video.frames) - 72) <= 1
        assert video.frames[-1].mean() < 8, "fade-out did not survive the encode-stage speed-up"
        assert video.audio is not None
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < 0.15

    def test_post_ops_behind_encode_stage_are_rejected(self, tmp_path):
        plan = VideoEdit.model_validate(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 2.0,
                        "end": 8.0,
                        "operations": [
                            {"op": "fade", "mode": "in", "duration": 0.5},
                            {"op": "crop", "width": 400, "height": 300},
                        ],
                    }
                ],
                "post_operations": [{"op": "color_adjust", "brightness": 0.2}],
            }
        )
        report = plan.streamability()
        post = report.entries[-1]
        assert post.streaming_class is StreamingClass.UNSTREAMABLE
        assert post.reason is not None and "encode-stage" in post.reason
        with pytest.raises(PlanValidationError, match="cannot stream"):
            plan.run_to_file(tmp_path / "out.mp4")


class TestMultiSegmentPostOps:
    """Post-op effects fold into per-segment schedules with global offsets."""

    @staticmethod
    def _plan(post: list[dict[str, Any]]) -> VideoEdit:
        return VideoEdit.model_validate(
            {
                "segments": [
                    {"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 3.0, "operations": []},
                    {"source": SMALL_VIDEO_PATH, "start": 6.0, "end": 9.0, "operations": []},
                ],
                "post_operations": post,
            }
        )

    def test_pixel_post_op_streams_across_segments(self, tmp_path):
        plan = self._plan([{"op": "vignette"}])
        assert plan.streamability().streamable

        video = _stream(plan, tmp_path)
        assert abs(len(video.frames) - 144) <= 2
        # The vignette darkens corners in BOTH segments.
        for idx in (36, 108):
            frame = video.frames[idx].astype(np.float32)
            assert frame[:30, :30].mean() < frame[235:265, 385:415].mean()

    def test_windowed_post_op_envelope_continues_across_boundary(self, tmp_path):
        """A fade-to-black VIDEO envelope spanning the concat boundary must
        not restart: brightness decreases monotonically into segment 2.

        Uses zoom, not fade (audio-coupled ops are excluded); a window-spanning
        ascending blur or zoom shows index continuity. Zoom with a window
        crossing the boundary: the zoom level at segment 2's first frames
        continues from segment 1's last frames.
        """
        plan = self._plan(
            [{"op": "zoom_effect", "mode": "in", "zoom_factor": 2.0, "window": {"start": 2.0, "stop": 4.0}}]
        )
        video = _stream(plan, tmp_path)

        # At the boundary (global frames 71/72, window-local 23/24 of 48) the
        # zoom is ~halfway. If the envelope restarted, segment 2's first
        # frame would be unzoomed (== source frame at 6.0s).
        seg2_first = video.frames[73].astype(np.float32)
        source2 = Video.from_path(SMALL_VIDEO_PATH, start_second=6.0, end_second=9.0)
        unzoomed_mae = np.abs(seg2_first - source2.frames[1].astype(np.float32)).mean()
        assert unzoomed_mae > 10, "zoom envelope restarted at the concat boundary"

    def test_audio_coupled_post_op_is_rejected(self, tmp_path):
        plan = self._plan([{"op": "fade", "mode": "out", "duration": 1.0}])
        report = plan.streamability()
        assert report.entries[-1].streaming_class is StreamingClass.UNSTREAMABLE
        assert "audio-coupled" in (report.entries[-1].reason or "")
        with pytest.raises(PlanValidationError, match="cannot stream"):
            plan.run_to_file(tmp_path / "out.mp4")

    def test_single_segment_fade_post_op_still_folds(self, tmp_path):
        plan = VideoEdit.model_validate(
            {
                "segments": [{"source": SMALL_VIDEO_PATH, "start": 2.0, "end": 8.0, "operations": []}],
                "post_operations": [{"op": "fade", "mode": "out", "duration": 1.0}],
            }
        )
        assert plan.streamability().streamable
        video = _stream(plan, tmp_path)
        assert video.frames[-1].mean() < 5

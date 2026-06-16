"""Tests for the per-segment audio filter graph (P1.9b).

Segment audio moved off the in-memory ``_load_segment_audio`` path into the
ffmpeg filter graph: the original source is a second ``-i`` input routed through
``-filter_complex`` in the SAME invocation as the video, compiled from each op's
``to_ffmpeg_audio_filter`` twin (``atempo``/silence-splice/``atrim``+``concat``
keep-windows/``volume`` envelope). These tests pin: per-op duration parity vs the old numpy
twins, the native ``anullsrc`` silent-track contract, the single-invocation
shape, and that ``run_to_file`` never materialises a full-length source
``Audio`` array.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.test_config import SMALL_VIDEO_PATH, TEST_AUDIO_PATH
from videopython.base.transcription import Transcription, TranscriptionWord
from videopython.base.video import Video
from videopython.editing import VideoEdit
from videopython.editing.operation import FilterCtx
from videopython.editing.streaming import (
    FrameEncoder,
    SegmentAudio,
    build_audio_filter_complex,
    source_has_audio_stream,
)

FPS = 24
SEGMENT = {"start": 2.0, "end": 8.0}  # 6 s cut -> 144 frames at 24 fps
TOL = 0.15  # the A/V duration tolerance the engine guarantees


@pytest.fixture(scope="module")
def audio_source(tmp_path_factory) -> str:
    """The small test video with a real audio track muxed in."""
    video = Video.from_path(SMALL_VIDEO_PATH).add_audio_from_file(TEST_AUDIO_PATH)
    out = tmp_path_factory.mktemp("p19b") / "with_audio.mp4"
    return str(video.save(out))


def _plan(operations: list[dict[str, Any]], source: str) -> VideoEdit:
    return VideoEdit.model_validate({"segments": [{"source": source, **SEGMENT, "operations": operations}]})


def _rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0


# --------------------------------------------------------------- per-op parity


class TestPerOpDurationParity:
    """The native af graph must produce the same output audio duration the old
    numpy twin did (which == the video duration), within the A/V tolerance."""

    def test_speedup_atempo_matches_video(self, render, audio_source):
        plan = _plan([{"op": "speed_change", "speed": 2.0}], audio_source)
        video = render(plan, name="speed.mp4")
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < TOL

    def test_slowdown_atempo_chain_matches_video(self, render, audio_source):
        # 0.5x exercises the atempo chaining boundary shared with time_stretch.
        plan = _plan([{"op": "speed_change", "speed": 0.5}], audio_source)
        video = render(plan, name="slow.mp4")
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < TOL

    def test_adjust_audio_false_leaves_audio_unstretched(self, render, audio_source):
        # adjust_audio=False compiles to no atempo; the length pin still trims
        # the (untouched) audio to the predicted output duration.
        plan = _plan([{"op": "speed_change", "speed": 2.0, "adjust_audio": False}], audio_source)
        video = render(plan, name="noadj.mp4")
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < TOL

    def test_freeze_silence_position_window(self, render, audio_source):
        plan = _plan([{"op": "freeze_frame", "timestamp": 1.0, "duration": 1.0}], audio_source)
        video = render(plan, name="freeze.mp4")
        sr = video.audio.metadata.sample_rate
        d = np.abs(video.audio.data)
        # The held window [1.0, 2.0) is the inserted silence; just before/after audible.
        inside = d[int(1.3 * sr) : int(1.7 * sr)].mean()
        before = d[int(0.2 * sr) : int(0.8 * sr)].mean()
        assert inside < before * 0.2, f"freeze window not silent: {inside} vs {before}"
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < TOL

    def test_silence_removal_keep_windows_match_video(self, render, audio_source):
        tr = Transcription(
            words=[
                TranscriptionWord(word="a", start=2.5, end=3.5),
                TranscriptionWord(word="b", start=3.5, end=4.5),
                TranscriptionWord(word="c", start=7.0, end=8.0),
            ]
        )
        plan = _plan([{"op": "silence_removal", "min_silence_duration": 1.0, "padding": 0.0}], audio_source)
        video = render(plan, name="sr.mp4", context={"transcription": tr})
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < TOL

    def test_fade_out_audio_decays(self, render, audio_source):
        plan = _plan([{"op": "fade", "mode": "out", "duration": 1.0}], audio_source)
        video = render(plan, name="fade.mp4")
        sr = video.audio.metadata.sample_rate
        d = video.audio.data
        head = _rms(d[: int(0.5 * sr)])
        tail = _rms(d[-int(0.3 * sr) :])
        assert tail < head * 0.5, f"fade-out did not decay: head={head}, tail={tail}"
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < TOL

    def test_windowed_fade_out_keeps_audio_after_window(self, render, audio_source):
        # 6s segment, fade-out windowed to [1.0, 3.0]. After the window the audio
        # must return to full volume -- the regression: native afade held gain 0
        # for the whole tail, silencing audio while the video kept playing.
        win = {"start": 1.0, "stop": 3.0}
        faded = render(
            _plan([{"op": "fade", "mode": "out", "duration": 0.5, "window": win}], audio_source),
            name="wf.mp4",
        )
        plain = render(_plan([], audio_source), name="plain.mp4")
        sr = faded.audio.metadata.sample_rate

        def _region(v: Video, lo: float, hi: float) -> np.ndarray:
            return v.audio.data[round(lo * sr) : round(hi * sr)]

        faded_after = _rms(_region(faded, 3.5, 5.5))
        plain_after = _rms(_region(plain, 3.5, 5.5))
        assert faded_after > 0.5 * plain_after, (
            f"audio after fade-out window was silenced: {faded_after} vs {plain_after}"
        )
        faded_ramp = _rms(_region(faded, 2.6, 3.0))
        plain_ramp = _rms(_region(plain, 2.6, 3.0))
        assert faded_ramp < 0.7 * plain_ramp, f"fade ramp did not attenuate: {faded_ramp} vs {plain_ramp}"

    def test_volume_adjust_window_mutes_in_sync(self, render, audio_source):
        plan = _plan([{"op": "volume_adjust", "volume": 0.0, "window": {"start": 2.0, "stop": 4.0}}], audio_source)
        video = render(plan, name="vol.mp4")
        sr = video.audio.metadata.sample_rate
        d = np.abs(video.audio.data)
        inside = d[int(2.3 * sr) : int(3.7 * sr)].mean()
        outside = d[int(0.2 * sr) : int(1.8 * sr)].mean()
        assert inside < outside * 0.05, f"volume window not muted: {inside} vs {outside}"
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < TOL


# ------------------------------------------------------- native silent track


class TestNoAudioSource:
    def test_source_without_audio_gets_native_silent_track(self, render):
        # SMALL_VIDEO_PATH has no audio stream -> anullsrc silence, not a Python
        # Audio.create_silent round-trip.
        assert (
            source_has_audio_stream(  # sanity: the with-audio path differs
                SMALL_VIDEO_PATH
            )
            is False
        )
        plan = _plan([], SMALL_VIDEO_PATH)
        video = render(plan, name="silent.mp4")
        # An AAC audio stream exists and is digitally silent.
        assert video.audio is not None
        assert video.audio.is_silent
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < TOL

    def test_anullsrc_input_built_for_silent_source(self):
        audio = SegmentAudio(
            source_path=SMALL_VIDEO_PATH,
            start_second=2.0,
            duration=6.0,
            af_filters=(),
            post_af_filters=(),
            has_audio_stream=False,
            output_seconds=6.0,
        )
        inputs, graph, out_label = build_audio_filter_complex(audio)
        assert any("anullsrc" in a for a in inputs)
        assert out_label == "[aout]"
        assert graph  # always carries at least the length pin + aresample


# ----------------------------------------------------- single ffmpeg command


class TestSingleInvocation:
    def test_frame_encoder_builds_one_filter_complex_command(self):
        audio = SegmentAudio(
            source_path=SMALL_VIDEO_PATH,
            start_second=2.0,
            duration=6.0,
            af_filters=("atempo=2.0",),
            post_af_filters=(),
            has_audio_stream=True,
            output_seconds=3.0,
        )
        enc = FrameEncoder(SMALL_VIDEO_PATH, width=64, height=64, fps=24, audio=audio)
        cmd = enc._build_command()
        # Exactly one ffmpeg process, one filter_complex, AAC audio, two -map.
        assert cmd[0] == "ffmpeg"
        assert cmd.count("-filter_complex") == 1
        assert cmd.count("ffmpeg") == 1
        assert "-c:a" in cmd and "aac" in cmd
        assert cmd.count("-map") == 2
        # The compiled op filter rides the single graph.
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "atempo=2.0" in graph
        # Two inputs: the rawvideo pipe and the source audio.
        assert cmd.count("-i") == 2

    def test_no_audio_means_an_and_no_filter_complex(self):
        enc = FrameEncoder(SMALL_VIDEO_PATH, width=64, height=64, fps=24, audio=None)
        cmd = enc._build_command()
        assert "-an" in cmd
        assert "-filter_complex" not in cmd
        assert cmd.count("-i") == 1


# ------------------------------------------------------- no eager audio decode


class TestNoFullSourceMaterialization:
    def test_run_to_file_never_decodes_full_source_audio(self, tmp_path, audio_source, monkeypatch):
        """The streaming file path must not decode the whole source audio into a
        numpy array (the old ``_load_segment_audio`` did). Spy on
        ``Audio.from_path`` and assert it is never called during ``run_to_file``
        -- segment audio now rides the ffmpeg filter graph instead."""
        import videopython.audio.audio as audio_mod

        calls: list[str] = []
        original = audio_mod.Audio.from_path

        def spy(cls, file_path):  # noqa: ANN001
            calls.append(str(file_path))
            return original(file_path)

        monkeypatch.setattr(audio_mod.Audio, "from_path", classmethod(spy))

        plan = _plan([{"op": "speed_change", "speed": 2.0}], audio_source)
        plan.run_to_file(tmp_path / "nomat.mp4")
        assert calls == [], f"run_to_file decoded source audio into memory: {calls}"


# ----------------------------------------------------------- builder coupling


class TestStagePlacementCoupling:
    def test_decode_stage_audio_filter_lands_in_af_filters(self, audio_source):
        # A leading speed_change is a decode-stage transform: its atempo must be
        # in af_filters (decode), not post_af_filters (encode).
        plan = _plan([{"op": "speed_change", "speed": 2.0}], audio_source)
        seg_plan = plan._compile_streaming_plans(None)[0]
        assert any("atempo" in f for f in seg_plan.af_filters)
        assert not seg_plan.post_af_filters

    def test_encode_stage_audio_filter_lands_in_post_af_filters(self, audio_source):
        # fade (frame effect, decode-stage audio) then speed (encode-stage):
        # the fade volume envelope in af_filters, atempo in post_af_filters --
        # coupled to the video vf/post_vf split.
        plan = _plan(
            [{"op": "fade", "mode": "out", "duration": 1.0}, {"op": "speed_change", "speed": 2.0}],
            audio_source,
        )
        seg_plan = plan._compile_streaming_plans(None)[0]
        assert any("volume=volume=" in f for f in seg_plan.af_filters)
        assert any("atempo" in f for f in seg_plan.post_af_filters)


# --------------------------------------------------------- compiled fragments


class TestCompiledFragments:
    def _ctx(self, frame_count: int = 144, fps: float = 24.0) -> FilterCtx:
        return FilterCtx(width=64, height=64, fps=fps, frame_count=frame_count, audio_label="f0")

    def test_freeze_after_splices_silence(self):
        from videopython.editing.transforms import FreezeFrame

        frag = FreezeFrame(timestamp=1.0, duration=1.0, position="after").to_ffmpeg_audio_filter(self._ctx())
        assert frag is not None
        assert "asplit=3" in frag and "concat=n=3:v=0:a=1" in frag and "volume=0" in frag

    def test_silence_removal_concats_keep_windows(self):
        from videopython.editing.transforms import SilenceRemoval

        tr = Transcription(
            words=[
                TranscriptionWord(word="a", start=2.5, end=3.5),
                TranscriptionWord(word="c", start=7.0, end=8.0),
            ]
        )
        ctx = FilterCtx(width=64, height=64, fps=24.0, frame_count=144, context={"transcription": tr}, audio_label="f0")
        frag = SilenceRemoval(min_silence_duration=1.0, padding=0.0).to_ffmpeg_audio_filter(ctx)
        assert frag is not None
        assert "asplit=" in frag and "concat=n=" in frag and "atrim=" in frag

    def test_speed_change_atempo_chain_for_extreme_speed(self):
        from videopython.editing.transforms import SpeedChange

        # 4x decomposes to two chained atempo=2.0 stages.
        frag = SpeedChange(speed=4.0).to_ffmpeg_audio_filter(self._ctx())
        assert frag == "atempo=2.0,atempo=2.0"

    def test_fade_compiles_to_windowed_volume_envelope(self):
        from videopython.editing.effects import Fade

        # A fade is a windowed volume envelope, NOT afade -- afade holds gain 0
        # outside the ramp, which would mute audio outside a windowed fade.
        frag = Fade(mode="in", duration=0.5, curve="sqrt").to_ffmpeg_audio_filter(self._ctx())
        assert frag is not None
        assert frag.startswith("volume=volume=") and "eval=frame" in frag
        assert "sqrt(" in frag  # sqrt curve preserved
        assert "afade" not in frag

    def test_windowed_fade_out_keeps_full_gain_after_window(self):
        from videopython.editing.effects import Fade

        # 6s segment, fade-out over [1.0, 3.0]. Gain must return to 1 after the
        # window (the regression: afade stayed at 0 -> silent tail).
        win_ctx = self._ctx()
        spec = {"start": 1.0, "stop": 3.0}
        frag = Fade(mode="out", duration=0.5, window=spec).to_ffmpeg_audio_filter(win_ctx)
        assert frag is not None
        # The fade ramp lives in [stop-ramp, stop] = [2.5, 3.0]; everything else
        # is the literal fallback gain of 1.
        assert "between(t,2.500000,3.000000)" in frag
        assert frag.endswith(",1)':eval=frame")  # gain falls back to 1 outside the ramp

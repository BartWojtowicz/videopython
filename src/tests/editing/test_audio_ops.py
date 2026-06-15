"""Tests for plan-level audio: the music bed and its transcription-derived duck (P1.11).

The music bed is NOT a per-segment op: it is a frozen ``MusicBed`` field on
``VideoEdit`` mixed under the WHOLE assembled program in a final ffmpeg ``amix``
pass after concat / transitions. ``run`` and ``run_to_file`` share one
``build_music_bed_filter_complex`` builder so they cannot diverge. Ducking is
deterministic and transcription-derived (a ``volume=...:eval=frame`` automation
over the shared ``speech_windows`` helper), single-segment only.

These tests pin: the model/schema surface, ``SOURCE_UNREADABLE`` source
validation, the exact ``amix`` and duck automation filter strings, the
multi-segment+duck rejection, ``speech_windows`` parity vs ``SilenceRemoval``,
the already-compiled ``VolumeAdjust`` window gain, and a tiny e2e (one short clip
+ short bed wav) where the bed is quieter under a ducked speech window and
audible flat otherwise. Kept to one tiny clip so the editing suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tests.test_config import SMALL_VIDEO_METADATA, SMALL_VIDEO_PATH
from videopython.audio import Audio
from videopython.audio.audio import AudioMetadata
from videopython.base.exceptions import PlanErrorCode, PlanValidationError
from videopython.base.transcription import Transcription, TranscriptionWord
from videopython.base.video import Video
from videopython.editing import VideoEdit
from videopython.editing.audio_ops import MusicBed, build_music_bed_filter_complex, duck_volume_expression
from videopython.editing.transforms import SilenceRemoval, speech_windows


def _rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0


@pytest.fixture(scope="module")
def bed_wav(tmp_path_factory) -> str:
    """A 4 s stereo noise bed at a constant level, so RMS over any window is well defined."""
    rng = np.random.default_rng(0)
    sr = 44100
    data = (rng.standard_normal((sr * 4, 2)).astype(np.float32)) * 0.3
    audio = Audio(
        data,
        AudioMetadata(sample_rate=sr, channels=2, sample_width=2, duration_seconds=4.0, frame_count=sr * 4),
    )
    path = tmp_path_factory.mktemp("p111") / "bed.wav"
    audio.save(path, format="wav")
    return str(path)


@pytest.fixture(scope="module")
def voiced_source(tmp_path_factory) -> str:
    """A 6 s clip carrying a constant-level tone, so the program audio level is well defined."""
    sr = 44100
    seconds = 6.0
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    wave = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    data = np.stack([wave, wave], axis=1)
    meta = AudioMetadata(sample_rate=sr, channels=2, sample_width=2, duration_seconds=seconds, frame_count=len(data))
    video = Video.from_image(np.zeros((64, 64, 3), dtype=np.uint8), fps=24, length_seconds=seconds)
    video.audio = Audio(data, meta)
    path = tmp_path_factory.mktemp("p111voiced") / "voiced.mp4"
    return str(video.save(path))


# ----------------------------------------------------------------- model / schema


class TestMusicBedModel:
    def test_defaults(self):
        bed = MusicBed(source="x.wav")
        assert bed.gain == 0.25
        assert bed.loop is True
        assert bed.fade_in == 0.0 and bed.fade_out == 0.0
        assert bed.duck is None
        assert bed.duck_attack == 0.2 and bed.duck_release == 0.5

    def test_frozen_and_closed(self):
        assert MusicBed.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            MusicBed(source="x.wav", bogus=1)

    def test_field_bounds(self):
        with pytest.raises(ValidationError):
            MusicBed(source="x.wav", gain=-0.1)
        with pytest.raises(ValidationError):
            MusicBed(source="x.wav", duck=1.5)
        with pytest.raises(ValidationError):
            MusicBed(source="x.wav", duck_attack=0.0)  # gt=0
        with pytest.raises(ValidationError):
            MusicBed(source="x.wav", duck_release=0.0)  # gt=0

    def test_music_bed_surfaces_in_video_edit_schema(self):
        schema = VideoEdit.json_schema()
        assert "music_bed" in schema["properties"]
        prop = schema["properties"]["music_bed"]
        # Inlined self-contained closed object, anyOf with null (optional).
        bed_obj = prop["anyOf"][0]
        assert bed_obj["additionalProperties"] is False
        assert {"type": "null"} in prop["anyOf"]
        assert set(bed_obj["properties"]) == {
            "source",
            "gain",
            "loop",
            "fade_in",
            "fade_out",
            "duck",
            "duck_attack",
            "duck_release",
        }

    def test_music_bed_in_strict_schema(self):
        strict = VideoEdit.json_schema(strict=True)
        assert "music_bed" in strict["required"]

    def test_plan_round_trips_music_bed(self):
        plan = VideoEdit.model_validate(
            {
                "segments": [{"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 2.0}],
                "music_bed": {"source": "bed.wav", "gain": 0.4, "duck": 0.6},
            }
        )
        assert isinstance(plan.music_bed, MusicBed)
        assert plan.to_dict()["music_bed"]["gain"] == 0.4


# --------------------------------------------------------------- source validation


class TestSourceValidation:
    def test_unreadable_source_rejected_with_source_unreadable(self):
        plan = VideoEdit.model_validate(
            {
                "segments": [{"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 2.0}],
                "music_bed": {"source": "/nope/missing.wav"},
            }
        )
        errors = plan.check(SMALL_VIDEO_METADATA)
        unreadable = [e for e in errors if e.code == PlanErrorCode.SOURCE_UNREADABLE]
        assert unreadable, "missing bed source did not surface SOURCE_UNREADABLE"
        assert unreadable[0].location == "music_bed"
        assert unreadable[0].field == "source"

    def test_unreadable_source_raises_in_validate(self):
        plan = VideoEdit.model_validate(
            {
                "segments": [{"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 2.0}],
                "music_bed": {"source": "/nope/missing.wav"},
            }
        )
        with pytest.raises(PlanValidationError) as exc:
            plan.validate_with_metadata(SMALL_VIDEO_METADATA)
        assert exc.value.errors[0].code == PlanErrorCode.SOURCE_UNREADABLE

    def test_readable_bed_passes_validation(self, bed_wav):
        plan = VideoEdit.model_validate(
            {
                "segments": [{"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 2.0}],
                "music_bed": {"source": bed_wav},
            }
        )
        errors = [e for e in plan.check(SMALL_VIDEO_METADATA) if e.code == PlanErrorCode.SOURCE_UNREADABLE]
        assert errors == []


# ------------------------------------------------------------- filter-string units


class TestBedMixFilterStrings:
    def test_amix_string_and_loop_input(self):
        bed = MusicBed(source="bed.wav", gain=0.25, loop=True)
        inputs, graph, out_label = build_music_bed_filter_complex(bed, 6.0)
        assert out_label == "[mixout]"
        # Loop the bed on the input side; the program is input 0, bed input 1.
        assert inputs[:2] == ["-stream_loop", "-1"]
        assert inputs[-2:] == ["-i", "bed.wav"]
        # The exact amix clause: duration=first clamps to the program length;
        # normalize=0 keeps the program at full level (default normalize=1 would
        # halve the dialogue when a bed is attached).
        assert any("amix=inputs=2:duration=first:dropout_transition=0:normalize=0" in s for s in graph)
        # The bed is gained, then pinned to the program length (atrim+apad).
        assert any("volume=0.250000" in s for s in graph)
        assert any("atrim=end=6.000000" in s for s in graph)
        assert any("apad=whole_dur=6.000000" in s for s in graph)

    def test_no_loop_omits_stream_loop(self):
        bed = MusicBed(source="bed.wav", loop=False)
        inputs, _graph, _ = build_music_bed_filter_complex(bed, 6.0)
        assert "-stream_loop" not in inputs

    def test_fade_in_out_stages(self):
        bed = MusicBed(source="bed.wav", fade_in=1.0, fade_out=2.0)
        _inputs, graph, _ = build_music_bed_filter_complex(bed, 6.0)
        joined = ";".join(graph)
        assert "afade=t=in:st=0:d=1.000000" in joined
        # fade-out starts at program_seconds - fade_out.
        assert "afade=t=out:st=4.000000:d=2.000000" in joined

    def test_duck_automation_string_exact(self):
        expr = duck_volume_expression([(1.0, 2.0)], duck=0.8, attack=0.2, release=0.5)
        assert expr.startswith("volume=volume='") and expr.endswith(":eval=frame")
        # Attack ramp over [1.0, 1.2], hold at floor over [1.2, 2.0], release over [2.0, 2.5].
        assert "between(t,1.000000,1.200000)" in expr
        assert "between(t,1.200000,2.000000)" in expr
        assert "between(t,2.000000,2.500000)" in expr
        # Holds the ducked floor (1 - duck) and falls back to 1 outside speech.
        assert "0.200000" in expr  # floor == 1 - 0.8
        # One speech window -> three nested if()s, all closing on the literal 1.
        assert expr.endswith(",1)))':eval=frame")  # outermost fallback gain is 1

    def test_empty_speech_is_constant_one(self):
        assert duck_volume_expression([], duck=0.8, attack=0.2, release=0.5) == "volume=volume='1':eval=frame"

    def test_duck_stage_present_in_bed_graph_when_speech_given(self):
        bed = MusicBed(source="bed.wav", duck=0.9)
        _inputs, graph, _ = build_music_bed_filter_complex(bed, 6.0, speech=[(1.0, 2.0)])
        assert any("eval=frame" in s and "between(t,1.000000" in s for s in graph)

    def test_no_duck_stage_when_duck_unset(self):
        bed = MusicBed(source="bed.wav")  # duck None
        _inputs, graph, _ = build_music_bed_filter_complex(bed, 6.0, speech=[(1.0, 2.0)])
        assert not any("eval=frame" in s for s in graph)


# ----------------------------------------------------------- speech_windows parity


class TestSpeechWindowsHelper:
    def test_silence_removal_silences_are_the_gaps_between_speech_windows(self):
        # SilenceRemoval derives its silences FROM speech_windows: a silence is
        # exactly a gap >= min_silence_duration between consecutive padded speech
        # windows (and before the first / after the last). Pin that the op reuses
        # the shared helper rather than re-deriving its own windows.
        words = [
            TranscriptionWord(word="a", start=0.5, end=1.0),
            TranscriptionWord(word="b", start=1.1, end=1.5),
            TranscriptionWord(word="c", start=5.0, end=5.5),
        ]
        total = 8.0
        padding, min_silence = 0.2, 1.0
        op = SilenceRemoval(min_silence_duration=min_silence, padding=padding)
        speech = speech_windows(words, padding, total)

        expected_silences: list[tuple[float, float]] = []
        prev_end = 0.0
        for s_start, s_end in speech:
            if s_start - prev_end >= min_silence:
                expected_silences.append((prev_end, s_start))
            prev_end = s_end
        if total - prev_end >= min_silence:
            expected_silences.append((prev_end, total))

        assert op._silence_ranges(words, total) == expected_silences
        # The padded speech windows themselves (the duck consumes these directly).
        assert speech == [(0.3, 1.7), (4.8, 5.7)]

    def test_speech_windows_merges_overlaps_and_clamps(self):
        words = [
            TranscriptionWord(word="a", start=0.0, end=1.0),
            TranscriptionWord(word="b", start=1.05, end=2.0),  # padded windows abut/overlap -> merge
        ]
        windows = speech_windows(words, 0.1, total_seconds=5.0)
        assert windows == [(0.0, 2.1)]  # clamped low to 0, merged into one


# ---------------------------------------------------------- multi-segment + duck


class TestMultiSegmentDuck:
    def _two_segment_plan(self, music_bed: dict[str, object]) -> VideoEdit:
        seg = {"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 3.0}
        return VideoEdit.model_validate({"segments": [seg, dict(seg)], "music_bed": music_bed})

    def test_multi_segment_with_duck_raises_structured_error(self):
        plan = self._two_segment_plan({"source": SMALL_VIDEO_PATH, "duck": 0.5})
        codes = [e.code for e in plan.check(SMALL_VIDEO_METADATA)]
        assert PlanErrorCode.MUSIC_BED_DUCK_MULTISEGMENT in codes

    def test_multi_segment_duck_run_guard_raises(self):
        plan = self._two_segment_plan({"source": SMALL_VIDEO_PATH, "duck": 0.5})
        with pytest.raises(PlanValidationError) as exc:
            plan._assert_music_bed_supported()
        assert exc.value.errors[0].code == PlanErrorCode.MUSIC_BED_DUCK_MULTISEGMENT
        assert exc.value.errors[0].location == "music_bed"

    def test_multi_segment_non_ducked_bed_is_allowed(self, bed_wav):
        plan = self._two_segment_plan({"source": bed_wav})
        codes = [e.code for e in plan.check(SMALL_VIDEO_METADATA)]
        assert PlanErrorCode.MUSIC_BED_DUCK_MULTISEGMENT not in codes


# --------------------------------------------------- VolumeAdjust gain (P1.9b) check


class TestVolumeAdjustWindowGainCompiles:
    """Gain is already done in P1.9b -- confirm VolumeAdjust compiles to the audio graph."""

    def _ctx(self, frame_count: int = 144, fps: float = 24.0):
        from videopython.editing.operation import FilterCtx

        return FilterCtx(width=64, height=64, fps=fps, frame_count=frame_count, audio_label="f0")

    def test_flat_window_gain_compiles_to_enable_between(self):
        from videopython.editing.effects import VolumeAdjust
        from videopython.editing.operation import TimeRange

        frag = VolumeAdjust(volume=0.5, window=TimeRange(start=1.0, stop=3.0)).to_ffmpeg_audio_filter(self._ctx())
        assert frag == "volume=0.500000:enable='between(t,1.000000,3.000000)'"

    def test_ramped_gain_compiles_to_eval_frame_expression(self):
        from videopython.editing.effects import VolumeAdjust
        from videopython.editing.operation import TimeRange

        frag = VolumeAdjust(
            volume=0.0, ramp_duration=0.5, window=TimeRange(start=1.0, stop=3.0)
        ).to_ffmpeg_audio_filter(self._ctx())
        assert frag is not None and frag.startswith("volume=volume=") and "eval=frame" in frag

    def test_noop_gain_compiles_to_none(self):
        from videopython.editing.effects import VolumeAdjust

        assert VolumeAdjust(volume=1.0).to_ffmpeg_audio_filter(self._ctx()) is None


# ----------------------------------------------------------------------- e2e


class TestMusicBedEndToEnd:
    SEG = {"source": SMALL_VIDEO_PATH, "start": 2.0, "end": 6.0}  # 4 s, SMALL_VIDEO has no audio -> silent program

    def _plan(self, music_bed: dict[str, object]) -> VideoEdit:
        return VideoEdit.model_validate({"segments": [{**self.SEG}], "music_bed": music_bed})

    def test_flat_bed_is_audible_under_silent_program(self, tmp_path, bed_wav):
        # The program is silent (no source audio), so the output RMS is the bed.
        plan = self._plan({"source": bed_wav, "gain": 0.8})
        out = Video.from_path(str(plan.run_to_file(tmp_path / "flat.mp4")))
        assert _rms(out.audio.data) > 0.05, "flat bed not audible in the mix"
        # The bed loops/trims to exactly the program timeline.
        assert abs(out.audio.metadata.duration_seconds - out.total_seconds) < 0.15

    def test_bed_preserves_program_audio_level(self, tmp_path, bed_wav, voiced_source):
        # Attaching a bed must NOT attenuate the program audio. amix's default
        # normalize=1 divides every input by 2, halving the dialogue; normalize=0
        # passes the program through at full level. A silent (gain=0) bed must
        # therefore leave the program RMS unchanged (ratio ~1.0, not ~0.5).
        seg = {"source": voiced_source, "start": 0.0, "end": 4.0}
        plain = VideoEdit.model_validate({"segments": [seg]}).run_to_file(tmp_path / "plain.mp4")
        plain_rms = _rms(Video.from_path(str(plain)).audio.data)
        with_bed = VideoEdit.model_validate(
            {"segments": [seg], "music_bed": {"source": bed_wav, "gain": 0.0}}
        ).run_to_file(tmp_path / "withbed.mp4")
        bed_rms = _rms(Video.from_path(str(with_bed)).audio.data)
        assert 0.8 < bed_rms / plain_rms < 1.25, f"bed attenuated the program: {bed_rms} vs {plain_rms}"

    def test_duck_lowers_bed_under_speech_window(self, tmp_path, bed_wav):
        # Speech at absolute [3, 4) -> segment-local [1, 2); the duck (with the
        # 0.15 padding) lowers the bed there and restores it away from speech.
        tr = Transcription(words=[TranscriptionWord(word="hello", start=3.0, end=4.0)])
        ducked = Video.from_path(
            str(
                self._plan({"source": bed_wav, "gain": 0.8, "duck": 0.9}).run_to_file(
                    tmp_path / "ducked.mp4", context={"transcription": tr}
                )
            )
        )
        flat = Video.from_path(str(self._plan({"source": bed_wav, "gain": 0.8}).run_to_file(tmp_path / "flat2.mp4")))
        sr = ducked.audio.metadata.sample_rate

        def region(v: Video, lo: float, hi: float) -> np.ndarray:
            return v.audio.data[round(lo * sr) : round(hi * sr)]

        ducked_in = _rms(region(ducked, 1.3, 1.8))  # inside the ducked speech window
        ducked_away = _rms(region(ducked, 3.0, 3.7))  # well after the window
        flat_in = _rms(region(flat, 1.3, 1.8))
        assert ducked_in < ducked_away * 0.5, f"duck did not lower bed under speech: {ducked_in} vs {ducked_away}"
        assert flat_in > ducked_in * 1.5, (
            f"flat bed should be louder than ducked under speech: {flat_in} vs {ducked_in}"
        )

    def test_run_and_run_to_file_bed_mix_agree(self, tmp_path, bed_wav):
        tr = Transcription(words=[TranscriptionWord(word="hello", start=3.0, end=4.0)])
        plan = self._plan({"source": bed_wav, "gain": 0.8, "duck": 0.9})
        in_memory = plan.run(context={"transcription": tr})
        on_disk = Video.from_path(str(plan.run_to_file(tmp_path / "eq.mp4", context={"transcription": tr})))
        sr = in_memory.audio.metadata.sample_rate

        def region(v: Video, lo: float, hi: float) -> np.ndarray:
            return v.audio.data[round(lo * sr) : round(hi * sr)]

        # Both paths duck the bed under the same window (shared builder).
        mem_in, mem_away = _rms(region(in_memory, 1.3, 1.8)), _rms(region(in_memory, 3.0, 3.7))
        disk_in, disk_away = _rms(region(on_disk, 1.3, 1.8)), _rms(region(on_disk, 3.0, 3.7))
        assert mem_in < mem_away * 0.5
        assert disk_in < disk_away * 0.5
        assert abs(in_memory.audio.metadata.duration_seconds - on_disk.audio.metadata.duration_seconds) < 0.15

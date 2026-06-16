"""Tests for the editing transforms (streaming-only, post eager-removal).

Since 0.44.0 there is no eager/in-memory ``apply`` path: a transform exists
only as a streaming compilation. So these tests assert the two decode-free
surfaces a transform exposes:

* ``predict_metadata(meta)`` -- exact output shape / fps / frame count, the
  fail-fast gate run during plan validation.
* ``to_ffmpeg_filter(FilterCtx(...))`` / ``to_ffmpeg_audio_filter(...)`` -- the
  exact ffmpeg filter expression the streaming engine appends to the graph.

End-to-end frame *content* (the time-warp curve, frozen-frame holds, the
silence cut, audio sync) is covered against real decoded output in
``test_native_transform_streaming.py``; it is not duplicated here. Anything
that used to assert cv2-exact pixels or cut-frame identity cannot survive the
move to ffmpeg (libswscale != cv2; a cut is a decode boundary), so those
asserts are replaced by filter-string + ``predict_metadata`` checks.
"""

import pytest
from pydantic import ValidationError

from videopython.base.exceptions import PlanErrorCode, PlanValidationError
from videopython.base.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import VideoMetadata
from videopython.editing.operation import FilterCtx
from videopython.editing.transforms import (
    Crop,
    CropMode,
    CutFrames,
    CutSeconds,
    FreezeFrame,
    ResampleFPS,
    Resize,
    SilenceRemoval,
    SpeedChange,
)

# A reusable source-metadata stand-in: 800x500 @ 24fps, 12 s (== the small test
# video). predict_metadata is decode-free, so a metadata object is all it needs.
SMALL_META = VideoMetadata(height=500, width=800, fps=24, frame_count=288, total_seconds=12.0)


def _ctx(meta: VideoMetadata, **kwargs) -> FilterCtx:
    """A FilterCtx mirroring a VideoMetadata, with the folded frame_count set."""
    return FilterCtx(
        width=meta.width,
        height=meta.height,
        fps=meta.fps,
        frame_count=meta.frame_count,
        **kwargs,
    )


@pytest.mark.parametrize("start, end", [(0, 100), (100, 101), (100, 120)])
def test_cut_frames_predicts_frame_count(start, end):
    """CutFrames(predict) yields exactly ``end - start`` frames."""
    result = CutFrames(start=start, end=end).predict_metadata(SMALL_META)
    assert result.frame_count == end - start
    assert result.total_seconds == round((end - start) / SMALL_META.fps, 4)


@pytest.mark.parametrize("start, end", [(0, 0.5), (0, 1), (0.5, 1.5)])
def test_cut_seconds_predicts_duration(start, end):
    """CutSeconds(predict) yields the frame-rounded duration of the window."""
    result = CutSeconds(start=start, end=end).predict_metadata(SMALL_META)
    start_f = round(start * SMALL_META.fps)
    end_f = round(end * SMALL_META.fps)
    assert result.total_seconds == round((end_f - start_f) / SMALL_META.fps, 4)


@pytest.mark.parametrize(
    "height,width",
    [
        (40, 60),
        (500, 700),
    ],
)
def test_resize_predicts_dims_and_compiles_scale(height, width):
    """Resize predicts the exact target dims and compiles to ``scale=W:H``."""
    resize = Resize(height=height, width=width)
    predicted = resize.predict_metadata(SMALL_META)
    assert (predicted.height, predicted.width) == (height, width)
    assert resize.to_ffmpeg_filter(_ctx(SMALL_META)) == f"scale={width}:{height}"


def test_resize_round_to_even_preserves_aspect_approximately():
    """Single-dimension resize keeps aspect, snapping the other to even."""
    meta = VideoMetadata(height=540, width=302, fps=30, frame_count=30, total_seconds=1.0)
    resize = Resize(width=1080)
    predicted = resize.predict_metadata(meta)
    assert (predicted.height, predicted.width) == (1932, 1080)
    assert resize.to_ffmpeg_filter(_ctx(meta)) == "scale=1080:1932"


def test_resample_fps_upsample_frame_count():
    meta = VideoMetadata(height=64, width=64, fps=10, frame_count=10, total_seconds=1.0)
    resample = ResampleFPS(fps=20)
    predicted = resample.predict_metadata(meta)
    assert predicted.fps == 20
    assert predicted.frame_count == 20
    assert resample.to_ffmpeg_filter(_ctx(meta)) == "fps=20.0"


def test_resample_fps_downsample_frame_count():
    meta = VideoMetadata(height=64, width=64, fps=20, frame_count=20, total_seconds=1.0)
    resample = ResampleFPS(fps=10)
    predicted = resample.predict_metadata(meta)
    assert predicted.fps == 10
    assert predicted.frame_count == 10
    assert resample.to_ffmpeg_filter(_ctx(meta)) == "fps=10.0"


class TestCrop:
    """Crop predicts the cropped dims and compiles to ``crop=W:H:X:Y``."""

    @pytest.fixture
    def meta(self):
        return VideoMetadata(height=500, width=800, fps=30, frame_count=30, total_seconds=1.0)

    def test_crop_center_pixels(self, meta):
        transform = Crop(width=100, height=80, mode=CropMode.CENTER)
        predicted = transform.predict_metadata(meta)
        assert (predicted.height, predicted.width) == (80, 100)
        # Center box: (800-100)//2 = 350, (500-80)//2 = 210.
        assert transform.to_ffmpeg_filter(_ctx(meta)) == "crop=100:80:350:210"

    def test_crop_center_normalized(self, meta):
        transform = Crop(width=0.5, height=0.5, mode=CropMode.CENTER)
        predicted = transform.predict_metadata(meta)
        assert (predicted.height, predicted.width) == (250, 400)
        assert transform.to_ffmpeg_filter(_ctx(meta)) == "crop=400:250:200:125"

    def test_crop_custom_position_pixels(self, meta):
        transform = Crop(width=50, height=40, x=10, y=20, mode=CropMode.CUSTOM)
        predicted = transform.predict_metadata(meta)
        assert (predicted.height, predicted.width) == (40, 50)
        assert transform.to_ffmpeg_filter(_ctx(meta)) == "crop=50:40:10:20"

    def test_crop_custom_position_normalized(self, meta):
        # Right half: x=0.5, width=0.5, full height.
        transform = Crop(width=0.5, height=1.0, x=0.5, y=0.0, mode=CropMode.CUSTOM)
        predicted = transform.predict_metadata(meta)
        assert (predicted.height, predicted.width) == (500, 400)
        assert transform.to_ffmpeg_filter(_ctx(meta)) == "crop=400:500:400:0"

    def test_crop_mixed_values(self, meta):
        # Width in pixels, height normalized.
        transform = Crop(width=100, height=0.5, mode=CropMode.CENTER)
        predicted = transform.predict_metadata(meta)
        assert predicted.width == 100
        assert predicted.height == 250

    def test_crop_preserves_frame_count(self, meta):
        transform = Crop(width=0.5, height=0.5, mode=CropMode.CENTER)
        predicted = transform.predict_metadata(meta)
        assert predicted.frame_count == meta.frame_count

    def test_crop_exceeds_source_raises(self, meta):
        with pytest.raises(PlanValidationError) as exc:
            Crop(width=2000, height=80, mode=CropMode.CENTER).predict_metadata(meta)
        assert exc.value.errors[0].code is PlanErrorCode.CROP_EXCEEDS_SOURCE


class TestSpeedChange:
    """SpeedChange predicts the new frame count and compiles setpts/atempo."""

    def test_speed_up_2x_halves_frame_count(self):
        predicted = SpeedChange(speed=2.0).predict_metadata(SMALL_META)
        assert predicted.frame_count == SMALL_META.frame_count // 2

    def test_slow_down_half_doubles_frame_count(self):
        predicted = SpeedChange(speed=0.5).predict_metadata(SMALL_META)
        assert predicted.frame_count == SMALL_META.frame_count * 2

    def test_speed_1x_no_change(self):
        predicted = SpeedChange(speed=1.0).predict_metadata(SMALL_META)
        assert predicted.frame_count == SMALL_META.frame_count

    def test_speed_ramp_uses_average(self):
        # Ramp 1x -> 2x averages 1.5x.
        predicted = SpeedChange(speed=1.0, end_speed=2.0).predict_metadata(SMALL_META)
        expected = int(SMALL_META.frame_count / 1.5)
        assert predicted.frame_count == expected

    def test_invalid_speed_raises(self):
        with pytest.raises(ValueError):
            SpeedChange(speed=0)
        with pytest.raises(ValueError):
            SpeedChange(speed=-1.0)
        with pytest.raises(ValueError):
            SpeedChange(speed=1.0, end_speed=0)

    def test_preserves_frame_shape(self):
        predicted = SpeedChange(speed=2.0).predict_metadata(SMALL_META)
        assert (predicted.height, predicted.width) == (SMALL_META.height, SMALL_META.width)

    def test_zero_frame_speed_raises(self):
        with pytest.raises(PlanValidationError) as exc:
            SpeedChange(speed=1000.0).predict_metadata(SMALL_META)
        assert exc.value.errors[0].code is PlanErrorCode.DEGENERATE_DURATION

    def test_constant_speedup_compiles_setpts_and_fps(self):
        chain = SpeedChange(speed=2.0).to_ffmpeg_filter(_ctx(SMALL_META))
        assert chain is not None
        retime, resample = chain.split(",")
        assert retime.startswith("setpts=(PTS-STARTPTS)/2")
        assert resample == "fps=24"

    def test_slowdown_with_interpolation_uses_framerate(self):
        # interpolate=True (default) on a slowdown blends via the framerate filter.
        chain = SpeedChange(speed=0.5).to_ffmpeg_filter(_ctx(SMALL_META))
        assert chain is not None
        assert chain.endswith("framerate=fps=24")

    def test_slowdown_no_interpolation_uses_fps(self):
        chain = SpeedChange(speed=0.5, interpolate=False).to_ffmpeg_filter(_ctx(SMALL_META))
        assert chain is not None
        assert chain.endswith("fps=24")
        assert "framerate" not in chain

    def test_ramp_needs_frame_count(self):
        # Unknown frame count -> ramp cannot compile -> not streamable here.
        ctx = FilterCtx(width=800, height=500, fps=24, frame_count=0)
        assert SpeedChange(speed=1.0, end_speed=2.0).to_ffmpeg_filter(ctx) is None


class TestSpeedChangeAudio:
    """SpeedChange's audio twin time-stretches via an atempo chain."""

    def test_speed_up_2x_audio_atempo(self):
        chain = SpeedChange(speed=2.0).to_ffmpeg_audio_filter(_ctx(SMALL_META))
        assert chain == "atempo=2.0"

    def test_slow_down_half_audio_atempo(self):
        chain = SpeedChange(speed=0.5).to_ffmpeg_audio_filter(_ctx(SMALL_META))
        assert chain == "atempo=0.5"

    def test_audio_adjust_false_is_noop(self):
        assert SpeedChange(speed=2.0, adjust_audio=False).to_ffmpeg_audio_filter(_ctx(SMALL_META)) is None

    def test_speed_1x_audio_is_noop(self):
        # An identity stretch yields an empty atempo chain -> None.
        assert SpeedChange(speed=1.0).to_ffmpeg_audio_filter(_ctx(SMALL_META)) is None

    def test_ramp_audio_uses_average_speed(self):
        # Ramp 1x -> 3x averages 2x, compiled as a single constant stretch.
        chain = SpeedChange(speed=1.0, end_speed=3.0).to_ffmpeg_audio_filter(_ctx(SMALL_META))
        assert chain == "atempo=2.0"


@pytest.fixture
def video_meta_1s():
    """1-second @ 30fps source metadata (30 frames)."""
    return VideoMetadata(height=64, width=64, fps=30, frame_count=30, total_seconds=1.0)


def _make_transcription(words_data: list[tuple[float, float, str]]) -> Transcription:
    """Helper to create a Transcription from (start, end, word) tuples."""
    words = [TranscriptionWord(start=s, end=e, word=w) for s, e, w in words_data]
    segment = TranscriptionSegment(
        start=words[0].start, end=words[-1].end, text=" ".join(w.word for w in words), words=words
    )
    return Transcription(segments=[segment])


class TestFreezeFrame:
    """FreezeFrame predicts the extended/replaced frame count.

    Frozen-frame *content* is asserted end-to-end in
    ``test_native_transform_streaming.py::TestFreezeFrameStreaming``.
    """

    def test_freeze_after_increases_duration(self, video_meta_1s):
        predicted = FreezeFrame(timestamp=0.5, duration=1.0, position="after").predict_metadata(video_meta_1s)
        assert predicted.frame_count == video_meta_1s.frame_count + round(1.0 * video_meta_1s.fps)

    def test_freeze_before_increases_duration(self, video_meta_1s):
        predicted = FreezeFrame(timestamp=0.5, duration=1.0, position="before").predict_metadata(video_meta_1s)
        assert predicted.frame_count == video_meta_1s.frame_count + round(1.0 * video_meta_1s.fps)

    def test_freeze_replace_maintains_approx_duration(self, video_meta_1s):
        predicted = FreezeFrame(timestamp=0.0, duration=0.5, position="replace").predict_metadata(video_meta_1s)
        assert abs(predicted.frame_count - video_meta_1s.frame_count) <= 1

    def test_replace_clamps_to_end(self, video_meta_1s):
        # A replace window running past the clip end stays valid (clamped).
        predicted = FreezeFrame(timestamp=0.9, duration=5.0, position="replace").predict_metadata(video_meta_1s)
        assert predicted.frame_count > 0

    def test_freeze_after_compiles_loop_chain(self, video_meta_1s):
        chain = FreezeFrame(timestamp=0.5, duration=0.5, position="after").to_ffmpeg_filter(_ctx(video_meta_1s))
        assert chain is not None
        # Held frame is index round(0.5*30)=15, held for round(0.5*30)=15 frames.
        assert chain.startswith("loop=loop=15:size=1:start=15")
        assert chain.endswith("fps=30")

    def test_freeze_needs_frame_count(self):
        ctx = FilterCtx(width=64, height=64, fps=30, frame_count=0)
        assert FreezeFrame(timestamp=0.5, duration=0.5).to_ffmpeg_filter(ctx) is None

    def test_timestamp_out_of_range_raises_predict(self, video_meta_1s):
        with pytest.raises(PlanValidationError) as exc:
            FreezeFrame(timestamp=5.0).predict_metadata(video_meta_1s)
        assert exc.value.errors[0].code is PlanErrorCode.OP_TIMESTAMP_OUT_OF_RANGE

    def test_timestamp_out_of_range_raises_compile(self, video_meta_1s):
        with pytest.raises(ValueError, match="must be less than"):
            FreezeFrame(timestamp=5.0, duration=0.5).to_ffmpeg_filter(_ctx(video_meta_1s))

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValidationError):
            FreezeFrame(timestamp=-1.0)

    def test_zero_duration_raises(self):
        with pytest.raises(ValidationError):
            FreezeFrame(timestamp=0.5, duration=0)


class TestSilenceRemoval:
    """SilenceRemoval predicts the cut frame count and compiles select windows.

    The end-to-end cut behavior (which frames survive, audio sync) is covered
    in ``test_native_transform_streaming.py::TestSilenceRemovalStreaming``.
    """

    @pytest.fixture
    def meta_5s(self):
        """5-second @ 10fps source metadata (50 frames)."""
        return VideoMetadata(height=32, width=32, fps=10, frame_count=50, total_seconds=5.0)

    @pytest.fixture
    def transcription_with_gap(self):
        """Speech at 0-1s and 3-4s (silence gap 1-3s)."""
        return _make_transcription(
            [
                (0.0, 0.5, "hello"),
                (0.5, 1.0, "world"),
                (3.0, 3.5, "foo"),
                (3.5, 4.0, "bar"),
            ]
        )

    def test_predict_cuts_silence(self, meta_5s, transcription_with_gap):
        predicted = SilenceRemoval(min_silence_duration=1.0, padding=0.0).predict_metadata(
            meta_5s, transcription=transcription_with_gap
        )
        assert predicted.frame_count < meta_5s.frame_count

    def test_predict_no_silence_unchanged(self, meta_5s):
        transcription = _make_transcription([(float(i), float(i + 1), f"word{i}") for i in range(5)])
        predicted = SilenceRemoval(min_silence_duration=1.0, padding=0.0).predict_metadata(
            meta_5s, transcription=transcription
        )
        assert predicted.frame_count == meta_5s.frame_count

    def test_predict_without_transcription_is_identity(self, meta_5s):
        # No transcription in the validate context -> predict_metadata is identity
        # (the raise lives on the compile path, asserted below).
        predicted = SilenceRemoval().predict_metadata(meta_5s)
        assert predicted.frame_count == meta_5s.frame_count

    def test_padding_keeps_at_least_as_many_frames(self, meta_5s, transcription_with_gap):
        padded = SilenceRemoval(min_silence_duration=1.0, padding=0.5).predict_metadata(
            meta_5s, transcription=transcription_with_gap
        )
        unpadded = SilenceRemoval(min_silence_duration=1.0, padding=0.0).predict_metadata(
            meta_5s, transcription=transcription_with_gap
        )
        assert padded.frame_count >= unpadded.frame_count

    def test_compile_keep_windows(self, meta_5s, transcription_with_gap):
        ctx = _ctx(meta_5s, context={"transcription": transcription_with_gap})
        chain = SilenceRemoval(min_silence_duration=1.0, padding=0.0).to_ffmpeg_filter(ctx)
        assert chain is not None
        assert chain.startswith("select='")
        assert "between(n," in chain

    def test_compile_missing_context_raises(self, meta_5s):
        ctx = _ctx(meta_5s)  # no transcription in context
        with pytest.raises(ValueError, match="requires transcription"):
            SilenceRemoval().to_ffmpeg_filter(ctx)

    def test_compile_audio_missing_context_raises(self, meta_5s):
        ctx = _ctx(meta_5s)
        with pytest.raises(ValueError, match="requires transcription"):
            SilenceRemoval().to_ffmpeg_audio_filter(ctx)

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="min_silence_duration"):
            SilenceRemoval(min_silence_duration=0)
        with pytest.raises(ValueError, match="padding"):
            SilenceRemoval(padding=-1)


class TestCutDurationErrors:
    """Typed `PlanValidationError` from the cut transforms' `predict_metadata`."""

    def test_cut_seconds_end_exceeds_duration(self):
        meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        with pytest.raises(PlanValidationError) as exc:
            CutSeconds(start=0.0, end=20.0).predict_metadata(meta)
        assert str(exc.value) == "end time (20.0) exceeds video duration (10.0)"
        err = exc.value.errors[0]
        assert err.code is PlanErrorCode.CUT_EXCEEDS_DURATION
        assert err.op == "cut"
        assert err.field == "end"
        assert err.value == 20.0
        assert err.limit == 10.0

    def test_cut_frames_end_exceeds_count(self):
        meta = VideoMetadata(height=500, width=800, fps=24, frame_count=100, total_seconds=10.0)
        with pytest.raises(PlanValidationError) as exc:
            CutFrames(start=0, end=200).predict_metadata(meta)
        assert str(exc.value) == "end frame (200) exceeds frame count (100)"
        err = exc.value.errors[0]
        assert err.code is PlanErrorCode.CUT_EXCEEDS_DURATION
        assert err.op == "cut_frames"
        assert err.field == "end"
        assert err.value == 200
        assert err.limit == 100


class TestCutDurationTolerance:
    """`DURATION_EPS` boundary behavior for the cut transforms' `predict_metadata`."""

    def test_cut_seconds_end_equals_total_passes(self):
        meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        result = CutSeconds(start=0.0, end=10.0).predict_metadata(meta)
        assert result.total_seconds == 10.0

    def test_cut_seconds_within_eps_passes(self):
        meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        # total + 5e-4 is inside DURATION_EPS, so it must pass.
        CutSeconds(start=0.0, end=10.0 + 5e-4).predict_metadata(meta)

    def test_cut_seconds_beyond_eps_rejects(self):
        meta = VideoMetadata(height=500, width=800, fps=24, frame_count=240, total_seconds=10.0)
        # total + 2e-3 is beyond DURATION_EPS, so it must reject.
        with pytest.raises(PlanValidationError) as exc:
            CutSeconds(start=0.0, end=10.0 + 2e-3).predict_metadata(meta)
        assert exc.value.errors[0].code is PlanErrorCode.CUT_EXCEEDS_DURATION

    def test_cut_frames_integer_parity(self):
        # Frames are ints; the seconds-scale eps never flips the compare.
        meta = VideoMetadata(height=500, width=800, fps=24, frame_count=100, total_seconds=10.0)
        # end == frame_count passes; end == frame_count + 1 rejects.
        assert CutFrames(start=0, end=100).predict_metadata(meta).frame_count == 100
        with pytest.raises(PlanValidationError):
            CutFrames(start=0, end=101).predict_metadata(meta)

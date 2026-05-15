import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from tests.test_config import TEST_FONT_PATH
from videopython.audio import Audio, AudioMetadata
from videopython.base.description import BoundingBox
from videopython.base.video import Video
from videopython.editing.effects import (
    Blur,
    ColorGrading,
    Fade,
    FullImageOverlay,
    KenBurns,
    TextOverlay,
    Vignette,
    VolumeAdjust,
    Zoom,
)
from videopython.editing.operation import TimeRange


def _write_overlay(arr: np.ndarray, tmp_path) -> str:
    """Persist an overlay array to a PNG and return the path for FullImageOverlay.source."""
    path = tmp_path / "overlay.png"
    if arr.shape[-1] == 3:
        Image.fromarray(arr, mode="RGB").save(path)
    else:
        Image.fromarray(arr, mode="RGBA").save(path)
    return str(path)


def test_full_image_overlay_rgba(black_frames_test_video, tmp_path):
    overlay_shape = (*black_frames_test_video.frame_shape[:2], 4)  # RGBA
    overlay = 255 * np.ones(shape=overlay_shape, dtype=np.uint8)
    overlay[:, :, 3] = 127
    source = _write_overlay(overlay, tmp_path)

    original_shape = black_frames_test_video.video_shape
    overlayed_video = FullImageOverlay(source=source).apply(black_frames_test_video)

    assert (overlayed_video.frames.flatten() == 127).all()
    assert overlayed_video.video_shape == original_shape


def test_full_image_overlay_rgb(black_frames_test_video, tmp_path):
    overlay = 255 * np.ones(shape=black_frames_test_video.frame_shape, dtype=np.uint8)
    source = _write_overlay(overlay, tmp_path)
    original_shape = black_frames_test_video.video_shape
    original_audio_length = len(black_frames_test_video.audio)
    overlayed_video = FullImageOverlay(source=source, alpha=0.5).apply(black_frames_test_video)

    assert (overlayed_video.frames.flatten() == 127).all()
    assert overlayed_video.video_shape == original_shape
    assert len(overlayed_video.audio) == original_audio_length


def test_full_image_overlay_with_fade(black_frames_test_video, tmp_path):
    overlay = 255 * np.ones(shape=black_frames_test_video.frame_shape, dtype=np.uint8)
    source = _write_overlay(overlay, tmp_path)
    original_shape = black_frames_test_video.video_shape
    overlayed_video = FullImageOverlay(source=source, alpha=0.5, fade_time=2.0).apply(black_frames_test_video)

    assert overlayed_video.video_shape == original_shape


def test_zoom_in_out(small_video):
    zoomed_in_video = Zoom(zoom_factor=2.0, mode="in").apply(small_video)
    zoomed_out_video = Zoom(zoom_factor=2.0, mode="out").apply(small_video)

    assert zoomed_in_video.video_shape == small_video.video_shape
    assert zoomed_in_video.metadata.frame_count == small_video.metadata.frame_count

    assert zoomed_out_video.video_shape == small_video.video_shape
    assert zoomed_out_video.metadata.frame_count == small_video.metadata.frame_count


def test_effect_window(small_video):
    blur = Blur(mode="constant", iterations=10, window=TimeRange(start=6.0))
    small_video_with_blur = blur.apply(small_video.copy())
    assert (small_video.frames[0] == small_video_with_blur.frames[0]).all()
    assert (small_video.frames[-1] != small_video_with_blur.frames[-1]).any()


class TestColorGrading:
    def test_default_no_change(self, small_video):
        original_shape = small_video.video_shape
        result = ColorGrading().apply(small_video)
        assert result.video_shape == original_shape

    def test_brightness_increase(self, black_frames_test_video):
        result = ColorGrading(brightness=0.5).apply(black_frames_test_video)
        assert result.frames.mean() > 0

    def test_saturation_zero_grayscale(self, small_video):
        result = ColorGrading(saturation=0.0).apply(small_video)
        for frame in result.frames[:5]:
            r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
            assert np.allclose(r, g, atol=5) or np.allclose(g, b, atol=5)

    def test_preserves_shape(self, small_video):
        result = ColorGrading(brightness=0.1, contrast=1.2, saturation=0.8).apply(small_video)
        assert result.video_shape == small_video.video_shape
        assert result.frame_shape == small_video.frame_shape

    def test_invalid_params_raise(self):
        with pytest.raises(ValidationError):
            ColorGrading(brightness=2.0)
        with pytest.raises(ValidationError):
            ColorGrading(contrast=0.1)
        with pytest.raises(ValidationError):
            ColorGrading(saturation=-1.0)
        with pytest.raises(ValidationError):
            ColorGrading(temperature=2.0)

    def test_windowed_application(self, small_video):
        original_first = small_video.frames[0].copy()
        result = ColorGrading(brightness=0.5, window=TimeRange(start=5.0)).apply(small_video.copy())
        assert np.array_equal(result.frames[0], original_first)


class TestVignette:
    def test_vignette_darkens_edges(self, black_frames_test_video):
        black_frames_test_video.frames[:] = 200
        h, w = black_frames_test_video.frames[0].shape[:2]
        result = Vignette(strength=0.8, radius=1.0).apply(black_frames_test_video)
        result_corner = float(result.frames[0][0, 0].mean())
        result_center = float(result.frames[0][h // 2, w // 2].mean())
        assert result_corner < result_center

    def test_preserves_shape(self, small_video):
        result = Vignette(strength=0.5).apply(small_video)
        assert result.video_shape == small_video.video_shape

    def test_zero_strength_minimal_change(self, small_video):
        original_mean = small_video.frames.mean()
        result = Vignette(strength=0.0).apply(small_video.copy())
        assert abs(result.frames.mean() - original_mean) < 1

    def test_invalid_params_raise(self):
        with pytest.raises(ValidationError):
            Vignette(strength=2.0)
        with pytest.raises(ValidationError):
            Vignette(strength=-0.5)
        with pytest.raises(ValidationError):
            Vignette(radius=0.1)

    def test_windowed_application(self, small_video):
        original_first = small_video.frames[0].copy()
        result = Vignette(strength=0.8, window=TimeRange(start=5.0)).apply(small_video.copy())
        assert np.array_equal(result.frames[0], original_first)


class TestKenBurns:
    def test_preserves_shape(self, small_video):
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        result = KenBurns(start_region=start, end_region=end).apply(small_video)
        assert result.video_shape == small_video.video_shape
        assert result.frame_shape == small_video.frame_shape

    def test_zoom_in_effect(self, black_frames_test_video):
        for i, frame in enumerate(black_frames_test_video.frames):
            frame[:] = i * 2
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        result = KenBurns(start_region=start, end_region=end).apply(black_frames_test_video)
        assert result.video_shape == black_frames_test_video.video_shape

    def test_zoom_out_effect(self, black_frames_test_video):
        start = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        end = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        result = KenBurns(start_region=start, end_region=end).apply(black_frames_test_video)
        assert result.video_shape == black_frames_test_video.video_shape

    def test_pan_effect(self, small_video):
        start = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)
        end = BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5)
        result = KenBurns(start_region=start, end_region=end).apply(small_video)
        assert result.video_shape == small_video.video_shape

    def test_easing_options(self, black_frames_test_video):
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        for easing in ["linear", "ease_in", "ease_out", "ease_in_out"]:
            result = KenBurns(start_region=start, end_region=end, easing=easing).apply(black_frames_test_video.copy())
            assert result.video_shape == black_frames_test_video.video_shape

    def test_invalid_region_raises(self):
        valid = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)
        with pytest.raises(ValidationError):
            KenBurns(start_region=BoundingBox(x=-0.1, y=0.0, width=0.5, height=0.5), end_region=valid)
        with pytest.raises(ValidationError):
            KenBurns(start_region=BoundingBox(x=0.8, y=0.0, width=0.5, height=0.5), end_region=valid)
        with pytest.raises(ValidationError):
            KenBurns(start_region=BoundingBox(x=0.0, y=0.0, width=0.0, height=0.5), end_region=valid)

    def test_invalid_easing_raises(self):
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        with pytest.raises(ValidationError):
            KenBurns(start_region=start, end_region=end, easing="invalid")

    def test_windowed_application(self, small_video):
        original_first = small_video.frames[0].copy()
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        result = KenBurns(start_region=start, end_region=end, window=TimeRange(start=5.0)).apply(small_video.copy())
        assert np.array_equal(result.frames[0], original_first)


@pytest.fixture
def video_1s():
    """1-second video at 30fps with non-black frames."""
    frames = np.full((30, 64, 64, 3), 200, dtype=np.uint8)
    return Video.from_frames(frames, fps=30)


@pytest.fixture
def video_with_audio():
    """1-second video at 30fps with a sine wave audio track."""
    frames = np.full((30, 64, 64, 3), 200, dtype=np.uint8)
    video = Video.from_frames(frames, fps=30)
    sample_rate = 44100
    t = np.linspace(0, 1.0, sample_rate, dtype=np.float32)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    video.audio = Audio(
        data=audio_data,
        metadata=AudioMetadata(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            duration_seconds=1.0,
            frame_count=sample_rate,
        ),
    )
    return video


class TestFade:
    def test_fade_in_starts_black(self, video_1s):
        result = Fade(mode="in", duration=0.5).apply(video_1s)
        assert result.frames[0].mean() == 0

    def test_fade_out_ends_black(self, video_1s):
        result = Fade(mode="out", duration=0.5).apply(video_1s)
        assert result.frames[-1].mean() == 0

    def test_fade_in_out(self, video_1s):
        result = Fade(mode="in_out", duration=0.3).apply(video_1s)
        assert result.frames[0].mean() == 0
        assert result.frames[-1].mean() == 0
        mid = len(result.frames) // 2
        assert result.frames[mid].mean() > 100

    def test_preserves_shape(self, video_1s):
        original_shape = video_1s.video_shape
        result = Fade(mode="in", duration=0.5).apply(video_1s)
        assert result.video_shape == original_shape

    def test_windowed_apply(self, video_1s):
        original_first = video_1s.frames[0].copy()
        result = Fade(mode="out", duration=0.3, window=TimeRange(start=0.5)).apply(video_1s.copy())
        assert np.array_equal(result.frames[0], original_first)

    def test_audio_fade_in(self, video_with_audio):
        result = Fade(mode="in", duration=0.5).apply(video_with_audio)
        assert abs(result.audio.data[0]) < 0.01

    def test_audio_fade_out(self, video_with_audio):
        result = Fade(mode="out", duration=0.5).apply(video_with_audio)
        assert abs(result.audio.data[-1]) < 0.01

    def test_all_curves(self, video_1s):
        for curve in ("sqrt", "linear", "exponential"):
            result = Fade(mode="in", duration=0.3, curve=curve).apply(video_1s.copy())
            assert result.frames[0].mean() == 0
            assert result.video_shape == video_1s.video_shape

    def test_invalid_mode_raises(self):
        with pytest.raises(ValidationError):
            Fade(mode="invalid")

    def test_invalid_duration_raises(self):
        with pytest.raises(ValidationError):
            Fade(mode="in", duration=0)


class TestVolumeAdjust:
    def test_mute(self, video_with_audio):
        result = VolumeAdjust(volume=0.0).apply(video_with_audio)
        assert np.allclose(result.audio.data, 0.0)

    def test_unchanged(self, video_with_audio):
        original_audio = video_with_audio.audio.data.copy()
        result = VolumeAdjust(volume=1.0).apply(video_with_audio)
        assert np.allclose(result.audio.data, original_audio)

    def test_half_volume(self, video_with_audio):
        original_rms = np.sqrt(np.mean(video_with_audio.audio.data**2))
        result = VolumeAdjust(volume=0.5).apply(video_with_audio)
        new_rms = np.sqrt(np.mean(result.audio.data**2))
        assert np.isclose(new_rms, original_rms * 0.5, atol=0.01)

    def test_windowed_apply(self, video_with_audio):
        original_start = video_with_audio.audio.data[:100].copy()
        result = VolumeAdjust(volume=0.0, window=TimeRange(start=0.5)).apply(video_with_audio)
        assert np.allclose(result.audio.data[:100], original_start)

    def test_ramp_duration(self, video_with_audio):
        result = VolumeAdjust(volume=0.0, ramp_duration=0.1).apply(video_with_audio)
        assert abs(result.audio.data[1000]) > 0.01

    def test_frames_unchanged(self, video_with_audio):
        original_frames = video_with_audio.frames.copy()
        result = VolumeAdjust(volume=0.5).apply(video_with_audio)
        assert np.array_equal(result.frames, original_frames)

    def test_isinstance_effect(self):
        from videopython.editing.effects import Effect

        assert isinstance(VolumeAdjust(), Effect)

    def test_invalid_volume_raises(self):
        with pytest.raises(ValidationError):
            VolumeAdjust(volume=-1.0)

    def test_invalid_ramp_raises(self):
        with pytest.raises(ValidationError):
            VolumeAdjust(ramp_duration=-0.1)

    def test_silent_audio_passthrough(self, video_1s):
        result = VolumeAdjust(volume=0.5).apply(video_1s)
        assert result.audio is not None


class TestTextOverlay:
    def test_renders_text_on_black(self):
        frames = np.zeros((10, 100, 200, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=10)
        result = TextOverlay(text="Hello", font_size=20, font_filename=TEST_FONT_PATH).apply(video)
        assert result.frames.max() > 0

    def test_preserves_shape(self, video_1s):
        original_shape = video_1s.video_shape
        result = TextOverlay(text="Test", font_size=16).apply(video_1s)
        assert result.video_shape == original_shape

    def test_windowed_apply(self, video_1s):
        original_first = video_1s.frames[0].copy()
        result = TextOverlay(text="Late", font_size=16, window=TimeRange(start=0.5)).apply(video_1s.copy())
        assert np.array_equal(result.frames[0], original_first)

    def test_multiline_text(self, video_1s):
        result = TextOverlay(text="Line 1\nLine 2", font_size=16).apply(video_1s)
        assert result.video_shape == video_1s.video_shape

    def test_no_background(self):
        frames = np.full((5, 100, 200, 3), 128, dtype=np.uint8)
        video = Video.from_frames(frames, fps=10)
        result = TextOverlay(text="Hi", font_size=16, background_color=None).apply(video)
        assert result.video_shape == video.video_shape

    def test_all_anchors(self, video_1s):
        for anchor in ("center", "top_left", "top_center", "bottom_center", "bottom_left", "bottom_right"):
            result = TextOverlay(text="X", font_size=16, anchor=anchor).apply(video_1s.copy())
            assert result.video_shape == video_1s.video_shape

    def test_empty_text_raises(self):
        with pytest.raises(ValidationError):
            TextOverlay(text="")

    def test_invalid_position_raises(self):
        with pytest.raises(ValidationError):
            TextOverlay(text="Hi", position=(1.5, 0.5))

    def test_invalid_font_size_raises(self):
        with pytest.raises(ValidationError):
            TextOverlay(text="Hi", font_size=0)

    def test_word_wrap(self):
        frames = np.zeros((5, 100, 200, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=10)
        long_text = "This is a very long text that should definitely wrap to multiple lines"
        result = TextOverlay(text=long_text, font_size=16, max_width=0.5, font_filename=TEST_FONT_PATH).apply(video)
        assert result.frames.max() > 0

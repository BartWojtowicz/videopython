import numpy as np
import pytest

from tests.test_config import TEST_FONT_PATH
from videopython.base.audio import Audio, AudioMetadata
from videopython.base.description import BoundingBox
from videopython.base.effects import (
    AudioEffect,
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
from videopython.base.video import Video


def test_full_image_overlay_rgba(black_frames_test_video):
    overlay_shape = (*black_frames_test_video.frame_shape[:2], 4)  # RGBA
    overlay = 255 * np.ones(shape=overlay_shape, dtype=np.uint8)
    overlay[:, :, 3] = 127

    original_shape = black_frames_test_video.video_shape
    overlayed_video = FullImageOverlay(overlay).apply(black_frames_test_video)

    assert (overlayed_video.frames.flatten() == 127).all()
    assert overlayed_video.video_shape == original_shape


def test_full_image_overlay_rgb(black_frames_test_video):
    overlay = 255 * np.ones(shape=black_frames_test_video.frame_shape, dtype=np.uint8)
    original_shape = black_frames_test_video.video_shape
    original_audio_length = len(black_frames_test_video.audio)
    overlayed_video = FullImageOverlay(overlay, alpha=0.5).apply(black_frames_test_video)

    assert (overlayed_video.frames.flatten() == 127).all()
    assert overlayed_video.video_shape == original_shape
    assert len(overlayed_video.audio) == original_audio_length


def test_full_image_overlay_with_fade(black_frames_test_video):
    overlay = 255 * np.ones(shape=black_frames_test_video.frame_shape, dtype=np.uint8)
    original_shape = black_frames_test_video.video_shape
    overlayed_video = FullImageOverlay(overlay, alpha=0.5, fade_time=2.0).apply(black_frames_test_video)

    assert overlayed_video.video_shape == original_shape


def test_zoom_in_out(small_video):
    zoomed_in_video = Zoom(zoom_factor=2.0, mode="in").apply(small_video)
    zoomed_out_video = Zoom(zoom_factor=2.0, mode="out").apply(small_video)

    assert zoomed_in_video.video_shape == small_video.video_shape
    assert zoomed_in_video.metadata.frame_count == small_video.metadata.frame_count

    assert zoomed_out_video.video_shape == small_video.video_shape
    assert zoomed_out_video.metadata.frame_count == small_video.metadata.frame_count


def test_effect_start_argument(small_video):
    blur = Blur(mode="constant", iterations=10)
    small_video_with_blur = blur.apply(small_video.copy(), start=6.0)
    assert (small_video.frames[0] == small_video_with_blur.frames[0]).all()
    assert (small_video.frames[-1] != small_video_with_blur.frames[-1]).any()


class TestColorGrading:
    """Tests for ColorGrading effect."""

    def test_default_no_change(self, small_video):
        """Test default parameters don't significantly change video."""
        original_shape = small_video.video_shape
        effect = ColorGrading()
        result = effect.apply(small_video)

        # Default params should result in minimal change
        assert result.video_shape == original_shape

    def test_brightness_increase(self, black_frames_test_video):
        """Test brightness increase makes frames brighter."""
        effect = ColorGrading(brightness=0.5)
        result = effect.apply(black_frames_test_video)

        # Black frames with brightness increase should be brighter
        assert result.frames.mean() > 0

    def test_saturation_zero_grayscale(self, small_video):
        """Test zero saturation produces grayscale-like output."""
        effect = ColorGrading(saturation=0.0)
        result = effect.apply(small_video)

        # Check that R, G, B channels are more similar (grayscale tendency)
        for frame in result.frames[:5]:  # Check first few frames
            r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
            # In grayscale, channels should be very similar
            assert np.allclose(r, g, atol=5) or np.allclose(g, b, atol=5)

    def test_preserves_shape(self, small_video):
        """Test that color grading preserves video dimensions."""
        effect = ColorGrading(brightness=0.1, contrast=1.2, saturation=0.8)
        result = effect.apply(small_video)

        assert result.video_shape == small_video.video_shape
        assert result.frame_shape == small_video.frame_shape

    def test_invalid_params_raise(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            ColorGrading(brightness=2.0)  # > 1.0

        with pytest.raises(ValueError):
            ColorGrading(contrast=0.1)  # < 0.5

        with pytest.raises(ValueError):
            ColorGrading(saturation=-1.0)  # < 0.0

        with pytest.raises(ValueError):
            ColorGrading(temperature=2.0)  # > 1.0

    def test_partial_application(self, small_video):
        """Test applying to only part of video."""
        original_first = small_video.frames[0].copy()
        effect = ColorGrading(brightness=0.5)
        result = effect.apply(small_video, start=5.0)  # Apply from 5 seconds

        # First frame should be unchanged
        assert np.array_equal(result.frames[0], original_first)


class TestVignette:
    """Tests for Vignette effect."""

    def test_vignette_darkens_edges(self, black_frames_test_video):
        """Test that vignette darkens the edges relative to original."""
        # Set all frames to uniform brightness for predictable testing
        black_frames_test_video.frames[:] = 200

        h, w = black_frames_test_video.frames[0].shape[:2]
        original_corner = 200.0
        original_center = 200.0

        effect = Vignette(strength=0.8, radius=1.0)
        result = effect.apply(black_frames_test_video)

        result_corner = float(result.frames[0][0, 0].mean())
        result_center = float(result.frames[0][h // 2, w // 2].mean())

        # Corner should be darker (more reduction than center)
        corner_reduction = original_corner - result_corner
        center_reduction = original_center - result_center

        # Vignette should darken corners more than center
        assert corner_reduction > center_reduction
        # Corner should be noticeably darker
        assert result_corner < result_center

    def test_preserves_shape(self, small_video):
        """Test that vignette preserves video dimensions."""
        effect = Vignette(strength=0.5)
        result = effect.apply(small_video)

        assert result.video_shape == small_video.video_shape

    def test_zero_strength_minimal_change(self, small_video):
        """Test that zero strength results in minimal change."""
        original_mean = small_video.frames.mean()
        effect = Vignette(strength=0.0)
        result = effect.apply(small_video)

        # With zero strength, image should be nearly unchanged
        assert abs(result.frames.mean() - original_mean) < 1

    def test_invalid_params_raise(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            Vignette(strength=2.0)  # > 1.0

        with pytest.raises(ValueError):
            Vignette(strength=-0.5)  # < 0.0

        with pytest.raises(ValueError):
            Vignette(radius=0.1)  # < 0.5

    def test_partial_application(self, small_video):
        """Test applying to only part of video."""
        original_first = small_video.frames[0].copy()
        effect = Vignette(strength=0.8)
        result = effect.apply(small_video, start=5.0)

        # First frame should be unchanged
        assert np.array_equal(result.frames[0], original_first)


class TestKenBurns:
    """Tests for KenBurns effect."""

    def test_preserves_shape(self, small_video):
        """Test that Ken Burns preserves video dimensions."""
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        effect = KenBurns(start_region=start, end_region=end)
        result = effect.apply(small_video)

        assert result.video_shape == small_video.video_shape
        assert result.frame_shape == small_video.frame_shape

    def test_zoom_in_effect(self, black_frames_test_video):
        """Test zoom-in effect (full frame to smaller region)."""
        # Set frames to gradient for testing
        h, w = black_frames_test_video.frame_shape[:2]
        for i, frame in enumerate(black_frames_test_video.frames):
            frame[:] = i * 2  # Different brightness per frame

        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        effect = KenBurns(start_region=start, end_region=end)
        result = effect.apply(black_frames_test_video)

        assert result.video_shape == black_frames_test_video.video_shape

    def test_zoom_out_effect(self, black_frames_test_video):
        """Test zoom-out effect (smaller region to full frame)."""
        start = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        end = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        effect = KenBurns(start_region=start, end_region=end)
        result = effect.apply(black_frames_test_video)

        assert result.video_shape == black_frames_test_video.video_shape

    def test_pan_effect(self, small_video):
        """Test panning effect (same size, different position)."""
        start = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)
        end = BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5)
        effect = KenBurns(start_region=start, end_region=end)
        result = effect.apply(small_video)

        assert result.video_shape == small_video.video_shape

    def test_easing_options(self, black_frames_test_video):
        """Test all easing options work."""
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)

        for easing in ["linear", "ease_in", "ease_out", "ease_in_out"]:
            effect = KenBurns(start_region=start, end_region=end, easing=easing)
            result = effect.apply(black_frames_test_video.copy())
            assert result.video_shape == black_frames_test_video.video_shape

    def test_invalid_region_raises(self):
        """Test that invalid regions raise errors."""
        valid = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)

        # Position out of bounds
        with pytest.raises(ValueError):
            KenBurns(
                start_region=BoundingBox(x=-0.1, y=0.0, width=0.5, height=0.5),
                end_region=valid,
            )

        # Region extends beyond bounds
        with pytest.raises(ValueError):
            KenBurns(
                start_region=BoundingBox(x=0.8, y=0.0, width=0.5, height=0.5),
                end_region=valid,
            )

        # Zero dimension
        with pytest.raises(ValueError):
            KenBurns(
                start_region=BoundingBox(x=0.0, y=0.0, width=0.0, height=0.5),
                end_region=valid,
            )

    def test_invalid_easing_raises(self):
        """Test that invalid easing raises error."""
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)

        with pytest.raises(ValueError):
            KenBurns(start_region=start, end_region=end, easing="invalid")

    def test_partial_application(self, small_video):
        """Test applying to only part of video."""
        original_first = small_video.frames[0].copy()
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        effect = KenBurns(start_region=start, end_region=end)
        result = effect.apply(small_video, start=5.0)

        # First frame should be unchanged
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

    def test_partial_apply(self, video_1s):
        original_first = video_1s.frames[0].copy()
        result = Fade(mode="out", duration=0.3).apply(video_1s, start=0.5)
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
        with pytest.raises(ValueError, match="mode"):
            Fade(mode="invalid")

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError, match="duration"):
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

    def test_partial_apply(self, video_with_audio):
        original_start = video_with_audio.audio.data[:100].copy()
        result = VolumeAdjust(volume=0.0).apply(video_with_audio, start=0.5)
        assert np.allclose(result.audio.data[:100], original_start)

    def test_ramp_duration(self, video_with_audio):
        result = VolumeAdjust(volume=0.0, ramp_duration=0.1).apply(video_with_audio)
        assert abs(result.audio.data[1000]) > 0.01

    def test_frames_unchanged(self, video_with_audio):
        original_frames = video_with_audio.frames.copy()
        result = VolumeAdjust(volume=0.5).apply(video_with_audio)
        assert np.array_equal(result.frames, original_frames)

    def test_isinstance_effect(self):
        from videopython.base.effects import Effect

        assert isinstance(VolumeAdjust(), Effect)
        assert isinstance(VolumeAdjust(), AudioEffect)

    def test_invalid_volume_raises(self):
        with pytest.raises(ValueError, match="volume"):
            VolumeAdjust(volume=-1.0)

    def test_invalid_ramp_raises(self):
        with pytest.raises(ValueError, match="ramp_duration"):
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

    def test_partial_apply(self, video_1s):
        original_first = video_1s.frames[0].copy()
        result = TextOverlay(text="Late", font_size=16).apply(video_1s, start=0.5)
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
        with pytest.raises(ValueError, match="text"):
            TextOverlay(text="")

    def test_invalid_position_raises(self):
        with pytest.raises(ValueError, match="position"):
            TextOverlay(text="Hi", position=(1.5, 0.5))

    def test_invalid_font_size_raises(self):
        with pytest.raises(ValueError, match="font_size"):
            TextOverlay(text="Hi", font_size=0)

    def test_word_wrap(self):
        frames = np.zeros((5, 100, 200, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=10)
        long_text = "This is a very long text that should definitely wrap to multiple lines"
        result = TextOverlay(text=long_text, font_size=16, max_width=0.5, font_filename=TEST_FONT_PATH).apply(video)
        assert result.frames.max() > 0

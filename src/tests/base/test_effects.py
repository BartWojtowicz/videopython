import numpy as np
import pytest

from videopython.base.description import BoundingBox
from videopython.base.effects import Blur, ColorGrading, FullImageOverlay, KenBurns, Vignette, Zoom


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

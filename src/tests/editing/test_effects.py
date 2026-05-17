import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from tests.test_config import TEST_FONT_PATH
from videopython.audio import Audio, AudioMetadata
from videopython.base.description import BoundingBox
from videopython.base.video import Video, VideoMetadata
from videopython.editing.effects import (
    Blur,
    ChromaticAberration,
    ColorGrading,
    Fade,
    FilmGrain,
    Flash,
    FullImageOverlay,
    Glitch,
    ImageOverlay,
    Kaleidoscope,
    KenBurns,
    MirrorFlip,
    Pixelate,
    PunchIn,
    Shake,
    Sharpen,
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


class TestImageOverlay:
    def _solid_overlay(self, tmp_path, w, h, rgb=(255, 0, 0), a=255):
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[:, :, 0], arr[:, :, 1], arr[:, :, 2] = rgb
        arr[:, :, 3] = a
        return _write_overlay(arr, tmp_path)

    def test_rgba_alpha_blend(self, tmp_path):
        # white overlay, alpha 127, on black -> ~127 where pasted, black elsewhere
        video = Video.from_frames(np.zeros((4, 80, 80, 3), dtype=np.uint8), fps=10)
        src = self._solid_overlay(tmp_path, 40, 40, rgb=(255, 255, 255), a=127)
        result = ImageOverlay(source=src, scale=0.5, anchor="top_left", position=(0.0, 0.0)).apply(video)
        assert np.allclose(result.frames[0][:40, :40], 127, atol=1)
        assert result.frames[0][50:, 50:].max() == 0

    def test_opacity_scales_blend(self, tmp_path):
        video = Video.from_frames(np.zeros((3, 60, 60, 3), dtype=np.uint8), fps=10)
        src = self._solid_overlay(tmp_path, 30, 30, rgb=(255, 255, 255), a=255)
        full = ImageOverlay(source=src, scale=1.0, anchor="top_left", position=(0.0, 0.0), opacity=1.0).apply(
            video.copy()
        )
        half = ImageOverlay(source=src, scale=1.0, anchor="top_left", position=(0.0, 0.0), opacity=0.5).apply(
            video.copy()
        )
        assert full.frames[0][0, 0, 0] == 255
        assert half.frames[0][0, 0, 0] == 127

    def test_resolution_independence(self, tmp_path):
        # Same op, two frame sizes -> overlay width is the same FRACTION of each.
        src = self._solid_overlay(tmp_path, 20, 20)
        small = Video.from_frames(np.zeros((2, 100, 100, 3), dtype=np.uint8), fps=10)
        large = Video.from_frames(np.zeros((2, 200, 200, 3), dtype=np.uint8), fps=10)
        rs = ImageOverlay(source=src, scale=0.25, anchor="top_left", position=(0.0, 0.0)).apply(small)
        rl = ImageOverlay(source=src, scale=0.25, anchor="top_left", position=(0.0, 0.0)).apply(large)
        small_w = int((rs.frames[0][:, :, 0] == 255).any(axis=0).sum())
        large_w = int((rl.frames[0][:, :, 0] == 255).any(axis=0).sum())
        assert small_w == 25
        assert large_w == 50

    def test_all_anchors_preserve_shape(self, video_1s, tmp_path):
        src = self._solid_overlay(tmp_path, 16, 16)
        for anchor in ("center", "top_left", "top_center", "bottom_center", "bottom_left", "bottom_right"):
            result = ImageOverlay(source=src, scale=0.2, anchor=anchor).apply(video_1s.copy())
            assert result.video_shape == video_1s.video_shape

    def test_off_frame_is_noop(self, tmp_path):
        # bottom_right anchor at (0, 0) places the whole box off the top-left -> no-op, no raise.
        original = np.full((3, 50, 50, 3), 70, dtype=np.uint8)
        video = Video.from_frames(original.copy(), fps=10)
        src = self._solid_overlay(tmp_path, 20, 20)
        result = ImageOverlay(source=src, scale=0.3, anchor="bottom_right", position=(0.0, 0.0)).apply(video)
        assert np.array_equal(result.frames, original)

    def test_window_restricts_effect(self, tmp_path):
        video = Video.from_frames(np.zeros((20, 40, 40, 3), dtype=np.uint8), fps=10)
        src = self._solid_overlay(tmp_path, 40, 40, rgb=(255, 255, 255), a=255)
        result = ImageOverlay(
            source=src, scale=1.0, anchor="top_left", position=(0.0, 0.0), window=TimeRange(start=1.0)
        ).apply(video)
        assert result.frames[0].max() == 0  # before window: untouched
        assert result.frames[-1].min() == 255  # inside window: fully overlaid

    def test_preserves_shape(self, video_1s, tmp_path):
        src = self._solid_overlay(tmp_path, 24, 24)
        result = ImageOverlay(source=src, scale=0.3).apply(video_1s)
        assert result.video_shape == video_1s.video_shape

    def test_opaque_rgb_source_blends_by_opacity(self, tmp_path):
        # RGB (no alpha) source -> alpha treated as 255, so opacity alone drives the blend.
        src = _write_overlay(255 * np.ones((20, 20, 3), dtype=np.uint8), tmp_path)
        video = Video.from_frames(np.zeros((2, 40, 40, 3), dtype=np.uint8), fps=10)
        result = ImageOverlay(source=src, scale=1.0, anchor="top_left", position=(0.0, 0.0), opacity=0.5).apply(video)
        assert result.frames[0][0, 0, 0] == 127

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scale": 0.0},
            {"scale": -0.1},
            {"scale": 1.5},
            {"opacity": -0.1},
            {"opacity": 1.1},
            {"position": (1.5, 0.5)},
            {"position": (0.5, -0.2)},
        ],
    )
    def test_invalid_params_raise(self, kwargs):
        with pytest.raises(ValidationError):
            ImageOverlay(source="logo.png", **kwargs)

    def test_predict_metadata_rejects_missing_source(self):
        meta = VideoMetadata(height=100, width=100, fps=10, frame_count=10, total_seconds=1.0)
        with pytest.raises(ValueError, match="not a readable image"):
            ImageOverlay(source="/no/such/file.png").predict_metadata(meta)

    def test_predict_metadata_allows_oversized_scale(self, tmp_path):
        # Contract: geometry run() can clip is NOT rejected at validate() time.
        src = self._solid_overlay(tmp_path, 4000, 10)
        meta = VideoMetadata(height=100, width=100, fps=10, frame_count=10, total_seconds=1.0)
        assert ImageOverlay(source=src, scale=1.0).predict_metadata(meta) == meta

    # --- SVG sources (rasterised at the target resolution via resvg) ---

    def _write_svg(self, tmp_path, body, name="logo.svg"):
        p = tmp_path / name
        p.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 40" width="100" height="40">{body}</svg>'
        )
        return str(p)

    def test_svg_rasterized_at_target_resolution(self, tmp_path):
        src = self._write_svg(tmp_path, '<rect width="100" height="40" fill="rgb(0,128,255)"/>')
        video = Video.from_frames(np.zeros((2, 120, 200, 3), dtype=np.uint8), fps=10)
        f = ImageOverlay(source=src, scale=0.5, anchor="top_left", position=(0.0, 0.0)).apply(video).frames[0]
        # Width is exactly the target box (0.5 * 200) and the fill is the exact
        # SVG color -> rendered at target, not an upscaled/blurred bitmap.
        assert int((f[:, :, 2] == 255).any(axis=0).sum()) == 100
        assert (f[0:16, 0:100] == np.array([0, 128, 255], dtype=np.uint8)).all()

    def test_svg_resolution_independence(self, tmp_path):
        src = self._write_svg(tmp_path, '<rect width="100" height="40" fill="rgb(0,128,255)"/>')
        small = Video.from_frames(np.zeros((2, 120, 200, 3), dtype=np.uint8), fps=10)
        large = Video.from_frames(np.zeros((2, 240, 400, 3), dtype=np.uint8), fps=10)
        rs = ImageOverlay(source=src, scale=0.25, anchor="top_left", position=(0.0, 0.0)).apply(small)
        rl = ImageOverlay(source=src, scale=0.25, anchor="top_left", position=(0.0, 0.0)).apply(large)
        assert int((rs.frames[0][:, :, 2] == 255).any(axis=0).sum()) == 50  # 0.25 * 200
        assert int((rl.frames[0][:, :, 2] == 255).any(axis=0).sum()) == 100  # 0.25 * 400

    def test_svg_transparent_background(self, tmp_path):
        src = self._write_svg(tmp_path, '<circle cx="50" cy="20" r="18" fill="red"/>', name="dot.svg")
        video = Video.from_frames(np.full((2, 80, 80, 3), 90, dtype=np.uint8), fps=10)
        f = ImageOverlay(source=src, scale=1.0, anchor="top_left", position=(0.0, 0.0)).apply(video).frames[0]
        assert np.array_equal(f[0, 0], [90, 90, 90])  # corner outside the dot: transparent -> untouched
        assert f[:, :, 0].max() > 150  # the red dot is composited

    def test_svg_predict_metadata_accepts_valid(self, tmp_path):
        src = self._write_svg(tmp_path, '<rect width="100" height="40" fill="blue"/>')
        meta = VideoMetadata(height=50, width=50, fps=10, frame_count=5, total_seconds=0.5)
        assert ImageOverlay(source=src).predict_metadata(meta) == meta

    def test_svg_predict_metadata_rejects_malformed(self, tmp_path):
        bad = tmp_path / "broken.svg"
        bad.write_text("<svg>oops")
        meta = VideoMetadata(height=50, width=50, fps=10, frame_count=5, total_seconds=0.5)
        with pytest.raises(ValueError, match="not a readable image"):
            ImageOverlay(source=str(bad)).predict_metadata(meta)

    def test_predict_metadata_rejects_missing_svg(self):
        meta = VideoMetadata(height=50, width=50, fps=10, frame_count=5, total_seconds=0.5)
        with pytest.raises(ValueError, match="not a readable image"):
            ImageOverlay(source="/no/such/logo.svg").predict_metadata(meta)


@pytest.fixture
def synthetic_color_video():
    """1-second 24fps video with random per-frame content (deterministic)."""
    rng = np.random.default_rng(42)
    frames = rng.integers(0, 255, (24, 64, 64, 3), dtype=np.uint8)
    return Video.from_frames(frames, fps=24)


class TestShake:
    def test_random_changes_frames(self, synthetic_color_video):
        original = synthetic_color_video.frames.copy()
        result = Shake(intensity_px=4.0, mode="random", seed=7).apply(synthetic_color_video.copy())
        assert result.video_shape == synthetic_color_video.video_shape
        assert not np.array_equal(result.frames, original)

    def test_rhythmic_oscillates(self, synthetic_color_video):
        result = Shake(intensity_px=4.0, mode="rhythmic", frequency_hz=4.0).apply(synthetic_color_video.copy())
        assert result.video_shape == synthetic_color_video.video_shape

    def test_decay_first_frame_changes_more_than_last(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (30, 64, 64, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=30)
        original = video.frames.copy()
        result = Shake(intensity_px=8.0, mode="decay", seed=1).apply(video)
        diff_first = np.abs(result.frames[0].astype(int) - original[0].astype(int)).mean()
        diff_last = np.abs(result.frames[-1].astype(int) - original[-1].astype(int)).mean()
        assert diff_first > diff_last

    def test_seed_reproducible(self, synthetic_color_video):
        a = Shake(intensity_px=4.0, mode="random", seed=42).apply(synthetic_color_video.copy())
        b = Shake(intensity_px=4.0, mode="random", seed=42).apply(synthetic_color_video.copy())
        assert np.array_equal(a.frames, b.frames)

    def test_invalid_intensity_raises(self):
        with pytest.raises(ValidationError):
            Shake(intensity_px=0)


class TestPunchIn:
    def test_preserves_shape(self, synthetic_color_video):
        result = PunchIn(zoom_factor=1.5, attack_frames=3).apply(synthetic_color_video)
        assert result.video_shape == synthetic_color_video.video_shape

    def test_attack_progression(self):
        frames = np.tile(np.arange(64, dtype=np.uint8)[:, None], (1, 64))
        frames = np.broadcast_to(frames[None, :, :, None], (12, 64, 64, 3)).copy()
        video = Video.from_frames(frames, fps=12)
        original = video.frames.copy()
        result = PunchIn(zoom_factor=2.0, attack_frames=4).apply(video)
        # First frame should be untouched (zoom 1.0), middle frame zoomed in
        assert np.array_equal(result.frames[0], original[0])
        assert not np.array_equal(result.frames[6], original[6])

    def test_release_returns_toward_original(self):
        frames = np.tile(np.arange(64, dtype=np.uint8)[:, None], (1, 64))
        frames = np.broadcast_to(frames[None, :, :, None], (20, 64, 64, 3)).copy()
        video = Video.from_frames(frames, fps=20)
        original = video.frames.copy()
        result = PunchIn(zoom_factor=2.0, attack_frames=3, release_frames=3).apply(video)
        # Final frame should be (close to) the original because release ends at zoom=1.0
        assert np.array_equal(result.frames[-1], original[-1])

    def test_invalid_zoom_raises(self):
        with pytest.raises(ValidationError):
            PunchIn(zoom_factor=0.9)
        with pytest.raises(ValidationError):
            PunchIn(zoom_factor=1.5, attack_frames=-1)


class TestFlash:
    def test_peak_blends_toward_color(self):
        frames = np.zeros((10, 32, 32, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=10)
        result = Flash(color=(255, 255, 255), peak_alpha=1.0, attack_frames=2, decay_frames=2).apply(video)
        # The frame right after attack should be (close to) fully white
        assert result.frames[2].mean() > 240

    def test_outside_attack_decay_unchanged(self):
        frames = np.zeros((10, 32, 32, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=10)
        result = Flash(color=(255, 0, 0), peak_alpha=1.0, attack_frames=2, decay_frames=2).apply(video)
        # Frames after attack+decay should still be black
        assert result.frames[-1].mean() == 0

    def test_preserves_shape(self, synthetic_color_video):
        result = Flash(peak_alpha=0.5).apply(synthetic_color_video)
        assert result.video_shape == synthetic_color_video.video_shape

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValidationError):
            Flash(peak_alpha=0)
        with pytest.raises(ValidationError):
            Flash(peak_alpha=1.5)


class TestChromaticAberration:
    def test_horizontal_shifts_channels(self):
        frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        frames[:, :, 15, :] = 200  # vertical line in the middle (all channels)
        video = Video.from_frames(frames, fps=4)
        result = ChromaticAberration(shift_px=3, mode="horizontal").apply(video)
        # Red channel should be shifted right, blue shifted left
        # So at column 18, red is bright; at column 12, blue is bright
        assert result.frames[0, 16, 18, 0] > 100  # red shifted right
        assert result.frames[0, 16, 12, 2] > 100  # blue shifted left

    def test_radial_preserves_shape(self, synthetic_color_video):
        result = ChromaticAberration(shift_px=4, mode="radial").apply(synthetic_color_video)
        assert result.video_shape == synthetic_color_video.video_shape

    def test_vertical_mode(self, synthetic_color_video):
        result = ChromaticAberration(shift_px=4, mode="vertical").apply(synthetic_color_video)
        assert result.video_shape == synthetic_color_video.video_shape

    def test_invalid_shift_raises(self):
        with pytest.raises(ValidationError):
            ChromaticAberration(shift_px=0)


class TestGlitch:
    def test_changes_frames(self, synthetic_color_video):
        original = synthetic_color_video.frames.copy()
        result = Glitch(intensity=0.5, seed=1).apply(synthetic_color_video.copy())
        assert not np.array_equal(result.frames, original)
        assert result.video_shape == synthetic_color_video.video_shape

    def test_seed_reproducible(self, synthetic_color_video):
        a = Glitch(intensity=0.5, seed=99).apply(synthetic_color_video.copy())
        b = Glitch(intensity=0.5, seed=99).apply(synthetic_color_video.copy())
        assert np.array_equal(a.frames, b.frames)

    def test_invalid_params_raise(self):
        with pytest.raises(ValidationError):
            Glitch(intensity=0)
        with pytest.raises(ValidationError):
            Glitch(intensity=1.5)
        with pytest.raises(ValidationError):
            Glitch(slice_count=0)


class TestFilmGrain:
    def test_grain_increases_variance(self):
        frames = np.full((8, 64, 64, 3), 128, dtype=np.uint8)
        video = Video.from_frames(frames, fps=8)
        original_std = video.frames.std()
        result = FilmGrain(intensity=0.1, seed=0).apply(video)
        assert result.frames.std() > original_std

    def test_monochrome_vs_color(self):
        frames = np.full((4, 32, 32, 3), 128, dtype=np.uint8)
        a = FilmGrain(intensity=0.1, monochrome=True, seed=0).apply(Video.from_frames(frames.copy(), fps=4))
        b = FilmGrain(intensity=0.1, monochrome=False, seed=0).apply(Video.from_frames(frames.copy(), fps=4))
        # Monochrome: per-pixel R==G==B; color: channels differ
        mono_diff = np.abs(a.frames[..., 0].astype(int) - a.frames[..., 1].astype(int)).max()
        color_diff = np.abs(b.frames[..., 0].astype(int) - b.frames[..., 1].astype(int)).max()
        assert mono_diff == 0
        assert color_diff > 0

    def test_seed_reproducible(self):
        frames = np.full((4, 32, 32, 3), 128, dtype=np.uint8)
        a = FilmGrain(intensity=0.1, seed=42).apply(Video.from_frames(frames.copy(), fps=4))
        b = FilmGrain(intensity=0.1, seed=42).apply(Video.from_frames(frames.copy(), fps=4))
        assert np.array_equal(a.frames, b.frames)

    def test_invalid_intensity_raises(self):
        with pytest.raises(ValidationError):
            FilmGrain(intensity=0)
        with pytest.raises(ValidationError):
            FilmGrain(intensity=2.0)


class TestSharpen:
    def test_preserves_shape(self, synthetic_color_video):
        result = Sharpen(amount=1.0).apply(synthetic_color_video)
        assert result.video_shape == synthetic_color_video.video_shape

    def test_zero_amount_returns_unchanged(self, synthetic_color_video):
        original = synthetic_color_video.frames.copy()
        result = Sharpen(amount=0.0).apply(synthetic_color_video.copy())
        assert np.array_equal(result.frames, original)

    def test_sharpens_blurred_edge(self):
        # A vertical edge: left half 0, right half 255.
        frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        frames[:, :, 16:, :] = 255
        # Blur it first
        from videopython.editing.effects import Blur

        blurred = Blur(mode="constant", iterations=10, kernel_size=(7, 7)).apply(
            Video.from_frames(frames.copy(), fps=4)
        )
        sharpened = Sharpen(amount=2.0, kernel_size=5).apply(Video.from_frames(blurred.frames.copy(), fps=4))
        # The edge transition should be steeper after sharpening
        blurred_gradient = float(np.abs(np.diff(blurred.frames[0, 16, :, 0].astype(int))).max())
        sharp_gradient = float(np.abs(np.diff(sharpened.frames[0, 16, :, 0].astype(int))).max())
        assert sharp_gradient > blurred_gradient

    def test_even_kernel_raises(self):
        with pytest.raises(ValidationError):
            Sharpen(amount=1.0, kernel_size=4)

    def test_invalid_amount_raises(self):
        with pytest.raises(ValidationError):
            Sharpen(amount=-0.1)
        with pytest.raises(ValidationError):
            Sharpen(amount=5.0)


class TestPixelate:
    def test_full_frame_blocks_uniform(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=4)
        result = Pixelate(block_size=16).apply(video)
        # Each 16x16 block should be uniform per-channel after nearest-neighbour upscale
        block = result.frames[0, :16, :16]
        for c in range(3):
            assert block[..., c].std() < 1.0

    def test_region_only_affects_region(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=4)
        original = video.frames.copy()
        result = Pixelate(
            block_size=8,
            region=BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5),
        ).apply(video)
        # Top-left corner outside the region should be unchanged
        assert np.array_equal(result.frames[0, :16, :16], original[0, :16, :16])
        # Bottom-right corner inside the region should differ
        assert not np.array_equal(result.frames[0, 48:, 48:], original[0, 48:, 48:])

    def test_invalid_block_size_raises(self):
        with pytest.raises(ValidationError):
            Pixelate(block_size=1)


class TestMirrorFlip:
    def test_horizontal_reverses_columns(self):
        frames = np.tile(np.arange(64, dtype=np.uint8)[None, :], (64, 1))
        frames = np.broadcast_to(frames[None, :, :, None], (4, 64, 64, 3)).copy()
        original = frames.copy()
        video = Video.from_frames(frames, fps=4)
        result = MirrorFlip(mode="horizontal").apply(video)
        # Column 0 should now equal what was column 63 in the original
        assert np.array_equal(result.frames[0, :, 0], original[0, :, 63])

    def test_vertical_reverses_rows(self):
        frames = np.tile(np.arange(64, dtype=np.uint8)[:, None], (1, 64))
        frames = np.broadcast_to(frames[None, :, :, None], (4, 64, 64, 3)).copy()
        original = frames.copy()
        video = Video.from_frames(frames, fps=4)
        result = MirrorFlip(mode="vertical").apply(video)
        assert np.array_equal(result.frames[0, 0, :], original[0, 63, :])

    def test_mirror_left_makes_right_match_left(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (2, 32, 32, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=2)
        result = MirrorFlip(mode="mirror_left").apply(video)
        left = result.frames[0, :, :16]
        right = result.frames[0, :, 16:]
        assert np.array_equal(left, right[:, ::-1])

    def test_mirror_top_makes_bottom_match_top(self):
        rng = np.random.default_rng(1)
        frames = rng.integers(0, 255, (2, 32, 32, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=2)
        result = MirrorFlip(mode="mirror_top").apply(video)
        top = result.frames[0, :16, :]
        bottom = result.frames[0, 16:, :]
        assert np.array_equal(top, bottom[::-1, :])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValidationError):
            MirrorFlip(mode="diagonal")


class TestKaleidoscope:
    def test_preserves_shape(self, synthetic_color_video):
        result = Kaleidoscope(segments=6).apply(synthetic_color_video)
        assert result.video_shape == synthetic_color_video.video_shape

    def test_has_mirror_symmetry(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=2)
        result = Kaleidoscope(segments=4).apply(video)
        # With segments=4 and angle_offset=0, the fold-reflect mapping yields
        # left-right and top-bottom mirror symmetry about the frame center.
        flipped_h = result.frames[0, :, ::-1]
        flipped_v = result.frames[0, ::-1, :]
        mae_h = float(np.abs(result.frames[0].astype(int) - flipped_h.astype(int)).mean())
        mae_v = float(np.abs(result.frames[0].astype(int) - flipped_v.astype(int)).mean())
        assert mae_h < 5, f"Kaleidoscope output lacks horizontal mirror symmetry, MAE={mae_h}"
        assert mae_v < 5, f"Kaleidoscope output lacks vertical mirror symmetry, MAE={mae_v}"

    def test_segments_bounds(self):
        with pytest.raises(ValidationError):
            Kaleidoscope(segments=1)
        with pytest.raises(ValidationError):
            Kaleidoscope(segments=100)

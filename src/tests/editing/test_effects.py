import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from tests.test_config import SMALL_VIDEO_PATH, TEST_AUDIO_PATH, TEST_FONT_PATH
from videopython.base.description import BoundingBox
from videopython.base.video import Video, VideoMetadata
from videopython.editing import VideoEdit
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
from videopython.editing.operation import Effect, FilterCtx, TimeRange


def _stream(effect: Effect, video: Video) -> np.ndarray:
    """Drive the streaming contract over ``video`` and return the stacked frames.

    The eager ``Effect.apply`` path was removed -- ``streaming_init`` +
    ``process_frame`` is the single source of truth for an effect's pixels, and
    it produces bit-identical output (no codec). This mirrors the AI-effect
    streaming template in ``tests/ai/test_effects.py``: init once with the
    stream geometry, then process each frame in order. Frames are copied so an
    in-place effect cannot mutate the caller's array.
    """
    frames = video.frames
    n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
    effect.streaming_init(n, video.fps, w, h)
    return np.stack([effect.process_frame(frames[i].copy(), i) for i in range(n)])


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

    out = _stream(FullImageOverlay(source=source), black_frames_test_video)

    assert (out.flatten() == 127).all()
    assert out.shape == black_frames_test_video.video_shape


def test_full_image_overlay_rgb(black_frames_test_video, tmp_path):
    overlay = 255 * np.ones(shape=black_frames_test_video.frame_shape, dtype=np.uint8)
    source = _write_overlay(overlay, tmp_path)
    out = _stream(FullImageOverlay(source=source, alpha=0.5), black_frames_test_video)

    assert (out.flatten() == 127).all()
    assert out.shape == black_frames_test_video.video_shape


def test_full_image_overlay_with_fade(black_frames_test_video, tmp_path):
    overlay = 255 * np.ones(shape=black_frames_test_video.frame_shape, dtype=np.uint8)
    source = _write_overlay(overlay, tmp_path)
    out = _stream(FullImageOverlay(source=source, alpha=0.5, fade_time=2.0), black_frames_test_video)

    assert out.shape == black_frames_test_video.video_shape


def test_zoom_in_out(small_video):
    zoomed_in = _stream(Zoom(zoom_factor=2.0, mode="in"), small_video)
    zoomed_out = _stream(Zoom(zoom_factor=2.0, mode="out"), small_video)

    assert zoomed_in.shape == small_video.video_shape
    assert zoomed_out.shape == small_video.video_shape


def test_effect_window(render):
    """Canonical window check: frames outside an effect's ``window`` are untouched.

    ``process_frame`` has no outer-timeline window concept -- the streaming
    engine resolves the window and routes only in-window frames through the
    effect. So the window is exercised end-to-end via a rendered plan: a Blur
    windowed to the second half of a 0-4s cut. A pre-window frame must match the
    un-windowed render (lossy x264, hence a tolerance) and an in-window frame
    must visibly change. Broader window coverage lives in
    ``test_streaming.py::TestStreamingEffects::test_effect_with_window``.
    """
    blur = {"op": "blur_effect", "mode": "constant", "iterations": 30, "window": {"start": 2.0}}
    seg = {"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 4.0}
    blurred = render(VideoEdit.from_dict({"segments": [{**seg, "operations": [blur]}]}), name="blurred.mp4")
    plain = render(VideoEdit.from_dict({"segments": [{**seg, "operations": []}]}), name="plain.mp4")

    # Frame 0 is before the window (starts at 2.0s == frame 48): ~ unblurred source.
    pre_diff = np.abs(blurred.frames[0].astype(int) - plain.frames[0].astype(int)).mean()
    # The last frame (~3.96s) is inside the window: heavily blurred -> very different.
    in_diff = np.abs(blurred.frames[-1].astype(int) - plain.frames[-1].astype(int)).mean()
    assert pre_diff < 5.0, f"pre-window frame changed: {pre_diff}"
    assert in_diff > pre_diff * 3, f"in-window frame not blurred: in={in_diff}, pre={pre_diff}"


def test_frame_effect_matches_numpy_in_pipeline(render):
    """A per-frame effect renders through ``run_to_file`` faithfully.

    Pixel effects run as numpy ``process_frame`` over the decoded rawvideo stream
    (the engine no longer compiles them to ffmpeg filters). This renders
    ``chromatic_aberration`` end-to-end and asserts it tracks its standalone numpy
    ``process_frame`` twin within an x264 tolerance -- a guard that the decode ->
    Python -> encode pipeline feeds the effect the right frames and order.
    """
    seg = {"source": SMALL_VIDEO_PATH, "start": 0.0, "end": 2.0}
    base = render(VideoEdit.from_dict({"segments": [{**seg, "operations": []}]}), name="base.mp4")
    out = render(
        VideoEdit.from_dict(
            {"segments": [{**seg, "operations": [{"op": "chromatic_aberration", "shift_px": 4, "mode": "horizontal"}]}]}
        ),
        name="chromatic.mp4",
    )
    eff = ChromaticAberration(shift_px=4, mode="horizontal")
    n, h, w = base.frames.shape[:3]
    eff.streaming_init(n, base.fps, w, h)
    ref = np.stack([eff.process_frame(base.frames[i].copy(), i) for i in range(n)])
    m = min(len(ref), len(out.frames))
    mean_diff = np.abs(ref[:m].astype(int) - out.frames[:m].astype(int)).mean()
    assert mean_diff < 6, f"chromatic_aberration diverges from its numpy twin in-pipeline: mean={mean_diff}"


def test_zoom_preserves_frame_count(render):
    """zoom is a per-frame effect, so it must preserve the cut's frame count.

    A 2s cut at 24fps is 48 frames; rendering a zoom over it must return ~48
    frames (effects are shape- and count-preserving). Guards against a regression
    where a zoom resamples or duplicates frames through the pipeline.
    """
    out = render(
        VideoEdit.from_dict(
            {
                "segments": [
                    {
                        "source": SMALL_VIDEO_PATH,
                        "start": 0.0,
                        "end": 2.0,
                        "operations": [{"op": "zoom_effect", "mode": "in", "zoom_factor": 2}],
                    }
                ]
            }
        ),
        name="zoom.mp4",
    )
    assert abs(len(out.frames) - 48) <= 2, f"zoom changed the frame count: {len(out.frames)}"


class TestColorGrading:
    def test_default_no_change(self, small_video):
        out = _stream(ColorGrading(), small_video)
        assert out.shape == small_video.video_shape

    def test_brightness_increase(self, black_frames_test_video):
        out = _stream(ColorGrading(brightness=0.5), black_frames_test_video)
        assert out.mean() > 0

    def test_saturation_zero_grayscale(self, small_video):
        out = _stream(ColorGrading(saturation=0.0), small_video)
        for frame in out[:5]:
            r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
            assert np.allclose(r, g, atol=5) or np.allclose(g, b, atol=5)

    def test_preserves_shape(self, small_video):
        out = _stream(ColorGrading(brightness=0.1, contrast=1.2, saturation=0.8), small_video)
        assert out.shape == small_video.video_shape

    def test_invalid_params_raise(self):
        with pytest.raises(ValidationError):
            ColorGrading(brightness=2.0)
        with pytest.raises(ValidationError):
            ColorGrading(contrast=0.1)
        with pytest.raises(ValidationError):
            ColorGrading(saturation=-1.0)
        with pytest.raises(ValidationError):
            ColorGrading(temperature=2.0)


class TestVignette:
    def test_vignette_darkens_edges(self, black_frames_test_video):
        black_frames_test_video.frames[:] = 200
        h, w = black_frames_test_video.frames[0].shape[:2]
        out = _stream(Vignette(strength=0.8, radius=1.0), black_frames_test_video)
        result_corner = float(out[0][0, 0].mean())
        result_center = float(out[0][h // 2, w // 2].mean())
        assert result_corner < result_center

    def test_preserves_shape(self, small_video):
        out = _stream(Vignette(strength=0.5), small_video)
        assert out.shape == small_video.video_shape

    def test_zero_strength_minimal_change(self, small_video):
        original_mean = small_video.frames.mean()
        out = _stream(Vignette(strength=0.0), small_video)
        assert abs(out.mean() - original_mean) < 1

    def test_invalid_params_raise(self):
        with pytest.raises(ValidationError):
            Vignette(strength=2.0)
        with pytest.raises(ValidationError):
            Vignette(strength=-0.5)
        with pytest.raises(ValidationError):
            Vignette(radius=0.1)


class TestKenBurns:
    def test_preserves_shape(self, small_video):
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        out = _stream(KenBurns(start_region=start, end_region=end), small_video)
        assert out.shape == small_video.video_shape

    def test_zoom_in_effect(self, black_frames_test_video):
        for i, frame in enumerate(black_frames_test_video.frames):
            frame[:] = i * 2
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        out = _stream(KenBurns(start_region=start, end_region=end), black_frames_test_video)
        assert out.shape == black_frames_test_video.video_shape

    def test_zoom_out_effect(self, black_frames_test_video):
        start = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        end = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        out = _stream(KenBurns(start_region=start, end_region=end), black_frames_test_video)
        assert out.shape == black_frames_test_video.video_shape

    def test_pan_effect(self, small_video):
        start = BoundingBox(x=0.0, y=0.0, width=0.5, height=0.5)
        end = BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5)
        out = _stream(KenBurns(start_region=start, end_region=end), small_video)
        assert out.shape == small_video.video_shape

    def test_easing_options(self, black_frames_test_video):
        start = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        end = BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5)
        for easing in ["linear", "ease_in", "ease_out", "ease_in_out"]:
            out = _stream(KenBurns(start_region=start, end_region=end, easing=easing), black_frames_test_video)
            assert out.shape == black_frames_test_video.video_shape

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


@pytest.fixture
def video_1s():
    """1-second video at 30fps with non-black frames."""
    frames = np.full((30, 64, 64, 3), 200, dtype=np.uint8)
    return Video.from_frames(frames, fps=30)


class TestFade:
    def test_fade_in_starts_black(self, video_1s):
        out = _stream(Fade(mode="in", duration=0.5), video_1s)
        assert out[0].mean() == 0

    def test_fade_out_ends_black(self, video_1s):
        out = _stream(Fade(mode="out", duration=0.5), video_1s)
        assert out[-1].mean() == 0

    def test_fade_in_out(self, video_1s):
        out = _stream(Fade(mode="in_out", duration=0.3), video_1s)
        assert out[0].mean() == 0
        assert out[-1].mean() == 0
        mid = len(out) // 2
        assert out[mid].mean() > 100

    def test_preserves_shape(self, video_1s):
        out = _stream(Fade(mode="in", duration=0.5), video_1s)
        assert out.shape == video_1s.video_shape

    def test_all_curves(self, video_1s):
        for curve in ("sqrt", "linear", "exponential"):
            out = _stream(Fade(mode="in", duration=0.3, curve=curve), video_1s)
            assert out[0].mean() == 0
            assert out.shape == video_1s.video_shape

    def test_audio_fade_in_compiles_leading_ramp(self):
        # The eager numpy audio twin is gone -- the fade's gain envelope now
        # compiles to a windowed `volume` expression. A fade-in ramps from 0 at
        # t=0 (the `between(t,0,...)` term) up to 1, the audio analogue of
        # process_frame starting at a black frame.
        ctx = FilterCtx(width=64, height=64, fps=30.0, frame_count=30, audio_label="f0")
        frag = Fade(mode="in", duration=0.5).to_ffmpeg_audio_filter(ctx)
        assert frag is not None
        assert frag.startswith("volume=volume=") and ":eval=frame" in frag
        assert "between(t,0.000000,0.500000)" in frag
        assert "sqrt((t-0.000000)/0.500000)" in frag  # default sqrt curve

    def test_audio_fade_out_compiles_trailing_ramp(self):
        # Fade-out ramps to 0 over the last `duration` of the clip (1.0s here),
        # the audio twin of process_frame ending on a black frame.
        ctx = FilterCtx(width=64, height=64, fps=30.0, frame_count=30, audio_label="f0")
        frag = Fade(mode="out", duration=0.5).to_ffmpeg_audio_filter(ctx)
        assert frag is not None
        assert "between(t,0.500000,1.000000)" in frag  # trailing ramp [stop-dur, stop]
        assert "sqrt((1.000000-t)/0.500000)" in frag

    def test_audio_fade_render_decays(self, render, tmp_path):
        # Coarse end-to-end sanity that the compiled envelope actually attenuates
        # the rendered audio tail (lossy AAC -> RMS comparison, not exact). The
        # small test video is silent, so an audible track is muxed in first.
        source = str(
            Video.from_path(SMALL_VIDEO_PATH).add_audio_from_file(TEST_AUDIO_PATH).save(tmp_path / "with_audio.mp4")
        )
        plan = VideoEdit.from_dict(
            {
                "segments": [
                    {
                        "source": source,
                        "start": 0.0,
                        "end": 3.0,
                        "operations": [{"op": "fade", "mode": "out", "duration": 1.0}],
                    }
                ]
            }
        )
        video = render(plan, name="fade_render.mp4")
        sr = video.audio.metadata.sample_rate
        head = float(np.sqrt(np.mean(video.audio.data[: int(0.5 * sr)] ** 2)))
        tail = float(np.sqrt(np.mean(video.audio.data[-int(0.3 * sr) :] ** 2)))
        assert tail < head * 0.5, f"fade-out did not decay: head={head}, tail={tail}"
        assert abs(video.audio.metadata.duration_seconds - len(video.frames) / video.fps) < 0.15

    def test_invalid_mode_raises(self):
        with pytest.raises(ValidationError):
            Fade(mode="invalid")

    def test_invalid_duration_raises(self):
        with pytest.raises(ValidationError):
            Fade(mode="in", duration=0)


class TestVolumeAdjust:
    def _ctx(self, frame_count=30, fps=30.0):
        return FilterCtx(width=64, height=64, fps=fps, frame_count=frame_count, audio_label="f0")

    def test_mute_compiles_zero_gain(self):
        frag = VolumeAdjust(volume=0.0).to_ffmpeg_audio_filter(self._ctx())
        assert frag is not None
        assert frag.startswith("volume=0.000000:enable=")

    def test_unchanged_is_noop(self):
        # volume == 1 with no ramp is a no-op -> no audio filter compiled.
        assert VolumeAdjust(volume=1.0).to_ffmpeg_audio_filter(self._ctx()) is None

    def test_half_volume_compiles_multiplier(self):
        frag = VolumeAdjust(volume=0.5).to_ffmpeg_audio_filter(self._ctx())
        assert frag == "volume=0.500000:enable='between(t,0.000000,1.000000)'"

    def test_windowed_apply_restricts_enable(self):
        # The window restricts the `enable` predicate so the gain only applies
        # within [start, stop); the old test asserted the pre-window samples were
        # untouched, which the `between` predicate now encodes.
        frag = VolumeAdjust(volume=0.0, window=TimeRange(start=0.5)).to_ffmpeg_audio_filter(self._ctx())
        assert frag is not None
        assert "between(t,0.500000,1.000000)" in frag

    def test_ramp_duration_compiles_piecewise(self):
        # A ramp_duration makes the gain a per-frame piecewise expression that
        # eases 1 -> volume and back, so the very start is not yet fully muted.
        frag = VolumeAdjust(volume=0.0, ramp_duration=0.1).to_ffmpeg_audio_filter(self._ctx())
        assert frag is not None
        assert ":eval=frame" in frag
        assert "sqrt((t-0.000000)/0.100000)" in frag

    def test_frames_unchanged(self, video_1s):
        # VolumeAdjust is pixel-passthrough: process_frame returns frames verbatim.
        out = _stream(VolumeAdjust(volume=0.5), video_1s)
        assert np.array_equal(out, video_1s.frames)

    def test_isinstance_effect(self):
        assert isinstance(VolumeAdjust(), Effect)

    def test_invalid_volume_raises(self):
        with pytest.raises(ValidationError):
            VolumeAdjust(volume=-1.0)

    def test_invalid_ramp_raises(self):
        with pytest.raises(ValidationError):
            VolumeAdjust(ramp_duration=-0.1)


class TestTextOverlay:
    def test_renders_text_on_black(self):
        frames = np.zeros((10, 100, 200, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=10)
        out = _stream(TextOverlay(text="Hello", font_size=20, font_filename=TEST_FONT_PATH), video)
        assert out.max() > 0

    def test_preserves_shape(self, video_1s):
        out = _stream(TextOverlay(text="Test", font_size=16), video_1s)
        assert out.shape == video_1s.video_shape

    def test_multiline_text(self, video_1s):
        out = _stream(TextOverlay(text="Line 1\nLine 2", font_size=16), video_1s)
        assert out.shape == video_1s.video_shape

    def test_no_background(self):
        frames = np.full((5, 100, 200, 3), 128, dtype=np.uint8)
        video = Video.from_frames(frames, fps=10)
        out = _stream(TextOverlay(text="Hi", font_size=16, background_color=None), video)
        assert out.shape == video.video_shape

    def test_all_anchors(self, video_1s):
        for anchor in ("center", "top_left", "top_center", "bottom_center", "bottom_left", "bottom_right"):
            out = _stream(TextOverlay(text="X", font_size=16, anchor=anchor), video_1s)
            assert out.shape == video_1s.video_shape

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
        out = _stream(TextOverlay(text=long_text, font_size=16, max_width=0.5, font_filename=TEST_FONT_PATH), video)
        assert out.max() > 0


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
        out = _stream(ImageOverlay(source=src, scale=0.5, anchor="top_left", position=(0.0, 0.0)), video)
        assert np.allclose(out[0][:40, :40], 127, atol=1)
        assert out[0][50:, 50:].max() == 0

    def test_opacity_scales_blend(self, tmp_path):
        video = Video.from_frames(np.zeros((3, 60, 60, 3), dtype=np.uint8), fps=10)
        src = self._solid_overlay(tmp_path, 30, 30, rgb=(255, 255, 255), a=255)
        full = _stream(ImageOverlay(source=src, scale=1.0, anchor="top_left", position=(0.0, 0.0), opacity=1.0), video)
        half = _stream(ImageOverlay(source=src, scale=1.0, anchor="top_left", position=(0.0, 0.0), opacity=0.5), video)
        assert full[0][0, 0, 0] == 255
        assert half[0][0, 0, 0] == 127

    def test_resolution_independence(self, tmp_path):
        # Same op, two frame sizes -> overlay width is the same FRACTION of each.
        src = self._solid_overlay(tmp_path, 20, 20)
        small = Video.from_frames(np.zeros((2, 100, 100, 3), dtype=np.uint8), fps=10)
        large = Video.from_frames(np.zeros((2, 200, 200, 3), dtype=np.uint8), fps=10)
        rs = _stream(ImageOverlay(source=src, scale=0.25, anchor="top_left", position=(0.0, 0.0)), small)
        rl = _stream(ImageOverlay(source=src, scale=0.25, anchor="top_left", position=(0.0, 0.0)), large)
        small_w = int((rs[0][:, :, 0] == 255).any(axis=0).sum())
        large_w = int((rl[0][:, :, 0] == 255).any(axis=0).sum())
        assert small_w == 25
        assert large_w == 50

    def test_all_anchors_preserve_shape(self, video_1s, tmp_path):
        src = self._solid_overlay(tmp_path, 16, 16)
        for anchor in ("center", "top_left", "top_center", "bottom_center", "bottom_left", "bottom_right"):
            out = _stream(ImageOverlay(source=src, scale=0.2, anchor=anchor), video_1s)
            assert out.shape == video_1s.video_shape

    def test_off_frame_is_noop(self, tmp_path):
        # bottom_right anchor at (0, 0) places the whole box off the top-left -> no-op, no raise.
        original = np.full((3, 50, 50, 3), 70, dtype=np.uint8)
        video = Video.from_frames(original.copy(), fps=10)
        src = self._solid_overlay(tmp_path, 20, 20)
        out = _stream(ImageOverlay(source=src, scale=0.3, anchor="bottom_right", position=(0.0, 0.0)), video)
        assert np.array_equal(out, original)

    def test_preserves_shape(self, video_1s, tmp_path):
        src = self._solid_overlay(tmp_path, 24, 24)
        out = _stream(ImageOverlay(source=src, scale=0.3), video_1s)
        assert out.shape == video_1s.video_shape

    def test_opaque_rgb_source_blends_by_opacity(self, tmp_path):
        # RGB (no alpha) source -> alpha treated as 255, so opacity alone drives the blend.
        src = _write_overlay(255 * np.ones((20, 20, 3), dtype=np.uint8), tmp_path)
        video = Video.from_frames(np.zeros((2, 40, 40, 3), dtype=np.uint8), fps=10)
        out = _stream(ImageOverlay(source=src, scale=1.0, anchor="top_left", position=(0.0, 0.0), opacity=0.5), video)
        assert out[0][0, 0, 0] == 127

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
        f = _stream(ImageOverlay(source=src, scale=0.5, anchor="top_left", position=(0.0, 0.0)), video)[0]
        # Width is exactly the target box (0.5 * 200) and the fill is the exact
        # SVG color -> rendered at target, not an upscaled/blurred bitmap.
        assert int((f[:, :, 2] == 255).any(axis=0).sum()) == 100
        assert (f[0:16, 0:100] == np.array([0, 128, 255], dtype=np.uint8)).all()

    def test_svg_resolution_independence(self, tmp_path):
        src = self._write_svg(tmp_path, '<rect width="100" height="40" fill="rgb(0,128,255)"/>')
        small = Video.from_frames(np.zeros((2, 120, 200, 3), dtype=np.uint8), fps=10)
        large = Video.from_frames(np.zeros((2, 240, 400, 3), dtype=np.uint8), fps=10)
        rs = _stream(ImageOverlay(source=src, scale=0.25, anchor="top_left", position=(0.0, 0.0)), small)
        rl = _stream(ImageOverlay(source=src, scale=0.25, anchor="top_left", position=(0.0, 0.0)), large)
        assert int((rs[0][:, :, 2] == 255).any(axis=0).sum()) == 50  # 0.25 * 200
        assert int((rl[0][:, :, 2] == 255).any(axis=0).sum()) == 100  # 0.25 * 400

    def test_svg_transparent_background(self, tmp_path):
        src = self._write_svg(tmp_path, '<circle cx="50" cy="20" r="18" fill="red"/>', name="dot.svg")
        video = Video.from_frames(np.full((2, 80, 80, 3), 90, dtype=np.uint8), fps=10)
        f = _stream(ImageOverlay(source=src, scale=1.0, anchor="top_left", position=(0.0, 0.0)), video)[0]
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
        out = _stream(Shake(intensity_px=4.0, mode="random", seed=7), synthetic_color_video)
        assert out.shape == synthetic_color_video.video_shape
        assert not np.array_equal(out, original)

    def test_rhythmic_oscillates(self, synthetic_color_video):
        out = _stream(Shake(intensity_px=4.0, mode="rhythmic", frequency_hz=4.0), synthetic_color_video)
        assert out.shape == synthetic_color_video.video_shape

    def test_decay_first_frame_changes_more_than_last(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (30, 64, 64, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=30)
        original = video.frames.copy()
        out = _stream(Shake(intensity_px=8.0, mode="decay", seed=1), video)
        diff_first = np.abs(out[0].astype(int) - original[0].astype(int)).mean()
        diff_last = np.abs(out[-1].astype(int) - original[-1].astype(int)).mean()
        assert diff_first > diff_last

    def test_seed_reproducible(self, synthetic_color_video):
        a = _stream(Shake(intensity_px=4.0, mode="random", seed=42), synthetic_color_video)
        b = _stream(Shake(intensity_px=4.0, mode="random", seed=42), synthetic_color_video)
        assert np.array_equal(a, b)

    def test_invalid_intensity_raises(self):
        with pytest.raises(ValidationError):
            Shake(intensity_px=0)


class TestPunchIn:
    def test_preserves_shape(self, synthetic_color_video):
        out = _stream(PunchIn(zoom_factor=1.5, attack_frames=3), synthetic_color_video)
        assert out.shape == synthetic_color_video.video_shape

    def test_attack_progression(self):
        frames = np.tile(np.arange(64, dtype=np.uint8)[:, None], (1, 64))
        frames = np.broadcast_to(frames[None, :, :, None], (12, 64, 64, 3)).copy()
        video = Video.from_frames(frames, fps=12)
        original = video.frames.copy()
        out = _stream(PunchIn(zoom_factor=2.0, attack_frames=4), video)
        # First frame should be untouched (zoom 1.0), middle frame zoomed in
        assert np.array_equal(out[0], original[0])
        assert not np.array_equal(out[6], original[6])

    def test_release_returns_toward_original(self):
        frames = np.tile(np.arange(64, dtype=np.uint8)[:, None], (1, 64))
        frames = np.broadcast_to(frames[None, :, :, None], (20, 64, 64, 3)).copy()
        video = Video.from_frames(frames, fps=20)
        original = video.frames.copy()
        out = _stream(PunchIn(zoom_factor=2.0, attack_frames=3, release_frames=3), video)
        # Final frame should be (close to) the original because release ends at zoom=1.0
        assert np.array_equal(out[-1], original[-1])

    def test_invalid_zoom_raises(self):
        with pytest.raises(ValidationError):
            PunchIn(zoom_factor=0.9)
        with pytest.raises(ValidationError):
            PunchIn(zoom_factor=1.5, attack_frames=-1)


class TestFlash:
    def test_peak_blends_toward_color(self):
        frames = np.zeros((10, 32, 32, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=10)
        out = _stream(Flash(color=(255, 255, 255), peak_alpha=1.0, attack_frames=2, decay_frames=2), video)
        # The frame right after attack should be (close to) fully white
        assert out[2].mean() > 240

    def test_outside_attack_decay_unchanged(self):
        frames = np.zeros((10, 32, 32, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=10)
        out = _stream(Flash(color=(255, 0, 0), peak_alpha=1.0, attack_frames=2, decay_frames=2), video)
        # Frames after attack+decay should still be black
        assert out[-1].mean() == 0

    def test_preserves_shape(self, synthetic_color_video):
        out = _stream(Flash(peak_alpha=0.5), synthetic_color_video)
        assert out.shape == synthetic_color_video.video_shape

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
        out = _stream(ChromaticAberration(shift_px=3, mode="horizontal"), video)
        # Red channel should be shifted right, blue shifted left
        # So at column 18, red is bright; at column 12, blue is bright
        assert out[0, 16, 18, 0] > 100  # red shifted right
        assert out[0, 16, 12, 2] > 100  # blue shifted left

    def test_radial_preserves_shape(self, synthetic_color_video):
        out = _stream(ChromaticAberration(shift_px=4, mode="radial"), synthetic_color_video)
        assert out.shape == synthetic_color_video.video_shape

    def test_vertical_mode(self, synthetic_color_video):
        out = _stream(ChromaticAberration(shift_px=4, mode="vertical"), synthetic_color_video)
        assert out.shape == synthetic_color_video.video_shape

    def test_invalid_shift_raises(self):
        with pytest.raises(ValidationError):
            ChromaticAberration(shift_px=0)


class TestGlitch:
    def test_changes_frames(self, synthetic_color_video):
        original = synthetic_color_video.frames.copy()
        out = _stream(Glitch(intensity=0.5, seed=1), synthetic_color_video)
        assert not np.array_equal(out, original)
        assert out.shape == synthetic_color_video.video_shape

    def test_seed_reproducible(self, synthetic_color_video):
        a = _stream(Glitch(intensity=0.5, seed=99), synthetic_color_video)
        b = _stream(Glitch(intensity=0.5, seed=99), synthetic_color_video)
        assert np.array_equal(a, b)

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
        out = _stream(FilmGrain(intensity=0.1, seed=0), video)
        assert out.std() > original_std

    def test_monochrome_vs_color(self):
        frames = np.full((4, 32, 32, 3), 128, dtype=np.uint8)
        a = _stream(FilmGrain(intensity=0.1, monochrome=True, seed=0), Video.from_frames(frames.copy(), fps=4))
        b = _stream(FilmGrain(intensity=0.1, monochrome=False, seed=0), Video.from_frames(frames.copy(), fps=4))
        # Monochrome: per-pixel R==G==B; color: channels differ
        mono_diff = np.abs(a[..., 0].astype(int) - a[..., 1].astype(int)).max()
        color_diff = np.abs(b[..., 0].astype(int) - b[..., 1].astype(int)).max()
        assert mono_diff == 0
        assert color_diff > 0

    def test_seed_reproducible(self):
        frames = np.full((4, 32, 32, 3), 128, dtype=np.uint8)
        a = _stream(FilmGrain(intensity=0.1, seed=42), Video.from_frames(frames.copy(), fps=4))
        b = _stream(FilmGrain(intensity=0.1, seed=42), Video.from_frames(frames.copy(), fps=4))
        assert np.array_equal(a, b)

    def test_invalid_intensity_raises(self):
        with pytest.raises(ValidationError):
            FilmGrain(intensity=0)
        with pytest.raises(ValidationError):
            FilmGrain(intensity=2.0)


class TestSharpen:
    def test_preserves_shape(self, synthetic_color_video):
        out = _stream(Sharpen(amount=1.0), synthetic_color_video)
        assert out.shape == synthetic_color_video.video_shape

    def test_zero_amount_returns_unchanged(self, synthetic_color_video):
        original = synthetic_color_video.frames.copy()
        out = _stream(Sharpen(amount=0.0), synthetic_color_video)
        assert np.array_equal(out, original)

    def test_sharpens_blurred_edge(self):
        # A vertical edge: left half 0, right half 255.
        frames = np.zeros((4, 32, 32, 3), dtype=np.uint8)
        frames[:, :, 16:, :] = 255
        # Blur it first
        blurred = _stream(
            Blur(mode="constant", iterations=10, kernel_size=(7, 7)),
            Video.from_frames(frames.copy(), fps=4),
        )
        sharpened = _stream(Sharpen(amount=2.0, kernel_size=5), Video.from_frames(blurred.copy(), fps=4))
        # The edge transition should be steeper after sharpening
        blurred_gradient = float(np.abs(np.diff(blurred[0, 16, :, 0].astype(int))).max())
        sharp_gradient = float(np.abs(np.diff(sharpened[0, 16, :, 0].astype(int))).max())
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
        out = _stream(Pixelate(block_size=16), video)
        # Each 16x16 block should be uniform per-channel after nearest-neighbour upscale
        block = out[0, :16, :16]
        for c in range(3):
            assert block[..., c].std() < 1.0

    def test_region_only_affects_region(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (4, 64, 64, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=4)
        original = video.frames.copy()
        out = _stream(
            Pixelate(block_size=8, region=BoundingBox(x=0.5, y=0.5, width=0.5, height=0.5)),
            video,
        )
        # Top-left corner outside the region should be unchanged
        assert np.array_equal(out[0, :16, :16], original[0, :16, :16])
        # Bottom-right corner inside the region should differ
        assert not np.array_equal(out[0, 48:, 48:], original[0, 48:, 48:])

    def test_invalid_block_size_raises(self):
        with pytest.raises(ValidationError):
            Pixelate(block_size=1)


class TestMirrorFlip:
    def test_horizontal_reverses_columns(self):
        frames = np.tile(np.arange(64, dtype=np.uint8)[None, :], (64, 1))
        frames = np.broadcast_to(frames[None, :, :, None], (4, 64, 64, 3)).copy()
        original = frames.copy()
        video = Video.from_frames(frames, fps=4)
        out = _stream(MirrorFlip(mode="horizontal"), video)
        # Column 0 should now equal what was column 63 in the original
        assert np.array_equal(out[0, :, 0], original[0, :, 63])

    def test_vertical_reverses_rows(self):
        frames = np.tile(np.arange(64, dtype=np.uint8)[:, None], (1, 64))
        frames = np.broadcast_to(frames[None, :, :, None], (4, 64, 64, 3)).copy()
        original = frames.copy()
        video = Video.from_frames(frames, fps=4)
        out = _stream(MirrorFlip(mode="vertical"), video)
        assert np.array_equal(out[0, 0, :], original[0, 63, :])

    def test_mirror_left_makes_right_match_left(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (2, 32, 32, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=2)
        out = _stream(MirrorFlip(mode="mirror_left"), video)
        left = out[0, :, :16]
        right = out[0, :, 16:]
        assert np.array_equal(left, right[:, ::-1])

    def test_mirror_top_makes_bottom_match_top(self):
        rng = np.random.default_rng(1)
        frames = rng.integers(0, 255, (2, 32, 32, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=2)
        out = _stream(MirrorFlip(mode="mirror_top"), video)
        top = out[0, :16, :]
        bottom = out[0, 16:, :]
        assert np.array_equal(top, bottom[::-1, :])

    def test_invalid_mode_raises(self):
        with pytest.raises(ValidationError):
            MirrorFlip(mode="diagonal")


class TestKaleidoscope:
    def test_preserves_shape(self, synthetic_color_video):
        out = _stream(Kaleidoscope(segments=6), synthetic_color_video)
        assert out.shape == synthetic_color_video.video_shape

    def test_has_mirror_symmetry(self):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, 255, (2, 64, 64, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=2)
        out = _stream(Kaleidoscope(segments=4), video)
        # With segments=4 and angle_offset=0, the fold-reflect mapping yields
        # left-right and top-bottom mirror symmetry about the frame center.
        flipped_h = out[0, :, ::-1]
        flipped_v = out[0, ::-1, :]
        mae_h = float(np.abs(out[0].astype(int) - flipped_h.astype(int)).mean())
        mae_v = float(np.abs(out[0].astype(int) - flipped_v.astype(int)).mean())
        assert mae_h < 5, f"Kaleidoscope output lacks horizontal mirror symmetry, MAE={mae_h}"
        assert mae_v < 5, f"Kaleidoscope output lacks vertical mirror symmetry, MAE={mae_v}"

    def test_segments_bounds(self):
        with pytest.raises(ValidationError):
            Kaleidoscope(segments=1)
        with pytest.raises(ValidationError):
            Kaleidoscope(segments=100)

import numpy as np

from videopython.base.effects import Blur, FullImageOverlay, Zoom


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

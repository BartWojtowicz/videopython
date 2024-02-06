import numpy as np

from videopython.base.effects import FullImageOverlay


def test_full_image_overlay_rgba(black_frames_video):
    overlay_shape = (*black_frames_video.frame_shape[:2], 4)  # RGBA
    overlay = 255 * np.ones(shape=overlay_shape, dtype=np.uint8)
    overlay[:, :, 3] = 127

    original_shape = black_frames_video.video_shape
    overlayed_video = FullImageOverlay(overlay).apply(black_frames_video)

    assert (overlayed_video.frames.flatten() == 127).all()
    assert overlayed_video.video_shape == original_shape


def test_full_image_overlay_rgb(black_frames_video):
    overlay = 255 * np.ones(shape=black_frames_video.frame_shape, dtype=np.uint8)
    original_shape = black_frames_video.video_shape
    overlayed_video = FullImageOverlay(overlay, alpha=0.5).apply(black_frames_video)

    assert (overlayed_video.frames.flatten() == 127).all()
    assert overlayed_video.video_shape == original_shape

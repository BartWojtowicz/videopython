from videopython.utils.image import SlideOverImage


def test_slide_over_image(small_image):
    slide = SlideOverImage(direction="left", video_shape=(150, small_image.shape[0]), fps=24.0, length_seconds=1.0)
    video = slide.apply(small_image)

    assert video.frames.shape[0] == 24
    assert video.frames.shape[1] == small_image.shape[0]
    assert video.frames.shape[2] == 150
    assert video.frames.shape[3] == 3

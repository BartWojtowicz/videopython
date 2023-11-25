import numpy as np

from videopython.base.compose import VideoComposer
from videopython.base.transforms import TransformationPipeline
from videopython.base.transitions import FadeTransition, InstantTransition


def test_vanilla_compose(small_video):
    original_video = small_video.copy()
    videos_to_compose = small_video.split()

    transformation_pipeline = TransformationPipeline([])
    transition = InstantTransition()
    composer = VideoComposer(transformation_pipeline, transition)
    composed_video = composer.compose(videos_to_compose)

    assert composed_video.total_seconds == original_video.total_seconds
    assert composed_video.fps == original_video.fps
    assert np.all(composed_video.frames[0] == original_video.frames[0])
    assert np.all(composed_video.frames[-1] == original_video.frames[-1])


def test_fade_compose_100_frames(small_video):
    original_video = small_video.copy()
    videos_to_compose = small_video.split()

    transformation_pipeline = TransformationPipeline([])
    transition = FadeTransition(effect_time_seconds=1)
    composer = VideoComposer(transformation_pipeline, transition)
    composed_video = composer.compose(videos_to_compose)

    assert composed_video.total_seconds == original_video.total_seconds - 1
    assert composed_video.fps == original_video.fps
    assert np.all(composed_video.frames[0] == original_video.frames[0])
    assert np.all(composed_video.frames[-1] == original_video.frames[-1])

from itertools import repeat
from multiprocessing import Pool

from videopython.base.transforms import TransformationPipeline
from videopython.base.transitions import InstantTransition, Transition
from videopython.base.video import Video


class VideoComposer:
    """
    Composes multiple Videos into single video using selected transformations
    on each video and applies transitions.
    """

    def __init__(
        self,
        transformation_pipeline: TransformationPipeline | None = None,
        transition: Transition = InstantTransition(),
    ):
        """Initializes VideoComposer.

        Args:
            transformation_pipeline: Pipeline of transformations to apply on each video.
            transition: Transition to apply between videos
        """
        self.transition = transition
        self.transformation_pipeline = transformation_pipeline

    def _apply_transformation(self, video: Video, transformation_pipeline: TransformationPipeline) -> Video:
        return transformation_pipeline(video)

    def compose(self, videos: list[Video]) -> Video:
        # Apply transformation on each video using multiprocessing pool:
        if self.transformation_pipeline:
            transformed_videos = []
            with Pool() as pool:
                transformed_videos = pool.starmap(
                    self._apply_transformation,
                    zip(videos, repeat(self.transformation_pipeline)),
                )
            videos = transformed_videos

        # Check if videos are compatible:
        self._compatibility_check(videos)

        # Apply transition:
        final_video = videos.pop(0)
        for _ in range(len(videos)):
            final_video = self.transition.apply((final_video, videos.pop(0)))

        return final_video

    @staticmethod
    def _compatibility_check(videos: list[Video]):
        assert all([videos[0].metadata.can_be_merged_with(other_video.metadata) for other_video in videos])

from videopython.base.transitions import Transition
from videopython.base.video import Video
from videopython.base.pipeline import TransformationPipeline


class VideoComposer:
    """
    Composes multiple Videos into single video using selected transformations
    on each video and applies transitions.
    """

    def __init__(self, transformation_pipeline: TransformationPipeline,
                 transition: Transition | list[Transition]):
        self.transition = transition
        self.transformation_pipeline = transformation_pipeline

    def compose(self, videos: list[Video]) -> Video:
        # TODO: Apply transitions somehow :)
        transformed_videos = []
        for video in videos:
            transformed_videos.append(
                self.transformation_pipeline.transform(video))

        self._compatibility_check(transformed_videos)
        return sum(transformed_videos)

    @staticmethod
    def _compatibility_check(videos: list[Video]):
        assert all([
            videos[0].metadata.can_be_merged_with(other_video.metadata)
            for other_video in videos
        ])

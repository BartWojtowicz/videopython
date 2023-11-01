from videopython.base.transforms import Transformation
from videopython.base.video import Video


# Early idea, should probably be rethinked a little
class TransformationPipeline:

    def __init__(self, transformations: Transformation | None):
        """Initializes pipeline."""
        self.transformations = transformations if transformations else []

    def add_transformation(self, transformation: Transformation):
        """Adds transformation to pipeline.

        Args:
            transformation: Transformation to add.

        Returns:
            Pipeline with added transformation.
        """
        self.transformations.append(transformation)
        return self

    def __str__(self):
        txt = ""
        for i, transformation in enumerate(self.transformations):
            txt += f"{i + 1}. {transformation}\n"
        return txt

    def transform(self, video: Video) -> Video:
        """Applies pipeline to video.

        Args:
            video: Video to transform.

        Returns:
            Transformed video.
        """
        frames = video.frames
        for transformation in self.transformations:
            frames = transformation.apply(frames)
        video.frames = frames
        return video

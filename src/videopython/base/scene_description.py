from __future__ import annotations

from dataclasses import dataclass

from videopython.base.frames import FrameDescription
from videopython.base.scenes import Scene


@dataclass
class SceneDescription:
    """Contains a scene and its frame descriptions.

    Attributes:
        scene: The scene object with timing information
        frame_descriptions: List of descriptions for frames sampled from this scene
    """

    scene: Scene
    frame_descriptions: list[FrameDescription]

    @property
    def num_frames_described(self) -> int:
        """Number of frames that were described in this scene."""
        return len(self.frame_descriptions)

    def get_description_summary(self) -> str:
        """Get a summary of all frame descriptions concatenated.

        Returns:
            Single string with all frame descriptions joined
        """
        return " ".join([fd.description for fd in self.frame_descriptions])

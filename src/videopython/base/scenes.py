from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scene:
    """Represents a detected scene in a video.

    A scene is a continuous segment of video where the visual content remains relatively consistent,
    bounded by scene changes or transitions.

    Attributes:
        start: Scene start time in seconds
        end: Scene end time in seconds
        start_frame: Index of the first frame in this scene
        end_frame: Index of the last frame in this scene (exclusive)
    """

    start: float
    end: float
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> float:
        """Duration of the scene in seconds."""
        return self.end - self.start

    @property
    def frame_count(self) -> int:
        """Number of frames in this scene."""
        return self.end_frame - self.start_frame

    def get_frame_indices(self, num_samples: int = 3) -> list[int]:
        """Get evenly distributed frame indices from this scene.

        Args:
            num_samples: Number of frames to sample from the scene

        Returns:
            List of frame indices evenly distributed throughout the scene
        """
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        if num_samples == 1:
            # Return middle frame
            return [self.start_frame + self.frame_count // 2]

        # Get evenly spaced frames including start and end
        step = (self.end_frame - self.start_frame - 1) / (num_samples - 1)
        return [int(self.start_frame + i * step) for i in range(num_samples)]

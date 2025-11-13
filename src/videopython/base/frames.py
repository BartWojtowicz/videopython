from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameDescription:
    """Represents a description of a video frame.

    Attributes:
        frame_index: Index of the frame in the video
        timestamp: Time in seconds when this frame appears
        description: Text description of what's in the frame
    """

    frame_index: int
    timestamp: float
    description: str

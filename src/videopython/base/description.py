from __future__ import annotations

from dataclasses import dataclass

from videopython.base.text.transcription import Transcription


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


@dataclass
class VideoDescription:
    """Complete understanding of a video including visual and audio analysis.

    Attributes:
        scene_descriptions: List of scene descriptions with frame analysis
        transcription: Optional audio transcription
    """

    scene_descriptions: list[SceneDescription]
    transcription: Transcription | None = None

    @property
    def num_scenes(self) -> int:
        """Number of scenes detected in the video."""
        return len(self.scene_descriptions)

    @property
    def total_frames_analyzed(self) -> int:
        """Total number of frames analyzed across all scenes."""
        return sum(sd.num_frames_described for sd in self.scene_descriptions)

    def get_scene_summary(self, scene_index: int) -> str:
        """Get a text summary of a specific scene.

        Args:
            scene_index: Index of the scene to summarize

        Returns:
            Text summary of the scene including timing and descriptions
        """
        if scene_index < 0 or scene_index >= len(self.scene_descriptions):
            raise ValueError(f"scene_index {scene_index} out of bounds (0-{len(self.scene_descriptions) - 1})")

        sd = self.scene_descriptions[scene_index]
        scene = sd.scene
        summary = f"Scene {scene_index + 1} ({scene.start:.2f}s - {scene.end:.2f}s, {scene.duration:.2f}s): "
        summary += sd.get_description_summary()

        return summary

    def get_full_summary(self) -> str:
        """Get a complete text summary of the entire video.

        Returns:
            Multi-line string with scene summaries and optional transcription
        """
        lines = [f"Video Analysis - {self.num_scenes} scenes, {self.total_frames_analyzed} frames analyzed\n"]

        for i in range(len(self.scene_descriptions)):
            lines.append(self.get_scene_summary(i))

        if self.transcription and self.transcription.segments:
            lines.append("\nTranscription:")
            for segment in self.transcription.segments:
                lines.append(f"  [{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")

        return "\n".join(lines)

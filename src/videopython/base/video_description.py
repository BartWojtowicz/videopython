from __future__ import annotations

from dataclasses import dataclass

from videopython.base.scene_description import SceneDescription
from videopython.base.text.transcription import Transcription


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

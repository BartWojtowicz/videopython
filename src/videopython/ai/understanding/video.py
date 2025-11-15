from __future__ import annotations

from typing import Literal

from videopython.ai.understanding.frames import ImageToText
from videopython.ai.understanding.scenes import SceneDetector
from videopython.base.scene_description import SceneDescription
from videopython.base.video import Video
from videopython.base.video_description import VideoDescription


class VideoAnalyzer:
    """Comprehensive video analysis combining scene detection, frame understanding, and transcription."""

    def __init__(
        self,
        scene_threshold: float = 0.3,
        min_scene_length: float = 0.5,
        device: str | None = None,
    ):
        """Initialize the video analyzer.

        Args:
            scene_threshold: Threshold for scene change detection (0.0-1.0)
            min_scene_length: Minimum scene duration in seconds
            device: Device for ImageToText model ('cuda', 'cpu', or None for auto)
        """
        self.scene_detector = SceneDetector(threshold=scene_threshold, min_scene_length=min_scene_length)
        self.image_to_text = ImageToText(device=device)

    def analyze(
        self,
        video: Video,
        frames_per_second: float = 1.0,
        transcribe: bool = False,
        transcription_model: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "base",
        description_prompt: str | None = None,
    ) -> VideoDescription:
        """Perform comprehensive video analysis.

        Args:
            video: Video object to analyze
            frames_per_second: Frame sampling rate for visual analysis (default: 1.0 fps)
            transcribe: Whether to generate audio transcription (default: False)
            transcription_model: Whisper model to use if transcribe=True (default: "base")
            description_prompt: Optional prompt to guide frame descriptions

        Returns:
            VideoDescription object with complete analysis
        """
        # Step 1: Detect scenes
        scenes = self.scene_detector.detect(video)

        # Step 2: Analyze frames from each scene
        scene_descriptions = []
        for scene in scenes:
            frame_descriptions = self.image_to_text.describe_scene(
                video, scene, frames_per_second=frames_per_second, prompt=description_prompt
            )
            scene_descriptions.append(SceneDescription(scene=scene, frame_descriptions=frame_descriptions))

        # Step 3: Optional transcription
        transcription = None
        if transcribe:
            from videopython.ai.understanding.transcribe import CreateTranscription

            transcriber = CreateTranscription(model_name=transcription_model)
            transcription = transcriber.transcribe(video)

        return VideoDescription(scene_descriptions=scene_descriptions, transcription=transcription)

    def analyze_scenes_only(self, video: Video) -> list[SceneDescription]:
        """Analyze video scenes without transcription (convenience method).

        Args:
            video: Video object to analyze

        Returns:
            List of SceneDescription objects
        """
        understanding = self.analyze(video, transcribe=False)
        return understanding.scene_descriptions

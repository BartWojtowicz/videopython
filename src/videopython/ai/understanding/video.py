from __future__ import annotations

from typing import Literal

import numpy as np

from videopython.ai.understanding.color import ColorAnalyzer
from videopython.ai.understanding.image import ImageToText
from videopython.base.description import Scene, SceneDescription, VideoDescription
from videopython.base.video import Video


class SceneDetector:
    """Detects scene changes in videos using histogram comparison.

    Scene changes are detected by comparing the color histograms of consecutive frames.
    When the histogram difference exceeds a threshold, a scene boundary is detected.
    """

    def __init__(self, threshold: float = 0.3, min_scene_length: float = 0.5):
        """Initialize the scene detector.

        Args:
            threshold: Sensitivity for scene change detection (0.0 to 1.0).
                      Lower values detect more scene changes. Default: 0.3
            min_scene_length: Minimum scene duration in seconds. Scenes shorter than
                            this will be merged with adjacent scenes. Default: 0.5
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if min_scene_length < 0:
            raise ValueError("min_scene_length must be non-negative")

        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.color_analyzer = ColorAnalyzer()

    def _calculate_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram difference between two frames.

        Args:
            frame1: First frame (H, W, 3) in RGB format
            frame2: Second frame (H, W, 3) in RGB format

        Returns:
            Difference score between 0.0 (identical) and 1.0 (completely different)
        """
        return self.color_analyzer.calculate_histogram_difference(frame1, frame2)

    def detect(self, video: Video) -> list[Scene]:
        """Detect scenes in a video.

        Args:
            video: Video object to analyze

        Returns:
            List of Scene objects representing detected scenes, ordered by time
        """
        if len(video.frames) == 0:
            return []

        if len(video.frames) == 1:
            # Single frame video is one scene
            return [Scene(start=0.0, end=video.total_seconds, start_frame=0, end_frame=1)]

        # Calculate frame differences
        scene_boundaries = [0]  # First frame is always a scene start

        for i in range(1, len(video.frames)):
            difference = self._calculate_histogram_difference(video.frames[i - 1], video.frames[i])

            if difference > self.threshold:
                scene_boundaries.append(i)

        # Last frame index (exclusive)
        scene_boundaries.append(len(video.frames))

        # Create Scene objects
        scenes = []
        for i in range(len(scene_boundaries) - 1):
            start_frame = scene_boundaries[i]
            end_frame = scene_boundaries[i + 1]

            start_time = start_frame / video.fps
            end_time = end_frame / video.fps

            scenes.append(
                Scene(
                    start=start_time,
                    end=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

        # Merge scenes that are too short
        if self.min_scene_length > 0:
            scenes = self._merge_short_scenes(scenes)

        return scenes

    def _merge_short_scenes(self, scenes: list[Scene]) -> list[Scene]:
        """Merge scenes that are shorter than min_scene_length.

        Args:
            scenes: List of scenes to process

        Returns:
            List of scenes with short scenes merged into adjacent ones
        """
        if not scenes:
            return scenes

        merged = [scenes[0]]

        for scene in scenes[1:]:
            last_scene = merged[-1]

            # If the last scene is too short, merge it with current scene
            if last_scene.duration < self.min_scene_length:
                # Merge by extending the previous scene to include this one
                merged[-1] = Scene(
                    start=last_scene.start,
                    end=scene.end,
                    start_frame=last_scene.start_frame,
                    end_frame=scene.end_frame,
                )
            else:
                merged.append(scene)

        # Handle edge case: if the final scene is too short, merge it backward
        if len(merged) > 1 and merged[-1].duration < self.min_scene_length:
            second_last = merged[-2]
            last = merged[-1]
            merged[-2] = Scene(
                start=second_last.start,
                end=last.end,
                start_frame=second_last.start_frame,
                end_frame=last.end_frame,
            )
            merged.pop()

        return merged


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

    async def analyze(
        self,
        video: Video,
        frames_per_second: float = 1.0,
        transcribe: bool = False,
        transcription_model: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "base",
        description_prompt: str | None = None,
        extract_colors: bool = False,
        include_full_histogram: bool = False,
    ) -> VideoDescription:
        """Perform comprehensive video analysis.

        Args:
            video: Video object to analyze
            frames_per_second: Frame sampling rate for visual analysis (default: 1.0 fps)
            transcribe: Whether to generate audio transcription (default: False)
            transcription_model: Whisper model to use if transcribe=True (default: "base")
            description_prompt: Optional prompt to guide frame descriptions
            extract_colors: Whether to extract color features from frames (default: False)
            include_full_histogram: Whether to include full HSV histogram in color features (default: False)

        Returns:
            VideoDescription object with complete analysis
        """
        # Step 1: Detect scenes
        scenes = self.scene_detector.detect(video)

        # Step 2: Analyze frames from each scene
        scene_descriptions = []
        for scene in scenes:
            frame_descriptions = await self.image_to_text.describe_scene(
                video,
                scene,
                frames_per_second=frames_per_second,
                prompt=description_prompt,
                extract_colors=extract_colors,
                include_full_histogram=include_full_histogram,
            )
            scene_descriptions.append(SceneDescription(scene=scene, frame_descriptions=frame_descriptions))

        # Step 3: Optional transcription
        transcription = None
        if transcribe:
            from videopython.ai.understanding.audio import AudioToText

            transcriber = AudioToText(model_name=transcription_model)
            transcription = await transcriber.transcribe(video)

        return VideoDescription(scene_descriptions=scene_descriptions, transcription=transcription)

    async def analyze_scenes_only(self, video: Video) -> list[SceneDescription]:
        """Analyze video scenes without transcription (convenience method).

        Args:
            video: Video object to analyze

        Returns:
            List of SceneDescription objects
        """
        understanding = await self.analyze(video, transcribe=False)
        return understanding.scene_descriptions

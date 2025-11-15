from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

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

    def _calculate_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram difference between two frames.

        Args:
            frame1: First frame (H, W, 3) in RGB format
            frame2: Second frame (H, W, 3) in RGB format

        Returns:
            Difference score between 0.0 (identical) and 1.0 (completely different)
        """
        # Convert RGB to HSV for better color comparison
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2HSV)

        # Calculate and normalize histograms for each channel (H, S, V)
        # Channel configs: (channel_index, num_bins, range)
        channels = [(0, 50, [0, 180]), (1, 60, [0, 256]), (2, 60, [0, 256])]
        histograms = []

        for channel, bins, range_vals in channels:
            hist1 = cv2.calcHist([hsv1], [channel], None, [bins], range_vals)
            hist2 = cv2.calcHist([hsv2], [channel], None, [bins], range_vals)
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            histograms.append((hist1, hist2))

        # Compare histograms using correlation (returns value between -1 and 1)
        # We use correlation because it's robust to lighting changes
        correlations = [cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) for h1, h2 in histograms]
        avg_correlation = sum(correlations) / len(correlations)

        # Convert correlation (1.0 = similar) to difference (0.0 = similar)
        difference = 1.0 - avg_correlation

        return difference

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
            from videopython.ai.understanding.audio import AudioToText

            transcriber = AudioToText(model_name=transcription_model)
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

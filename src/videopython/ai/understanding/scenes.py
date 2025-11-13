from __future__ import annotations

import cv2
import numpy as np

from videopython.base.scenes import Scene
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

        # Calculate histogram for each channel
        hist1_h = cv2.calcHist([hsv1], [0], None, [50], [0, 180])
        hist1_s = cv2.calcHist([hsv1], [1], None, [60], [0, 256])
        hist1_v = cv2.calcHist([hsv1], [2], None, [60], [0, 256])

        hist2_h = cv2.calcHist([hsv2], [0], None, [50], [0, 180])
        hist2_s = cv2.calcHist([hsv2], [1], None, [60], [0, 256])
        hist2_v = cv2.calcHist([hsv2], [2], None, [60], [0, 256])

        # Normalize histograms
        cv2.normalize(hist1_h, hist1_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist1_s, hist1_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist1_v, hist1_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2_h, hist2_h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2_s, hist2_s, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hist2_v, hist2_v, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Compare histograms using correlation (returns value between -1 and 1)
        # We use correlation because it's robust to lighting changes
        corr_h = cv2.compareHist(hist1_h, hist2_h, cv2.HISTCMP_CORREL)
        corr_s = cv2.compareHist(hist1_s, hist2_s, cv2.HISTCMP_CORREL)
        corr_v = cv2.compareHist(hist1_v, hist2_v, cv2.HISTCMP_CORREL)

        # Average correlation across channels
        avg_correlation = (corr_h + corr_s + corr_v) / 3

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

        return merged

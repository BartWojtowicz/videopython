"""Scene detection using histogram comparison.

This module provides lightweight scene detection that uses only OpenCV for
histogram-based frame comparison, without requiring any ML dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from videopython.base.description import SceneDescription

if TYPE_CHECKING:
    from videopython.base.video import Video


class SceneDetector:
    """Detects scene changes in videos using histogram comparison.

    Scene changes are detected by comparing the color histograms of consecutive frames.
    When the histogram difference exceeds a threshold, a scene boundary is detected.

    This is a lightweight implementation using only OpenCV, suitable for the base module.

    Example:
        >>> from videopython.base import Video, SceneDetector
        >>> video = Video.from_path("video.mp4")
        >>> detector = SceneDetector(threshold=0.3, min_scene_length=0.5)
        >>> scenes = detector.detect(video)
        >>> for scene in scenes:
        ...     print(f"Scene: {scene.start:.2f}s - {scene.end:.2f}s")
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

    def _calculate_hsv_histogram(self, hsv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate normalized HSV histograms for all channels.

        Args:
            hsv: HSV image as numpy array (H, W, 3)

        Returns:
            Tuple of (hue_hist, saturation_hist, value_hist), each normalized 0-1
        """
        h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [60], [0, 256])

        cv2.normalize(h_hist, h_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(s_hist, s_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(v_hist, v_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        return (h_hist, s_hist, v_hist)

    def _calculate_histogram_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram difference between two frames.

        Args:
            frame1: First frame (H, W, 3) in RGB format
            frame2: Second frame (H, W, 3) in RGB format

        Returns:
            Difference score between 0.0 (identical) and 1.0 (completely different)
        """
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2HSV)

        hist1 = self._calculate_hsv_histogram(hsv1)
        hist2 = self._calculate_hsv_histogram(hsv2)

        correlations = [cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL) for h1, h2 in zip(hist1, hist2)]
        avg_correlation = sum(correlations) / len(correlations)

        return 1.0 - avg_correlation

    def detect(self, video: Video) -> list[SceneDescription]:
        """Detect scenes in a video.

        Args:
            video: Video object to analyze

        Returns:
            List of SceneDescription objects representing detected scenes, ordered by time.
            Frame descriptions are not populated - use VideoAnalyzer for full analysis.
        """
        if len(video.frames) == 0:
            return []

        if len(video.frames) == 1:
            return [SceneDescription(start=0.0, end=video.total_seconds, start_frame=0, end_frame=1)]

        scene_boundaries = [0]

        for i in range(1, len(video.frames)):
            difference = self._calculate_histogram_difference(video.frames[i - 1], video.frames[i])

            if difference > self.threshold:
                scene_boundaries.append(i)

        scene_boundaries.append(len(video.frames))

        scenes = []
        for i in range(len(scene_boundaries) - 1):
            start_frame = scene_boundaries[i]
            end_frame = scene_boundaries[i + 1]

            start_time = start_frame / video.fps
            end_time = end_frame / video.fps

            scenes.append(
                SceneDescription(
                    start=start_time,
                    end=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

        if self.min_scene_length > 0:
            scenes = self._merge_short_scenes(scenes)

        return scenes

    def _merge_short_scenes(self, scenes: list[SceneDescription]) -> list[SceneDescription]:
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

            if last_scene.duration < self.min_scene_length:
                merged[-1] = SceneDescription(
                    start=last_scene.start,
                    end=scene.end,
                    start_frame=last_scene.start_frame,
                    end_frame=scene.end_frame,
                )
            else:
                merged.append(scene)

        if len(merged) > 1 and merged[-1].duration < self.min_scene_length:
            second_last = merged[-2]
            last = merged[-1]
            merged[-2] = SceneDescription(
                start=second_last.start,
                end=last.end,
                start_frame=second_last.start_frame,
                end_frame=last.end_frame,
            )
            merged.pop()

        return merged

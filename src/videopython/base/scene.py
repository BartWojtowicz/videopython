"""Scene detection using histogram comparison.

This module provides lightweight scene detection that uses only OpenCV for
histogram-based frame comparison, without requiring any ML dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from videopython.base.description import SceneBoundary

if TYPE_CHECKING:
    from videopython.base.video import Video


def _detect_segment_worker(
    segment: tuple[float, float],
    path: str,
    threshold: float,
) -> tuple[list[int], int, int]:
    """Worker function for parallel scene detection.

    Must be at module level for multiprocessing pickling.

    Args:
        segment: (start_second, end_second) tuple
        path: Path to video file
        threshold: Scene detection threshold

    Returns:
        Tuple of (scene_boundaries, first_frame_idx, last_frame_idx)
    """
    from videopython.base.video import FrameIterator

    start_second, end_second = segment

    scene_boundaries: list[int] = []
    prev_frame: np.ndarray | None = None
    first_frame_idx: int | None = None
    last_frame_idx: int = 0

    # Create temporary detector for histogram calculation
    detector = SceneDetector(threshold=threshold)

    with FrameIterator(path, start_second, end_second) as frames:
        for frame_idx, frame in frames:
            if first_frame_idx is None:
                first_frame_idx = frame_idx
                scene_boundaries.append(frame_idx)

            if prev_frame is not None:
                difference = detector._calculate_histogram_difference(prev_frame, frame)
                if difference > threshold:
                    scene_boundaries.append(frame_idx)

            prev_frame = frame
            last_frame_idx = frame_idx

    return (scene_boundaries, first_frame_idx or 0, last_frame_idx)


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

    def detect(self, video: Video) -> list[SceneBoundary]:
        """Detect scenes in a video.

        Args:
            video: Video object to analyze

        Returns:
            List of SceneBoundary objects representing detected scenes, ordered by time.
            Returns SceneBoundary objects with timing info only.
        """
        if len(video.frames) == 0:
            return []

        if len(video.frames) == 1:
            return [SceneBoundary(start=0.0, end=video.total_seconds, start_frame=0, end_frame=1)]

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
                SceneBoundary(
                    start=start_time,
                    end=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

        if self.min_scene_length > 0:
            scenes = self._merge_short_scenes(scenes)

        return scenes

    def detect_streaming(
        self,
        path: str | Path,
        start_second: float | None = None,
        end_second: float | None = None,
    ) -> list[SceneBoundary]:
        """Detect scenes by streaming frames from file.

        Memory usage is O(1) - only 2 frames in memory at any time.
        This is suitable for processing very long videos that would not
        fit in memory when loaded entirely.

        Args:
            path: Path to video file
            start_second: Optional start time for analysis
            end_second: Optional end time for analysis

        Returns:
            List of SceneBoundary objects representing detected scenes.

        Example:
            >>> detector = SceneDetector(threshold=0.3)
            >>> scenes = detector.detect_streaming("long_video.mp4")
            >>> for scene in scenes:
            ...     print(f"Scene: {scene.start:.2f}s - {scene.end:.2f}s")
        """
        from videopython.base.video import FrameIterator, VideoMetadata

        metadata = VideoMetadata.from_path(path)

        scene_boundaries: list[int] = []
        prev_frame: np.ndarray | None = None
        first_frame_idx: int | None = None
        last_frame_idx: int = 0

        with FrameIterator(path, start_second, end_second) as frames:
            for frame_idx, frame in frames:
                if first_frame_idx is None:
                    first_frame_idx = frame_idx
                    scene_boundaries.append(frame_idx)

                if prev_frame is not None:
                    difference = self._calculate_histogram_difference(prev_frame, frame)
                    if difference > self.threshold:
                        scene_boundaries.append(frame_idx)

                prev_frame = frame
                last_frame_idx = frame_idx

        # Handle empty video case
        if first_frame_idx is None:
            return []

        # Add end boundary (one past the last frame)
        scene_boundaries.append(last_frame_idx + 1)

        # Build SceneBoundary objects
        scenes = []
        for i in range(len(scene_boundaries) - 1):
            start_frame = scene_boundaries[i]
            end_frame = scene_boundaries[i + 1]

            start_time = start_frame / metadata.fps
            end_time = end_frame / metadata.fps

            scenes.append(
                SceneBoundary(
                    start=start_time,
                    end=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

        if self.min_scene_length > 0:
            scenes = self._merge_short_scenes(scenes)

        return scenes

    def detect_parallel(
        self,
        path: str | Path,
        num_workers: int | None = None,
        start_second: float | None = None,
        end_second: float | None = None,
    ) -> list[SceneBoundary]:
        """Detect scenes using parallel processing.

        Splits video into segments and processes them in parallel using
        multiple CPU cores. Most efficient for long videos on multi-core systems.

        Memory usage: O(workers * chunk_size) - each worker processes independently.

        Args:
            path: Path to video file
            num_workers: Number of parallel workers (default: CPU count)
            start_second: Optional start time for analysis
            end_second: Optional end time for analysis

        Returns:
            List of SceneBoundary objects representing detected scenes.

        Example:
            >>> detector = SceneDetector(threshold=0.3)
            >>> scenes = detector.detect_parallel("long_video.mp4", num_workers=4)
            >>> for scene in scenes:
            ...     print(f"Scene: {scene.start:.2f}s - {scene.end:.2f}s")
        """
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial

        from videopython.base.video import VideoMetadata

        metadata = VideoMetadata.from_path(path)

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        # Determine time range
        actual_start = start_second if start_second is not None else 0.0
        actual_end = end_second if end_second is not None else metadata.total_seconds

        total_duration = actual_end - actual_start

        # For short videos, just use streaming
        if total_duration < 10 or num_workers <= 1:
            return self.detect_streaming(path, start_second, end_second)

        # Split into segments for parallel processing
        segment_duration = total_duration / num_workers
        segments: list[tuple[float, float]] = []

        for i in range(num_workers):
            seg_start = actual_start + i * segment_duration
            seg_end = actual_start + (i + 1) * segment_duration
            if i == num_workers - 1:
                seg_end = actual_end  # Ensure last segment covers to end
            segments.append((seg_start, seg_end))

        # Process segments in parallel
        worker_func = partial(
            _detect_segment_worker,
            path=str(path),
            threshold=self.threshold,
        )

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(worker_func, segments))

        # Merge results from all segments
        all_boundaries: list[int] = []
        boundary_frames_at_edges: list[tuple[int, int]] = []  # (last_frame_of_seg, first_frame_of_next)

        for i, (boundaries, first_frame, last_frame) in enumerate(results):
            if i == 0:
                all_boundaries.extend(boundaries)
            else:
                # Check boundary between segments
                prev_last_frame = results[i - 1][2]
                # We need to check if there's a scene change at segment boundary
                # For now, include boundaries from this segment (excluding first which was start)
                if boundaries:
                    # First boundary of non-first segment is the segment start
                    # We may need to check for scene change here
                    all_boundaries.extend(boundaries[1:] if len(boundaries) > 1 else [])
                boundary_frames_at_edges.append((prev_last_frame, first_frame))

        # Handle empty case
        if not all_boundaries:
            return []

        # Add end boundary
        last_frame_idx = results[-1][2]
        all_boundaries.append(last_frame_idx + 1)

        # Deduplicate and sort
        all_boundaries = sorted(set(all_boundaries))

        # Build SceneBoundary objects
        scenes = []
        for i in range(len(all_boundaries) - 1):
            start_frame = all_boundaries[i]
            end_frame = all_boundaries[i + 1]

            start_time = start_frame / metadata.fps
            end_time = end_frame / metadata.fps

            scenes.append(
                SceneBoundary(
                    start=start_time,
                    end=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )

        if self.min_scene_length > 0:
            scenes = self._merge_short_scenes(scenes)

        return scenes

    @classmethod
    def detect_from_path(
        cls,
        path: str | Path,
        threshold: float = 0.3,
        min_scene_length: float = 0.5,
    ) -> list[SceneBoundary]:
        """Convenience method for one-shot streaming scene detection.

        Creates a SceneDetector instance and runs streaming detection.
        Memory usage is O(1), suitable for very long videos.

        Args:
            path: Path to video file
            threshold: Scene change threshold (0.0-1.0). Lower values
                      detect more scene changes. Default: 0.3
            min_scene_length: Minimum scene duration in seconds. Default: 0.5

        Returns:
            List of SceneBoundary objects representing detected scenes.

        Example:
            >>> scenes = SceneDetector.detect_from_path("video.mp4", threshold=0.3)
            >>> print(f"Found {len(scenes)} scenes")
        """
        detector = cls(threshold=threshold, min_scene_length=min_scene_length)
        return detector.detect_streaming(path)

    def _merge_short_scenes(self, scenes: list[SceneBoundary]) -> list[SceneBoundary]:
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
                merged[-1] = SceneBoundary(
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
            merged[-2] = SceneBoundary(
                start=second_last.start,
                end=last.end,
                start_frame=second_last.start_frame,
                end_frame=last.end_frame,
            )
            merged.pop()

        return merged

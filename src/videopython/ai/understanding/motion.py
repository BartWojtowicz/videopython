"""Motion analysis using optical flow.

This module provides motion detection and analysis capabilities using optical flow,
building on the existing CameraMotionDetector but adding motion magnitude scores
and scene-level aggregation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from videopython.base.description import MotionInfo

if TYPE_CHECKING:
    from videopython.base.video import Video

__all__ = ["MotionAnalyzer"]


class MotionAnalyzer:
    """Analyzes motion characteristics in video using optical flow.

    Detects both camera motion (pan, tilt, zoom) and overall motion magnitude,
    which is useful for identifying dynamic vs static scenes.

    Example:
        >>> from videopython.ai import MotionAnalyzer
        >>> from videopython.base import Video
        >>>
        >>> analyzer = MotionAnalyzer()
        >>> video = Video.from_path("video.mp4")
        >>>
        >>> # Analyze motion between two frames
        >>> motion = analyzer.analyze_frames(video.frames[0], video.frames[1])
        >>> print(f"Motion type: {motion.motion_type}, magnitude: {motion.magnitude:.2f}")
        >>>
        >>> # Analyze motion for a list of frames (returns list of MotionInfo)
        >>> motions = analyzer.analyze_frame_sequence(video.frames[:10])
    """

    MOTION_TYPES: list[str] = ["static", "pan", "tilt", "zoom", "complex"]

    def __init__(
        self,
        motion_threshold: float = 2.0,
        zoom_threshold: float = 0.1,
        magnitude_cap: float = 50.0,
    ):
        """Initialize motion analyzer.

        Args:
            motion_threshold: Minimum average flow magnitude to consider as motion.
                Values below this are classified as "static". Default: 2.0 pixels/frame.
            zoom_threshold: Threshold for detecting zoom based on flow pattern.
                Default: 0.1 (10% difference between center and edges).
            magnitude_cap: Cap for normalizing magnitude to 0-1 range.
                Motion above this value maps to 1.0. Default: 50.0 pixels/frame.
        """
        self.motion_threshold = motion_threshold
        self.zoom_threshold = zoom_threshold
        self.magnitude_cap = magnitude_cap

    def analyze_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
    ) -> MotionInfo:
        """Analyze motion between two consecutive frames.

        Args:
            frame1: First frame as numpy array (H, W, 3) RGB.
            frame2: Second frame as numpy array (H, W, 3) RGB.

        Returns:
            MotionInfo with motion type and magnitude.
        """
        import cv2

        # Convert to grayscale
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        else:
            gray1 = frame1

        if len(frame2.shape) == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        else:
            gray2 = frame2

        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        # Calculate magnitude
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        avg_magnitude = float(np.mean(magnitude))

        # Normalize magnitude to 0-1 range
        normalized_magnitude = min(avg_magnitude / self.magnitude_cap, 1.0)

        # Classify motion type
        if avg_magnitude < self.motion_threshold:
            motion_type = "static"
        else:
            motion_type = self._classify_motion(flow, gray1.shape, avg_magnitude)

        return MotionInfo(
            motion_type=motion_type,
            magnitude=normalized_magnitude,
            raw_magnitude=avg_magnitude,
        )

    def _classify_motion(
        self,
        flow: np.ndarray,
        shape: tuple[int, int],
        avg_magnitude: float,
    ) -> str:
        """Classify the type of motion based on optical flow pattern.

        Args:
            flow: Optical flow array (H, W, 2) with x and y components.
            shape: Frame shape (H, W).
            avg_magnitude: Average flow magnitude.

        Returns:
            Motion type: "pan", "tilt", "zoom", or "complex".
        """
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # Calculate mean flow direction
        mean_flow_x = np.mean(flow_x)
        mean_flow_y = np.mean(flow_y)

        # Check for zoom by analyzing flow from center
        h, w = shape
        cy, cx = h // 2, w // 2

        # Sample flow at different distances from center
        center_region = magnitude[cy - h // 4 : cy + h // 4, cx - w // 4 : cx + w // 4]
        edge_region_top = magnitude[: h // 4, :]
        edge_region_bottom = magnitude[-h // 4 :, :]
        edge_region_left = magnitude[:, : w // 4]
        edge_region_right = magnitude[:, -w // 4 :]

        center_mag = np.mean(center_region) if center_region.size > 0 else 0
        edge_mag = np.mean(
            [
                np.mean(edge_region_top) if edge_region_top.size > 0 else 0,
                np.mean(edge_region_bottom) if edge_region_bottom.size > 0 else 0,
                np.mean(edge_region_left) if edge_region_left.size > 0 else 0,
                np.mean(edge_region_right) if edge_region_right.size > 0 else 0,
            ]
        )

        # Zoom detection: edges move more than center (zoom in) or vice versa
        if edge_mag > 0 and abs(edge_mag - center_mag) / edge_mag > self.zoom_threshold:
            return "zoom"

        # Determine dominant motion direction
        abs_x = abs(mean_flow_x)
        abs_y = abs(mean_flow_y)

        if abs_x > abs_y * 1.5:
            return "pan"  # Horizontal motion
        elif abs_y > abs_x * 1.5:
            return "tilt"  # Vertical motion
        else:
            return "complex"  # Mixed motion

    def analyze_frame_sequence(
        self,
        frames: list[np.ndarray],
    ) -> list[MotionInfo]:
        """Analyze motion for a sequence of frames.

        Returns motion info for each pair of consecutive frames.
        Result list has length len(frames) - 1.

        Args:
            frames: List of frames as numpy arrays.

        Returns:
            List of MotionInfo objects for each frame transition.
        """
        if len(frames) < 2:
            return []

        motions = []
        for i in range(len(frames) - 1):
            motion = self.analyze_frames(frames[i], frames[i + 1])
            motions.append(motion)

        return motions

    def analyze_video(
        self,
        video: Video,
        sample_interval: int = 1,
    ) -> list[MotionInfo]:
        """Analyze motion throughout a video.

        Args:
            video: Video object to analyze.
            sample_interval: Analyze every Nth frame pair. Default: 1 (all frames).

        Returns:
            List of MotionInfo objects for sampled frame transitions.
        """
        frames = video.frames
        if len(frames) < 2:
            return []

        motions = []
        for i in range(0, len(frames) - 1, sample_interval):
            motion = self.analyze_frames(frames[i], frames[i + 1])
            motions.append(motion)

        return motions

    def analyze_video_path(
        self,
        path: str | Path,
        frames_per_second: float = 1.0,
    ) -> list[tuple[float, MotionInfo]]:
        """Analyze motion from video file with minimal memory usage.

        Streams frames from the video file instead of loading entire video.
        Returns timestamped motion info.

        Args:
            path: Path to video file.
            frames_per_second: How many frames per second to analyze. Default: 1.0.

        Returns:
            List of (timestamp, MotionInfo) tuples.
        """
        import cv2

        path = Path(path)
        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / frames_per_second))

        results: list[tuple[float, MotionInfo]] = []
        prev_frame = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if prev_frame is not None:
                    motion = self.analyze_frames(prev_frame, frame_rgb)
                    timestamp = frame_idx / fps
                    results.append((timestamp, motion))

                prev_frame = frame_rgb

            frame_idx += 1

        cap.release()
        return results

    @staticmethod
    def aggregate_motion(motions: list[MotionInfo]) -> tuple[float, str]:
        """Aggregate motion info into scene-level statistics.

        Args:
            motions: List of MotionInfo objects from frames in a scene.

        Returns:
            Tuple of (average_magnitude, dominant_motion_type).
        """
        if not motions:
            return 0.0, "static"

        avg_magnitude = sum(m.magnitude for m in motions) / len(motions)

        # Find dominant motion type (excluding static if there's any motion)
        motion_types = [m.motion_type for m in motions]
        type_counts = Counter(motion_types)

        # If mostly static, return static
        static_ratio = type_counts.get("static", 0) / len(motions)
        if static_ratio > 0.7:
            dominant_type = "static"
        else:
            # Find most common non-static type
            non_static = {k: v for k, v in type_counts.items() if k != "static"}
            if non_static:
                dominant_type = max(non_static, key=lambda k: non_static[k])
            else:
                dominant_type = "static"

        return avg_magnitude, dominant_type

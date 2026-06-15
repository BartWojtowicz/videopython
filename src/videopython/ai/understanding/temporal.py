"""Temporal understanding for video analysis.

Provides ML-based scene detection (TransNetV2). Action recognition was
removed in 0.29.0 -- it was never wired into ``VideoAnalyzer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from videopython.ai._device import log_device_initialization, release_device_memory, select_device
from videopython.base.description import SceneBoundary

if TYPE_CHECKING:
    from videopython.base.video import Video


class SemanticSceneDetector:
    """ML-based scene detection using TransNetV2.

    TransNetV2 is a neural network specifically designed for shot boundary
    detection, providing more accurate scene boundaries than histogram-based
    methods, especially for gradual transitions.

    Uses the transnetv2-pytorch package with pretrained weights.

    Example:
        >>> from videopython.ai.understanding import SemanticSceneDetector
        >>> detector = SemanticSceneDetector()
        >>> scenes = detector.detect_streaming("video.mp4")
        >>> for scene in scenes:
        ...     print(f"Scene: {scene.start:.2f}s - {scene.end:.2f}s")
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_scene_length: float = 0.5,
        device: str | None = None,
    ):
        """Initialize the semantic scene detector.

        Args:
            threshold: Confidence threshold for scene boundaries (0.0-1.0).
                Higher values = fewer, more confident boundaries.
            min_scene_length: Minimum scene duration in seconds.
            device: Device to run on ('cuda', 'mps', 'cpu', or None for auto).
                Note: MPS may have numerical inconsistencies; use 'cpu' for
                reproducible results.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        if min_scene_length < 0:
            raise ValueError("min_scene_length must be non-negative")

        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.device: str | None = device
        self._model: Any = None

    def _init_local(self) -> None:
        """Load the TransNetV2 model with pretrained weights."""
        if self._model is not None:
            return

        from videopython.ai._optional import require

        TransNetV2 = require("transnetv2_pytorch", "vision", feature="SemanticSceneDetector").TransNetV2

        requested_device = self.device
        device = select_device(self.device, mps_allowed=True)
        log_device_initialization(
            "SemanticSceneDetector",
            requested_device=requested_device,
            resolved_device=device,
        )
        self.device = device
        self._model = TransNetV2(device=device)
        self._model.eval()

    def unload(self) -> None:
        """Release the TransNetV2 model so the next call re-initializes."""
        self._model = None
        release_device_memory(self.device)

    def detect(self, video: Video) -> list[SceneBoundary]:
        """Detect scenes in a video using ML-based boundary detection.

        Note: This method requires saving video to a temporary file for
        TransNetV2 processing. For better performance, use detect_streaming()
        with a file path directly.

        Args:
            video: Video object to analyze.

        Returns:
            List of SceneBoundary objects representing detected scenes.
        """
        import tempfile

        if len(video.frames) == 0:
            return []

        if len(video.frames) == 1:
            return [SceneBoundary(start=0.0, end=video.total_seconds, start_frame=0, end_frame=1)]

        # Save video to temp file for TransNetV2 processing
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            video.save(tmp.name)
            return self.detect_streaming(tmp.name)

    def detect_streaming(self, path: str | Path) -> list[SceneBoundary]:
        """Detect scenes from a video file.

        Uses TransNetV2 with pretrained weights for accurate shot boundary
        detection.

        Args:
            path: Path to video file.

        Returns:
            List of SceneBoundary objects representing detected scenes.
        """
        self._init_local()

        # Use TransNetV2's detect_scenes which handles everything internally
        raw_scenes = self._model.detect_scenes(str(path), threshold=self.threshold)

        # Convert to SceneBoundary objects
        scenes = []
        for scene_data in raw_scenes:
            start_frame = scene_data["start_frame"]
            end_frame = scene_data["end_frame"]
            start_time = float(scene_data["start_time"])
            end_time = float(scene_data["end_time"])

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

    def _merge_short_scenes(self, scenes: list[SceneBoundary]) -> list[SceneBoundary]:
        """Merge scenes that are shorter than min_scene_length.

        Args:
            scenes: List of scenes to process.

        Returns:
            List of scenes with short scenes merged into adjacent ones.
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

    @classmethod
    def detect_from_path(
        cls,
        path: str | Path,
        threshold: float = 0.5,
        min_scene_length: float = 0.5,
    ) -> list[SceneBoundary]:
        """Convenience method for one-shot scene detection.

        Args:
            path: Path to video file.
            threshold: Scene boundary threshold (0.0-1.0).
            min_scene_length: Minimum scene duration in seconds.

        Returns:
            List of SceneBoundary objects representing detected scenes.
        """
        detector = cls(threshold=threshold, min_scene_length=min_scene_length)
        return detector.detect_streaming(path)

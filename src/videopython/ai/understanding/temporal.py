"""Temporal understanding for video analysis.

This module provides action/activity recognition and semantic scene detection
using deep learning models for temporal video understanding.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from videopython.base.description import DetectedAction, SceneBoundary

if TYPE_CHECKING:
    from videopython.base.video import Video


def _get_device(device: str | None) -> str:
    """Get the best available device for inference.

    Args:
        device: Explicit device choice, or None for auto-detection.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if device is not None:
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Kinetics-400 action labels (subset of common actions)
# Full list at: https://github.com/google-research/big_transfer/blob/master/bit_hyperrule.py
KINETICS_LABELS: list[str] | None = None


def _get_kinetics_labels() -> list[str]:
    """Lazy load Kinetics-400 labels from the model config."""
    global KINETICS_LABELS
    if KINETICS_LABELS is None:
        try:
            from transformers import AutoConfig  # type: ignore[attr-defined]

            config = AutoConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            KINETICS_LABELS = [config.id2label[i] for i in range(len(config.id2label))]
        except Exception:
            # Fallback to a minimal set if model not available
            KINETICS_LABELS = ["unknown"]
    return KINETICS_LABELS


class ActionRecognizer:
    """Recognizes actions/activities in video clips using VideoMAE.

    VideoMAE is a masked autoencoder pre-trained on video data and fine-tuned
    for action recognition on Kinetics-400 (400 action classes).

    Example:
        >>> from videopython.base import Video
        >>> from videopython.ai.understanding import ActionRecognizer
        >>> video = Video.from_path("video.mp4")
        >>> recognizer = ActionRecognizer()
        >>> actions = recognizer.recognize(video)
        >>> for action in actions:
        ...     print(f"{action.label}: {action.confidence:.2f}")
    """

    # Model variants available
    MODEL_VARIANTS = Literal["base", "large"]

    def __init__(
        self,
        model_size: MODEL_VARIANTS = "base",
        device: str | None = None,
        confidence_threshold: float = 0.1,
        num_frames: int = 16,
    ):
        """Initialize the action recognizer.

        Args:
            model_size: Model size - "base" (faster) or "large" (more accurate).
            device: Device to run on ('cuda', 'cpu', or None for auto).
            confidence_threshold: Minimum confidence for reported actions.
            num_frames: Number of frames to sample per clip (default 16 for VideoMAE).
        """
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.num_frames = num_frames

        # Lazy load model
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = device

    def _load_model(self) -> None:
        """Load the VideoMAE model and processor."""
        if self._model is not None:
            return

        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor  # type: ignore[attr-defined]

        model_name = (
            "MCG-NJU/videomae-base-finetuned-kinetics"
            if self.model_size == "base"
            else "MCG-NJU/videomae-large-finetuned-kinetics"
        )

        self._processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self._model = VideoMAEForVideoClassification.from_pretrained(model_name)

        self._device = _get_device(self._device)

        self._model = self._model.to(self._device)
        self._model.eval()

    def _sample_frames(self, frames: np.ndarray, num_samples: int) -> np.ndarray:
        """Sample frames uniformly from a video clip.

        Args:
            frames: Video frames array (N, H, W, 3)
            num_samples: Number of frames to sample

        Returns:
            Sampled frames array (num_samples, H, W, 3)
        """
        total_frames = len(frames)
        if total_frames <= num_samples:
            # Pad by repeating last frame if needed
            if total_frames < num_samples:
                pad_count = num_samples - total_frames
                padding = np.repeat(frames[-1:], pad_count, axis=0)
                return np.concatenate([frames, padding], axis=0)
            return frames

        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        return frames[indices]

    def recognize(
        self,
        video: Video,
        top_k: int = 5,
    ) -> list[DetectedAction]:
        """Recognize actions in a video.

        Processes the entire video as a single clip and returns top-k predictions.

        Args:
            video: Video object to analyze.
            top_k: Number of top predictions to return.

        Returns:
            List of DetectedAction objects with recognized activities.
        """
        self._load_model()

        import torch

        # Sample frames for the model
        sampled_frames = self._sample_frames(video.frames, self.num_frames)

        # Convert to list of PIL images (processor expects this format)
        frames_list = [sampled_frames[i] for i in range(len(sampled_frames))]

        # Process frames
        inputs = self._processor(frames_list, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

        # Build results
        labels = _get_kinetics_labels()
        actions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            if prob >= self.confidence_threshold:
                actions.append(
                    DetectedAction(
                        label=labels[idx],
                        confidence=float(prob),
                        start_frame=0,
                        end_frame=len(video.frames),
                        start_time=0.0,
                        end_time=video.total_seconds,
                    )
                )

        return actions

    def recognize_path(
        self,
        path: str | Path,
        top_k: int = 5,
        start_second: float | None = None,
        end_second: float | None = None,
    ) -> list[DetectedAction]:
        """Recognize actions from a video file with memory-efficient loading.

        Args:
            path: Path to video file.
            top_k: Number of top predictions to return.
            start_second: Optional start time for analysis.
            end_second: Optional end time for analysis.

        Returns:
            List of DetectedAction objects with recognized activities.
        """
        from videopython.base.video import VideoMetadata, extract_frames_at_times

        self._load_model()

        import torch

        metadata = VideoMetadata.from_path(path)

        # Determine time range
        start = start_second if start_second is not None else 0.0
        end = end_second if end_second is not None else metadata.total_seconds

        # Sample timestamps uniformly
        timestamps = np.linspace(start, end - 0.001, self.num_frames).tolist()
        frames = extract_frames_at_times(path, timestamps)

        if len(frames) < self.num_frames:
            # Pad if needed
            frames = self._sample_frames(frames, self.num_frames)

        # Convert to list for processor
        frames_list = [frames[i] for i in range(len(frames))]

        # Process and run inference
        inputs = self._processor(frames_list, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

        # Build results
        labels = _get_kinetics_labels()
        actions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            if prob >= self.confidence_threshold:
                start_frame = int(start * metadata.fps)
                end_frame = int(end * metadata.fps)
                actions.append(
                    DetectedAction(
                        label=labels[idx],
                        confidence=float(prob),
                        start_frame=start_frame,
                        end_frame=end_frame,
                        start_time=start,
                        end_time=end,
                    )
                )

        return actions


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
        self._device: str | None = device
        self._model: Any = None

    def _load_model(self) -> None:
        """Load the TransNetV2 model with pretrained weights."""
        if self._model is not None:
            return

        from transnetv2_pytorch import TransNetV2

        # Use 'auto' for transnetv2-pytorch's device selection, or explicit device
        device = self._device if self._device is not None else "auto"
        self._model = TransNetV2(device=device)
        self._model.eval()

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

    def detect_streaming(
        self,
        path: str | Path,
        start_second: float | None = None,
        end_second: float | None = None,
    ) -> list[SceneBoundary]:
        """Detect scenes from a video file.

        Uses TransNetV2 with pretrained weights for accurate shot boundary
        detection.

        Args:
            path: Path to video file.
            start_second: Optional start time for analysis (not yet supported).
            end_second: Optional end time for analysis (not yet supported).

        Returns:
            List of SceneBoundary objects representing detected scenes.
        """
        if start_second is not None or end_second is not None:
            import warnings

            warnings.warn(
                "start_second and end_second are not yet supported by SemanticSceneDetector. Processing entire video.",
                UserWarning,
                stacklevel=2,
            )

        self._load_model()

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

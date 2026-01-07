from __future__ import annotations

from typing import Literal

import numpy as np

from videopython.ai.backends import ImageToTextBackend
from videopython.ai.understanding.color import ColorAnalyzer
from videopython.ai.understanding.image import ImageToText
from videopython.base.description import FrameDescription, SceneDescription, VideoDescription
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
            # Single frame video is one scene
            return [SceneDescription(start=0.0, end=video.total_seconds, start_frame=0, end_frame=1)]

        # Calculate frame differences
        scene_boundaries = [0]  # First frame is always a scene start

        for i in range(1, len(video.frames)):
            difference = self._calculate_histogram_difference(video.frames[i - 1], video.frames[i])

            if difference > self.threshold:
                scene_boundaries.append(i)

        # Last frame index (exclusive)
        scene_boundaries.append(len(video.frames))

        # Create SceneDescription objects
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

        # Merge scenes that are too short
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

            # If the last scene is too short, merge it with current scene
            if last_scene.duration < self.min_scene_length:
                # Merge by extending the previous scene to include this one
                merged[-1] = SceneDescription(
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
            merged[-2] = SceneDescription(
                start=second_last.start,
                end=last.end,
                start_frame=second_last.start_frame,
                end_frame=last.end_frame,
            )
            merged.pop()

        return merged


class FrameAnalyzer:
    """Analyzes individual frames for objects, faces, text, and other features.

    For cloud backends (openai/gemini), uses a single API call via CombinedFrameAnalyzer
    to reduce costs and latency. For local backend, uses individual detectors.
    """

    def __init__(
        self,
        backend: ImageToTextBackend | None = None,
        api_key: str | None = None,
        object_detection: bool = True,
        face_detection: bool = True,
        text_detection: bool = False,
        shot_type_detection: bool = False,
        yolo_model_size: str = "n",
        ocr_languages: list[str] | None = None,
    ):
        """Initialize the frame analyzer.

        Args:
            backend: Backend for detection ('local', 'openai', 'gemini').
            api_key: API key for cloud backends.
            object_detection: Whether to detect objects in frames.
            face_detection: Whether to detect faces in frames.
            text_detection: Whether to detect text (OCR) in frames.
            shot_type_detection: Whether to classify shot type (cloud only).
            yolo_model_size: YOLO model size for local object detection ('n', 's', 'm', 'l', 'x').
            ocr_languages: Languages for OCR (default: ['en']).
        """
        from videopython.ai.config import get_default_backend

        self.backend: ImageToTextBackend = (
            backend if backend is not None else get_default_backend("image_to_text")  # type: ignore[assignment]
        )
        self.api_key = api_key
        self.object_detection = object_detection
        self.face_detection = face_detection
        self.text_detection = text_detection
        self.shot_type_detection = shot_type_detection

        # For cloud backends, use combined analyzer for efficiency
        self._combined_analyzer = None
        self._object_detector = None
        self._face_detector = None
        self._text_detector = None

        if self.backend in ("openai", "gemini"):
            # Use single API call for all detections
            from videopython.ai.understanding.detection import CombinedFrameAnalyzer

            self._combined_analyzer = CombinedFrameAnalyzer(backend=self.backend, api_key=api_key)
        else:
            # Use individual local detectors
            if object_detection:
                from videopython.ai.understanding.detection import ObjectDetector

                self._object_detector = ObjectDetector(
                    backend="local",
                    model_size=yolo_model_size,
                    api_key=api_key,
                )

            if face_detection:
                from videopython.ai.understanding.detection import FaceDetector

                self._face_detector = FaceDetector()

            if text_detection:
                from videopython.ai.understanding.detection import TextDetector

                self._text_detector = TextDetector(
                    backend="local",
                    languages=ocr_languages or ["en"],
                    api_key=api_key,
                )

    async def analyze_frame(
        self,
        frame: np.ndarray,
        frame_description: FrameDescription,
    ) -> FrameDescription:
        """Analyze a frame and populate detection fields.

        Args:
            frame: Frame image as numpy array (H, W, 3) in RGB format.
            frame_description: FrameDescription to populate with detection results.

        Returns:
            Updated FrameDescription with detection results.
        """
        if self._combined_analyzer:
            # Cloud backend: single API call for all detections
            result = await self._combined_analyzer.analyze(frame)
            if self.object_detection:
                frame_description.detected_objects = result.detected_objects
            if self.face_detection:
                frame_description.detected_faces = result.face_count
            if self.text_detection:
                frame_description.detected_text = result.detected_text
            if self.shot_type_detection:
                frame_description.shot_type = result.shot_type
        else:
            # Local backend: individual detectors
            if self._object_detector:
                frame_description.detected_objects = await self._object_detector.detect(frame)

            if self._face_detector:
                frame_description.detected_faces = await self._face_detector.detect(frame)

            if self._text_detector:
                frame_description.detected_text = await self._text_detector.detect(frame)

        return frame_description


def _aggregate_detected_entities(frame_descriptions: list[FrameDescription]) -> list[str]:
    """Aggregate unique detected entities from frame descriptions.

    Args:
        frame_descriptions: List of FrameDescription objects with detected_objects.

    Returns:
        Sorted list of unique entity labels.
    """
    entities = set()
    for fd in frame_descriptions:
        if fd.detected_objects:
            for obj in fd.detected_objects:
                entities.add(obj.label)
    return sorted(entities)


def _aggregate_dominant_colors(
    frame_descriptions: list[FrameDescription],
    num_colors: int = 5,
) -> list[tuple[int, int, int]]:
    """Aggregate dominant colors from frame-level color histograms.

    Args:
        frame_descriptions: List of FrameDescription objects with color_histogram.
        num_colors: Number of dominant colors to return.

    Returns:
        List of dominant RGB color tuples.
    """
    from collections import Counter

    all_colors: list[tuple[int, int, int]] = []
    for fd in frame_descriptions:
        if fd.color_histogram and fd.color_histogram.dominant_colors:
            all_colors.extend(fd.color_histogram.dominant_colors)

    if not all_colors:
        return []

    # Count color occurrences and return most common
    color_counts = Counter(all_colors)
    return [color for color, _ in color_counts.most_common(num_colors)]


class VideoAnalyzer:
    """Comprehensive video analysis combining scene detection, frame understanding, and transcription."""

    def __init__(
        self,
        scene_threshold: float = 0.3,
        min_scene_length: float = 0.5,
        device: str | None = None,
        detection_backend: ImageToTextBackend | None = None,
        api_key: str | None = None,
    ):
        """Initialize the video analyzer.

        Args:
            scene_threshold: Threshold for scene change detection (0.0-1.0)
            min_scene_length: Minimum scene duration in seconds
            device: Device for ImageToText model ('cuda', 'cpu', or None for auto)
            detection_backend: Backend for object/text detection ('local', 'openai', 'gemini')
            api_key: API key for cloud backends
        """
        self.scene_detector = SceneDetector(threshold=scene_threshold, min_scene_length=min_scene_length)
        self.image_to_text = ImageToText(device=device)
        self.detection_backend = detection_backend
        self.api_key = api_key

    async def analyze(
        self,
        video: Video,
        frames_per_second: float = 1.0,
        transcribe: bool = False,
        transcription_model: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "base",
        description_prompt: str | None = None,
        extract_colors: bool = False,
        include_full_histogram: bool = False,
        detect_objects: bool = False,
        detect_faces: bool = False,
        detect_text: bool = False,
        detect_shot_type: bool = False,
        generate_summaries: bool = False,
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
            detect_objects: Whether to detect objects in frames (default: False)
            detect_faces: Whether to detect faces in frames (default: False)
            detect_text: Whether to detect text (OCR) in frames (default: False)
            detect_shot_type: Whether to classify shot type (cloud backends only) (default: False)
            generate_summaries: Whether to generate LLM summaries for scenes (default: False)

        Returns:
            VideoDescription object with complete analysis
        """
        # Step 1: Detect scenes (returns SceneDescription objects with timing only)
        scene_descriptions = self.scene_detector.detect(video)

        # Step 2: Set up frame analyzer if any detection is enabled
        frame_analyzer = None
        if detect_objects or detect_faces or detect_text or detect_shot_type:
            frame_analyzer = FrameAnalyzer(
                backend=self.detection_backend,
                api_key=self.api_key,
                object_detection=detect_objects,
                face_detection=detect_faces,
                text_detection=detect_text,
                shot_type_detection=detect_shot_type,
            )

        # Step 3: Analyze frames from each scene and populate frame_descriptions
        for scene_desc in scene_descriptions:
            frame_descriptions = await self.image_to_text.describe_scene(
                video,
                scene_desc,
                frames_per_second=frames_per_second,
                prompt=description_prompt,
                extract_colors=extract_colors,
                include_full_histogram=include_full_histogram,
            )

            # Run detection on each frame if enabled
            if frame_analyzer:
                for fd in frame_descriptions:
                    frame = video.frames[fd.frame_index]
                    await frame_analyzer.analyze_frame(frame, fd)

            scene_desc.frame_descriptions = frame_descriptions

            # Populate scene-level aggregations
            if detect_objects:
                scene_desc.detected_entities = _aggregate_detected_entities(frame_descriptions)

            if extract_colors:
                scene_desc.dominant_colors = _aggregate_dominant_colors(frame_descriptions)

        # Step 4: Optional transcription
        transcription = None
        if transcribe:
            from videopython.ai.understanding.audio import AudioToText

            transcriber = AudioToText(model_name=transcription_model)
            transcription = await transcriber.transcribe(video)

        # Create VideoDescription and distribute transcription to scenes
        video_description = VideoDescription(scene_descriptions=scene_descriptions, transcription=transcription)
        if transcription:
            video_description.distribute_transcription()

        # Step 5: Generate summaries if requested
        if generate_summaries:
            from videopython.ai.understanding.text import LLMSummarizer

            summarizer = LLMSummarizer(backend=self.detection_backend, api_key=self.api_key)
            for scene_desc in scene_descriptions:
                scene_desc.summary = await summarizer.summarize_scene_description(scene_desc)

        return video_description

    async def analyze_scenes_only(self, video: Video) -> list[SceneDescription]:
        """Analyze video scenes without transcription (convenience method).

        Args:
            video: Video object to analyze

        Returns:
            List of SceneDescription objects
        """
        understanding = await self.analyze(video, transcribe=False)
        return understanding.scene_descriptions

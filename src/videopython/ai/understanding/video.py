from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from videopython.ai.backends import ImageToTextBackend
from videopython.ai.understanding.image import ImageToText
from videopython.base.description import (
    AudioClassification,
    DetectedFace,
    FrameDescription,
    SceneDescription,
    VideoDescription,
)
from videopython.base.scene import SceneDetector
from videopython.base.video import Video, VideoMetadata, extract_frames_at_indices

if TYPE_CHECKING:
    from videopython.ai.understanding.temporal import SemanticSceneDetector

# Sampling strategy type
SamplingStrategy = Literal["fixed", "adaptive"]


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

    def analyze_frame(
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
            result = self._combined_analyzer.analyze(frame)
            if self.object_detection:
                frame_description.detected_objects = result.detected_objects
            if self.face_detection:
                # Cloud backends return count only, create DetectedFace objects without bounding boxes
                frame_description.detected_faces = [DetectedFace() for _ in range(result.face_count)]
            if self.text_detection:
                frame_description.detected_text = result.detected_text
            if self.shot_type_detection:
                frame_description.shot_type = result.shot_type
        else:
            # Local backend: individual detectors
            if self._object_detector:
                frame_description.detected_objects = self._object_detector.detect(frame)

            if self._face_detector:
                frame_description.detected_faces = self._face_detector.detect(frame)

            if self._text_detector:
                frame_description.detected_text = self._text_detector.detect(frame)

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


def _distribute_audio_events(
    scene_descriptions: list[SceneDescription],
    audio_classification: AudioClassification,
) -> None:
    """Distribute audio events to scenes based on timestamps.

    Args:
        scene_descriptions: List of SceneDescription objects.
        audio_classification: AudioClassification with detected events.
    """
    for scene_desc in scene_descriptions:
        scene_events = [
            event
            for event in audio_classification.events
            if event.start < scene_desc.end and event.end > scene_desc.start
        ]
        scene_desc.audio_events = scene_events if scene_events else None


def _aggregate_motion(frame_descriptions: list[FrameDescription]) -> tuple[float | None, str | None]:
    """Aggregate motion info from frame descriptions into scene-level stats.

    Args:
        frame_descriptions: List of FrameDescription objects with motion info.

    Returns:
        Tuple of (avg_motion_magnitude, dominant_motion_type), or (None, None) if no motion data.
    """
    from videopython.ai.understanding.motion import MotionAnalyzer

    motions = [fd.motion for fd in frame_descriptions if fd.motion is not None]
    if not motions:
        return None, None

    avg_magnitude, dominant_type = MotionAnalyzer.aggregate_motion(motions)
    return avg_magnitude, dominant_type


def _get_frame_indices_for_scene(
    scene: SceneDescription,
    fps: float,
    strategy: SamplingStrategy,
    frames_per_second: float,
) -> list[int]:
    """Get frame indices to sample for a scene based on sampling strategy.

    Args:
        scene: SceneDescription with start/end frame info.
        fps: Video frames per second.
        strategy: Sampling strategy ('fixed' or 'adaptive').
        frames_per_second: Frame sampling rate (used for 'fixed' strategy).

    Returns:
        List of frame indices to sample.
    """
    start_frame = scene.start_frame
    end_frame = scene.end_frame - 1  # last frame of scene

    if end_frame <= start_frame:
        return [start_frame]

    if strategy == "fixed":
        # Original behavior: sample at fixed fps rate
        frame_interval = max(1, int(fps / frames_per_second))
        indices = list(range(start_frame, scene.end_frame, frame_interval))
        return indices if indices else [start_frame]

    # Adaptive strategy: start + ln(1 + duration) + end
    # For short scenes (<=2s): 1 fps sampling
    if scene.duration <= 2:
        n_frames = max(1, int(scene.duration))
        if n_frames == 1:
            return [(start_frame + end_frame) // 2]
        # Evenly distribute frames
        indices = []
        for i in range(n_frames):
            t = (i + 0.5) / n_frames
            idx = int(start_frame + t * (end_frame - start_frame))
            indices.append(idx)
        return indices

    # For longer scenes: start + floor(ln(1 + duration)) middle frames + end
    n_middle = int(math.log(1 + scene.duration))

    if n_middle == 0:
        return [start_frame, end_frame]

    middle_frames = []
    for i in range(1, n_middle + 1):
        t = i / (n_middle + 1)
        frame_idx = int(start_frame + t * (end_frame - start_frame))
        middle_frames.append(frame_idx)

    return [start_frame] + middle_frames + [end_frame]


class VideoAnalyzer:
    """Comprehensive video analysis combining scene detection, frame understanding, and transcription."""

    def __init__(
        self,
        scene_threshold: float = 0.3,
        min_scene_length: float = 0.5,
        device: str | None = None,
        detection_backend: ImageToTextBackend | None = None,
        api_key: str | None = None,
        use_semantic_scenes: bool = False,
        sampling_strategy: SamplingStrategy = "fixed",
    ):
        """Initialize the video analyzer.

        Args:
            scene_threshold: Threshold for scene change detection (0.0-1.0)
            min_scene_length: Minimum scene duration in seconds
            device: Device for ImageToText model ('cuda', 'cpu', or None for auto)
            detection_backend: Backend for object/text detection ('local', 'openai', 'gemini')
            api_key: API key for cloud backends
            use_semantic_scenes: Use ML-based scene detection (SemanticSceneDetector)
                instead of histogram-based detection. More accurate but requires
                additional dependencies.
            sampling_strategy: Frame sampling strategy for scene analysis:
                - 'fixed': Sample at fixed fps rate (default, uses frames_per_second param)
                - 'adaptive': Smart sampling using start + ln(1+duration) + end formula.
                  For scenes <=2s uses 1fps, for longer scenes samples logarithmically.
        """
        self.scene_threshold = scene_threshold
        self.min_scene_length = min_scene_length
        self.use_semantic_scenes = use_semantic_scenes
        self.sampling_strategy: SamplingStrategy = sampling_strategy

        self.scene_detector: SceneDetector | SemanticSceneDetector
        if use_semantic_scenes:
            from videopython.ai.understanding.temporal import SemanticSceneDetector

            self.scene_detector = SemanticSceneDetector(
                threshold=scene_threshold, min_scene_length=min_scene_length, device=device
            )
        else:
            self.scene_detector = SceneDetector(threshold=scene_threshold, min_scene_length=min_scene_length)

        self.image_to_text = ImageToText(device=device)
        self.detection_backend = detection_backend
        self.api_key = api_key
        self._device = device

    def analyze(
        self,
        video: Video,
        frames_per_second: float = 1.0,
        transcribe: bool = False,
        transcription_model: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "base",
        description_prompt: str | None = None,
        extract_colors: bool = False,
        detect_objects: bool = False,
        detect_faces: bool = False,
        detect_text: bool = False,
        detect_shot_type: bool = False,
        generate_summaries: bool = False,
        classify_audio: bool = False,
        audio_classifier_threshold: float = 0.3,
        analyze_motion: bool = False,
        recognize_actions: bool = False,
        action_confidence_threshold: float = 0.1,
    ) -> VideoDescription:
        """Perform comprehensive video analysis.

        Args:
            video: Video object to analyze
            frames_per_second: Frame sampling rate for visual analysis (default: 1.0 fps)
            transcribe: Whether to generate audio transcription (default: False)
            transcription_model: Whisper model to use if transcribe=True (default: "base")
            description_prompt: Optional prompt to guide frame descriptions
            extract_colors: Whether to extract color features from frames (default: False)
            detect_objects: Whether to detect objects in frames (default: False)
            detect_faces: Whether to detect faces in frames (default: False)
            detect_text: Whether to detect text (OCR) in frames (default: False)
            detect_shot_type: Whether to classify shot type (cloud backends only) (default: False)
            generate_summaries: Whether to generate LLM summaries for scenes (default: False)
            classify_audio: Whether to classify audio events/sounds (default: False)
            audio_classifier_threshold: Minimum confidence for audio events (default: 0.3)
            analyze_motion: Whether to analyze motion between frames (default: False)
            recognize_actions: Whether to recognize actions/activities in scenes (default: False)
            action_confidence_threshold: Minimum confidence for action recognition (default: 0.1)

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

        # Set up motion analyzer if enabled
        motion_analyzer = None
        if analyze_motion:
            from videopython.ai.understanding.motion import MotionAnalyzer

            motion_analyzer = MotionAnalyzer()

        # Step 3: Analyze frames from each scene and populate frame_descriptions
        for scene_desc in scene_descriptions:
            # Calculate which frames to sample based on strategy
            frame_indices = _get_frame_indices_for_scene(
                scene_desc, video.fps, self.sampling_strategy, frames_per_second
            )

            # Generate frame descriptions
            frame_descriptions = []
            for frame_idx in frame_indices:
                if frame_idx >= len(video.frames):
                    continue

                frame = video.frames[frame_idx]
                description = self.image_to_text.describe_image(frame, description_prompt)
                timestamp = frame_idx / video.fps

                color_histogram = None
                if extract_colors:
                    color_histogram = self.image_to_text.color_analyzer.extract_color_features(frame)

                fd = FrameDescription(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    description=description,
                    color_histogram=color_histogram,
                )

                # Run detection on frame if enabled
                if frame_analyzer:
                    frame_analyzer.analyze_frame(frame, fd)

                frame_descriptions.append(fd)

            # Run motion analysis between consecutive frames
            if motion_analyzer and len(frame_descriptions) >= 2:
                for i in range(len(frame_descriptions) - 1):
                    fd_curr = frame_descriptions[i]
                    fd_next = frame_descriptions[i + 1]
                    frame_curr = video.frames[fd_curr.frame_index]
                    frame_next = video.frames[fd_next.frame_index]
                    motion_info = motion_analyzer.analyze_frames(frame_curr, frame_next)
                    # Assign motion to the current frame (motion leading into next frame)
                    fd_curr.motion = motion_info

            scene_desc.frame_descriptions = frame_descriptions

            # Populate scene-level aggregations
            if detect_objects:
                scene_desc.detected_entities = _aggregate_detected_entities(frame_descriptions)

            if extract_colors:
                scene_desc.dominant_colors = _aggregate_dominant_colors(frame_descriptions)

            if analyze_motion:
                avg_mag, dom_type = _aggregate_motion(frame_descriptions)
                scene_desc.avg_motion_magnitude = avg_mag
                scene_desc.dominant_motion_type = dom_type

        # Step 4: Optional transcription
        transcription = None
        if transcribe:
            from videopython.ai.understanding.audio import AudioToText

            transcriber = AudioToText(model_name=transcription_model)
            transcription = transcriber.transcribe(video)

        # Create VideoDescription and distribute transcription to scenes
        video_description = VideoDescription(scene_descriptions=scene_descriptions, transcription=transcription)
        if transcription:
            video_description.distribute_transcription()

        # Step 5: Optional audio classification
        if classify_audio:
            from videopython.ai.understanding.audio import AudioClassifier

            classifier = AudioClassifier(confidence_threshold=audio_classifier_threshold)
            audio_classification = classifier.classify(video)
            _distribute_audio_events(scene_descriptions, audio_classification)

        # Step 6: Generate summaries if requested
        if generate_summaries:
            from videopython.ai.understanding.text import LLMSummarizer

            summarizer = LLMSummarizer(backend=self.detection_backend, api_key=self.api_key)
            for scene_desc in scene_descriptions:
                scene_desc.summary = summarizer.summarize_scene_description(scene_desc)

        # Step 7: Recognize actions in each scene if requested
        if recognize_actions:
            from videopython.ai.understanding.temporal import ActionRecognizer

            action_recognizer = ActionRecognizer(
                device=self._device,
                confidence_threshold=action_confidence_threshold,
            )
            action_recognizer.recognize_scenes(video, scene_descriptions)

        return video_description

    def analyze_scenes_only(self, video: Video) -> list[SceneDescription]:
        """Analyze video scenes without transcription (convenience method).

        Args:
            video: Video object to analyze

        Returns:
            List of SceneDescription objects
        """
        understanding = self.analyze(video, transcribe=False)
        return understanding.scene_descriptions

    def analyze_path(
        self,
        path: str | Path,
        frames_per_second: float = 1.0,
        transcribe: bool = False,
        transcription_model: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "base",
        description_prompt: str | None = None,
        extract_colors: bool = False,
        detect_objects: bool = False,
        detect_faces: bool = False,
        detect_text: bool = False,
        detect_shot_type: bool = False,
        generate_summaries: bool = False,
        classify_audio: bool = False,
        audio_classifier_threshold: float = 0.3,
        analyze_motion: bool = False,
        recognize_actions: bool = False,
        action_confidence_threshold: float = 0.1,
    ) -> VideoDescription:
        """Analyze video from path with minimal memory usage.

        Unlike analyze(), this method:
        1. Uses streaming scene detection (O(1) memory for scene detection)
        2. Only loads frames needed for analysis (sampled frames per scene)

        Memory usage: O(sampled_frames_per_scene) instead of O(total_frames).
        This makes it possible to analyze very long videos that would not
        fit in memory when loaded entirely.

        Args:
            path: Path to video file
            frames_per_second: Frame sampling rate for visual analysis (default: 1.0 fps)
            transcribe: Whether to generate audio transcription (default: False)
            transcription_model: Whisper model to use if transcribe=True (default: "base")
            description_prompt: Optional prompt to guide frame descriptions
            extract_colors: Whether to extract color features from frames (default: False)
            detect_objects: Whether to detect objects in frames (default: False)
            detect_faces: Whether to detect faces in frames (default: False)
            detect_text: Whether to detect text (OCR) in frames (default: False)
            detect_shot_type: Whether to classify shot type (cloud backends only) (default: False)
            generate_summaries: Whether to generate LLM summaries for scenes (default: False)
            classify_audio: Whether to classify audio events/sounds (default: False)
            audio_classifier_threshold: Minimum confidence for audio events (default: 0.3)
            analyze_motion: Whether to analyze motion between frames (default: False)
            recognize_actions: Whether to recognize actions/activities in scenes (default: False)
            action_confidence_threshold: Minimum confidence for action recognition (default: 0.1)

        Returns:
            VideoDescription object with complete analysis

        Example:
            >>> analyzer = VideoAnalyzer()
            >>> # For 20-minute video at 0.2 fps, loads ~240 frames instead of ~28,800
            >>> result = analyzer.analyze_path("long_video.mp4", frames_per_second=0.2)
        """
        path = Path(path)
        metadata = VideoMetadata.from_path(path)

        # Step 1: Stream scene detection (O(1) memory)
        scene_descriptions = self.scene_detector.detect_streaming(path)

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

        # Set up motion analyzer if enabled
        motion_analyzer = None
        if analyze_motion:
            from videopython.ai.understanding.motion import MotionAnalyzer

            motion_analyzer = MotionAnalyzer()

        # Step 3: Process each scene, loading only sampled frames
        for scene_desc in scene_descriptions:
            # Calculate which frames to sample based on strategy
            frame_indices = _get_frame_indices_for_scene(
                scene_desc, metadata.fps, self.sampling_strategy, frames_per_second
            )

            # Extract only needed frames from file
            scene_frames = extract_frames_at_indices(path, frame_indices)

            # Generate frame descriptions
            frame_descriptions = []
            for i, frame_idx in enumerate(frame_indices):
                if i >= len(scene_frames):
                    break

                frame = scene_frames[i]

                # Get description using ImageToText
                description = self.image_to_text.describe_image(frame, description_prompt)
                timestamp = frame_idx / metadata.fps

                # Color features if requested
                color_histogram = None
                if extract_colors:
                    color_histogram = self.image_to_text.color_analyzer.extract_color_features(frame)

                fd = FrameDescription(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    description=description,
                    color_histogram=color_histogram,
                )

                # Detection if enabled
                if frame_analyzer:
                    frame_analyzer.analyze_frame(frame, fd)

                frame_descriptions.append(fd)

            # Run motion analysis between consecutive frames
            if motion_analyzer and len(scene_frames) >= 2:
                for i in range(len(frame_descriptions) - 1):
                    if i < len(scene_frames) - 1:
                        motion_info = motion_analyzer.analyze_frames(scene_frames[i], scene_frames[i + 1])
                        frame_descriptions[i].motion = motion_info

            scene_desc.frame_descriptions = frame_descriptions

            # Aggregate scene-level data
            if detect_objects:
                scene_desc.detected_entities = _aggregate_detected_entities(frame_descriptions)

            if extract_colors:
                scene_desc.dominant_colors = _aggregate_dominant_colors(frame_descriptions)

            if analyze_motion:
                avg_mag, dom_type = _aggregate_motion(frame_descriptions)
                scene_desc.avg_motion_magnitude = avg_mag
                scene_desc.dominant_motion_type = dom_type

        # Step 4: Optional transcription (audio-only, already memory efficient)
        transcription = None
        if transcribe:
            from videopython.ai.understanding.audio import AudioToText
            from videopython.base.audio import Audio

            audio = Audio.from_path(str(path))
            transcriber = AudioToText(model_name=transcription_model)
            transcription = transcriber.transcribe(audio)

        # Create VideoDescription and distribute transcription to scenes
        video_description = VideoDescription(
            scene_descriptions=scene_descriptions,
            transcription=transcription,
        )
        if transcription:
            video_description.distribute_transcription()

        # Step 5: Optional audio classification
        if classify_audio:
            from videopython.ai.understanding.audio import AudioClassifier
            from videopython.base.audio import Audio

            audio = Audio.from_path(str(path))
            classifier = AudioClassifier(confidence_threshold=audio_classifier_threshold)
            audio_classification = classifier.classify(audio)
            _distribute_audio_events(scene_descriptions, audio_classification)

        # Step 6: Generate summaries if requested
        if generate_summaries:
            from videopython.ai.understanding.text import LLMSummarizer

            summarizer = LLMSummarizer(backend=self.detection_backend, api_key=self.api_key)
            for scene_desc in scene_descriptions:
                scene_desc.summary = summarizer.summarize_scene_description(scene_desc)

        # Step 7: Recognize actions in each scene if requested
        if recognize_actions:
            from videopython.ai.understanding.temporal import ActionRecognizer

            action_recognizer = ActionRecognizer(
                device=self._device,
                confidence_threshold=action_confidence_threshold,
            )
            # For path-based analysis, process each scene by loading its frames
            for scene_desc in scene_descriptions:
                actions = action_recognizer.recognize_path(
                    path,
                    start_second=scene_desc.start,
                    end_second=scene_desc.end,
                    top_k=3,
                )
                scene_desc.detected_actions = actions

        return video_description

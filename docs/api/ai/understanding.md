# AI Understanding

Analyze videos, transcribe audio, and describe visual content.

For a single aggregate, serializable analysis object across multiple analyzers, see [Video Analysis](video_analysis.md).

## Local Model Support

| Class | Local Model Family |
|-------|--------------------|
| ImageToText | BLIP |
| AudioToText | Whisper |
| AudioClassifier | AST |
| ObjectDetector | YOLO |
| TextDetector | EasyOCR |
| FaceDetector | OpenCV / YOLOv8-face |
| CameraMotionDetector | OpenCV |
| MotionAnalyzer | OpenCV |
| ActionRecognizer | VideoMAE |
| SemanticSceneDetector | TransNetV2 |

## AudioToText

::: videopython.ai.AudioToText

## AudioClassifier

Detect and classify sounds, music, and audio events with timestamps using Audio Spectrogram Transformer (AST), a state-of-the-art model achieving 0.485 mAP on AudioSet.

### Basic Usage

```python
from videopython.ai import AudioClassifier
from videopython.base import Video

classifier = AudioClassifier(confidence_threshold=0.3)
video = Video.from_path("video.mp4")

result = classifier.classify(video)

# Clip-level predictions (overall audio content)
for label, confidence in result.clip_predictions.items():
    print(f"{label}: {confidence:.2f}")

# Timestamped events
for event in result.events:
    print(f"{event.start:.1f}s - {event.end:.1f}s: {event.label} ({event.confidence:.2f})")
```

::: videopython.ai.AudioClassifier

## ImageToText

::: videopython.ai.ImageToText

## Detection Classes

### ObjectDetector

::: videopython.ai.ObjectDetector

### FaceDetector

::: videopython.ai.FaceDetector

### TextDetector

::: videopython.ai.TextDetector

### CameraMotionDetector

::: videopython.ai.CameraMotionDetector

### MotionAnalyzer

Analyze motion in video frames using optical flow. Detects camera motion types (pan, tilt, zoom) and measures motion magnitude.

```python
from videopython.ai import MotionAnalyzer
from videopython.base import Video

analyzer = MotionAnalyzer()
video = Video.from_path("video.mp4")

# Analyze motion between two frames
motion = analyzer.analyze_frames(video.frames[0], video.frames[1])
print(f"Motion type: {motion.motion_type}, magnitude: {motion.magnitude:.2f}")

# Analyze entire video (memory-efficient)
results = analyzer.analyze_video_path("video.mp4", frames_per_second=1.0)
for timestamp, motion in results:
    print(f"{timestamp:.1f}s: {motion.motion_type} ({motion.magnitude:.2f})")
```

::: videopython.ai.MotionAnalyzer

### ActionRecognizer

Recognize actions and activities in video clips using VideoMAE, a masked autoencoder fine-tuned on Kinetics-400 (400 action classes like "walking", "running", "dancing", "answering questions").

```python
from videopython.ai import ActionRecognizer

recognizer = ActionRecognizer(model_size="base", confidence_threshold=0.1)

# Recognize actions in entire video
actions = recognizer.recognize_path("video.mp4", top_k=5)
for action in actions:
    print(f"{action.label}: {action.confidence:.1%}")

# Output: answering questions: 37.2%
#         using computer: 12.2%
```

::: videopython.ai.ActionRecognizer

### SemanticSceneDetector

ML-based scene boundary detection using TransNetV2. More accurate than histogram-based detection, especially for gradual transitions like fades and dissolves.

```python
from videopython.ai import SemanticSceneDetector

detector = SemanticSceneDetector(threshold=0.5, min_scene_length=1.0)
scenes = detector.detect_streaming("video.mp4")

for scene in scenes:
    print(f"Scene: {scene.start:.1f}s - {scene.end:.1f}s ({scene.duration:.1f}s)")
```

::: videopython.ai.SemanticSceneDetector

## Scene Data Classes

These classes are used by `SceneDetector` to represent analysis results:

### SceneBoundary

::: videopython.base.SceneBoundary

### BoundingBox

::: videopython.base.BoundingBox

### DetectedObject

::: videopython.base.DetectedObject

### AudioEvent

::: videopython.base.AudioEvent

### AudioClassification

::: videopython.base.AudioClassification

### MotionInfo

::: videopython.base.MotionInfo

### DetectedAction

::: videopython.base.DetectedAction

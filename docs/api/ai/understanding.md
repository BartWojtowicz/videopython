# AI Understanding

Analyze videos, transcribe audio, and describe visual content.

## Backend Support

| Class | local | openai | gemini | elevenlabs |
|-------|-------|--------|--------|------------|
| ImageToText | BLIP | GPT-4o | Gemini | - |
| AudioToText | Whisper | Whisper API | Gemini | - |
| AudioClassifier | AST | - | - | - |
| LLMSummarizer | Ollama | GPT-4o | Gemini | - |
| ObjectDetector | YOLO | GPT-4o | Gemini | - |
| TextDetector | EasyOCR | GPT-4o | Gemini | - |
| FaceDetector | OpenCV | - | - | - |
| ShotTypeClassifier | - | GPT-4o | Gemini | - |
| CameraMotionDetector | OpenCV | - | - | - |
| MotionAnalyzer | OpenCV | - | - | - |
| ActionRecognizer | VideoMAE | - | - | - |
| SemanticSceneDetector | TransNetV2 | - | - | - |

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

### With VideoAnalyzer

```python
from videopython.ai import VideoAnalyzer

analyzer = VideoAnalyzer()
result = analyzer.analyze(video, classify_audio=True, audio_classifier_threshold=0.3)

for scene in result.scene_descriptions:
    if scene.audio_events:
        print(f"Scene {scene.start:.1f}s - {scene.end:.1f}s:")
        for event in scene.audio_events:
            print(f"  {event.label}: {event.start:.1f}s - {event.end:.1f}s")
```

::: videopython.ai.AudioClassifier

## ImageToText

::: videopython.ai.ImageToText

## LLMSummarizer

::: videopython.ai.LLMSummarizer

## VideoAnalyzer

Comprehensive video analysis combining scene detection, frame understanding, and transcription.

### Basic Usage (In-Memory)

```python
from videopython.base import Video
from videopython.ai import VideoAnalyzer

video = Video.from_path("video.mp4")
analyzer = VideoAnalyzer()

# Full analysis with transcription
result = analyzer.analyze(video, frames_per_second=1.0, transcribe=True)

for scene in result.scene_descriptions:
    print(f"Scene: {scene.start:.1f}s - {scene.end:.1f}s")
    for fd in scene.frame_descriptions:
        print(f"  Frame {fd.frame_index}: {fd.description}")
```

### Memory-Efficient Analysis (Recommended for Long Videos)

For long videos, use `analyze_path()` which combines streaming scene detection with selective frame extraction. Only sampled frames are loaded into memory.

```python
from videopython.ai import VideoAnalyzer

analyzer = VideoAnalyzer()

# Analyze 20-minute video with minimal memory usage
# At 0.2 fps, loads ~240 frames instead of ~28,800
result = analyzer.analyze_path(
    "long_video.mp4",
    frames_per_second=0.2,  # Sample 1 frame every 5 seconds
    transcribe=True,
)

print(f"Detected {len(result.scene_descriptions)} scenes")
print(f"Summary: {result.get_full_summary()}")
```

### Key Frame Extraction

Extract representative frames from each scene for thumbnails or previews:

```python
result = analyzer.analyze_path(
    "video.mp4",
    extract_key_frames=True,
    key_frame_width=640,  # Width in pixels, height auto-scaled
)

for scene in result.scene_descriptions:
    if scene.key_frame:
        # scene.key_frame contains JPEG bytes
        with open(f"scene_{scene.scene_index}.jpg", "wb") as f:
            f.write(scene.key_frame)
        print(f"Scene {scene.scene_index}: key frame at {scene.key_frame_timestamp:.1f}s")
```

### Adaptive Frame Sampling

Use adaptive sampling to intelligently reduce frame count while maintaining coverage:

```python
result = analyzer.analyze_path(
    "video.mp4",
    sampling_strategy="adaptive",  # "fixed" or "adaptive"
)
# Uses start + ln(1+duration) + end formula per scene
# Reduces frames by ~27% compared to fixed sampling
```

### JSON Serialization

Export and import analysis results for caching or API responses:

```python
# Export to JSON-compatible dict
result = analyzer.analyze_path("video.mp4")
data = result.to_dict()

import json
with open("analysis.json", "w") as f:
    json.dump(data, f)

# Import from dict
from videopython.base import VideoDescription
with open("analysis.json") as f:
    data = json.load(f)
result = VideoDescription.from_dict(data)
```

::: videopython.ai.VideoAnalyzer

## Detection Classes

### ObjectDetector

::: videopython.ai.ObjectDetector

### FaceDetector

::: videopython.ai.FaceDetector

### TextDetector

::: videopython.ai.TextDetector

### ShotTypeClassifier

::: videopython.ai.ShotTypeClassifier

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

With VideoAnalyzer:

```python
from videopython.ai import VideoAnalyzer

analyzer = VideoAnalyzer()
result = analyzer.analyze(video, analyze_motion=True)

for scene in result.scene_descriptions:
    print(f"Scene {scene.start:.1f}s: {scene.dominant_motion_type} (avg magnitude: {scene.avg_motion_magnitude:.2f})")
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

With VideoAnalyzer:

```python
from videopython.ai import VideoAnalyzer

analyzer = VideoAnalyzer()
result = analyzer.analyze_path(
    "video.mp4",
    recognize_actions=True,
    action_confidence_threshold=0.1,
)

for scene in result.scene_descriptions:
    print(f"Scene {scene.start:.1f}s - {scene.end:.1f}s:")
    if scene.detected_actions:
        for action in scene.detected_actions:
            print(f"  {action.label}: {action.confidence:.1%}")
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

With VideoAnalyzer (use `use_semantic_scenes=True`):

```python
from videopython.ai import VideoAnalyzer

# Use ML-based scene detection instead of histogram-based
analyzer = VideoAnalyzer(use_semantic_scenes=True)
result = analyzer.analyze_path("video.mp4")

print(f"Detected {result.num_scenes} scenes")
```

::: videopython.ai.SemanticSceneDetector

### CombinedFrameAnalyzer

::: videopython.ai.CombinedFrameAnalyzer

## Scene Data Classes

These classes are used by `SceneDetector` and `VideoAnalyzer` to represent analysis results:

### SceneDescription

::: videopython.base.SceneDescription

### VideoDescription

::: videopython.base.VideoDescription

### FrameDescription

::: videopython.base.FrameDescription

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

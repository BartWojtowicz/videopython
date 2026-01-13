# AI Understanding

Analyze videos, transcribe audio, and describe visual content.

## Backend Support

| Class | local | openai | gemini | elevenlabs |
|-------|-------|--------|--------|------------|
| ImageToText | BLIP | GPT-4o | Gemini | - |
| AudioToText | Whisper | Whisper API | Gemini | - |
| AudioClassifier | PANNs | - | - | - |
| LLMSummarizer | Ollama | GPT-4o | Gemini | - |
| ObjectDetector | YOLO | GPT-4o | Gemini | - |
| TextDetector | EasyOCR | GPT-4o | Gemini | - |
| FaceDetector | OpenCV | - | - | - |
| ShotTypeClassifier | - | GPT-4o | Gemini | - |
| CameraMotionDetector | OpenCV | - | - | - |
| MotionAnalyzer | OpenCV | - | - | - |

## AudioToText

::: videopython.ai.AudioToText

## AudioClassifier

Detect and classify sounds, music, and audio events with timestamps using PANNs (Pretrained Audio Neural Networks).

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

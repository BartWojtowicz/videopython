# AI Understanding

Analyze videos, transcribe audio, and describe visual content.

## Backend Support

| Class | local | openai | gemini | elevenlabs |
|-------|-------|--------|--------|------------|
| ImageToText | BLIP | GPT-4o | Gemini | - |
| AudioToText | Whisper | Whisper API | Gemini | - |
| LLMSummarizer | Ollama | GPT-4o | Gemini | - |
| ObjectDetector | YOLO | GPT-4o | Gemini | - |
| TextDetector | EasyOCR | GPT-4o | Gemini | - |
| FaceDetector | OpenCV | - | - | - |
| ShotTypeClassifier | - | GPT-4o | Gemini | - |
| CameraMotionDetector | OpenCV | - | - | - |

## AudioToText

::: videopython.ai.AudioToText

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

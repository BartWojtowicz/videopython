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

## SceneDetector

::: videopython.ai.SceneDetector

## VideoAnalyzer

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

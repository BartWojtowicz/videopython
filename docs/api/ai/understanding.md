# AI Understanding

Analyze videos, transcribe audio, and describe visual content.

For a single aggregate, serializable analysis object across multiple analyzers, see [Video Analysis](video_analysis.md).

## Local Model Support

| Class | Local Model Family |
|-------|--------------------|
| SceneVLM | Qwen3-VL |
| AudioToText | Whisper |
| AudioClassifier | AST |
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

## SceneVLM

`SceneVLM` supports both Qwen3-VL `2B` and `4B` model variants.
Device selection is automatic by default (`cuda` -> `cpu`).

::: videopython.ai.SceneVLM

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

### DetectedText

::: videopython.base.DetectedText

### AudioEvent

::: videopython.base.AudioEvent

### AudioClassification

::: videopython.base.AudioClassification

### DetectedAction

::: videopython.base.DetectedAction

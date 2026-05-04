# AI Understanding

Analyze videos, transcribe audio, and describe visual content.

For a single aggregate, serializable analysis object across multiple analyzers, see [Video Analysis](video_analysis.md).

## Local Model Support

| Class | Local Model Family |
|-------|--------------------|
| SceneVLM | Qwen3.5 |
| AudioToText | Whisper |
| AudioClassifier | AST |
| SemanticSceneDetector | TransNetV2 |

## AudioToText

### Anti-hallucination knobs

Three Whisper decoder kwargs are surfaced for tuning on noisy or sparse-speech
audio:

```python
from videopython.ai import AudioToText

# Defaults: condition_on_previous_text=False (the cascading-hallucination fix),
# no_speech_threshold=0.6, logprob_threshold=-1.0.
transcriber = AudioToText()

# Tighter no-speech gate to drop more low-confidence windows on a film with
# heavy ambient music.
transcriber = AudioToText(no_speech_threshold=0.85)

# Restore Whisper's upstream default conditioning (e.g. for clean podcasts
# where cross-window context helps disambiguate homophones).
transcriber = AudioToText(condition_on_previous_text=True)
```

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

`SceneVLM` supports both Qwen3.5 `2B` and `4B` model variants.
Device selection is automatic by default (`cuda` -> `cpu`).

::: videopython.ai.SceneVLM

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

These classes are used by scene and audio analyzers to represent analysis results:

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

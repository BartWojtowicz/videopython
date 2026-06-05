# AI Understanding

Analyze videos, transcribe audio, describe visual content, and track faces per shot.

For a single aggregate, serializable analysis object across multiple analyzers, see [Video Analysis](video_analysis.md).

## Local Model Support

| Class | Local Model Family |
|-------|--------------------|
| SceneVLM | Qwen3.5 (4B / 9B / 27B) |
| AudioToText | Whisper |
| AudioClassifier | AST |
| SemanticSceneDetector | TransNetV2 |
| FaceTracker | YOLOv8-face |
| ObjectDetector | YOLOv8-COCO |

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

### Brand-name vocabulary biasing

Bias Whisper's first-window decoder toward a caller-supplied list of brand
names, product names, or proper nouns via the native `initial_prompt`
channel. Recovers near-mishears (e.g. Klarna → "carna", InPost →
"in post") on brand-monitoring inputs without any new model
dependencies.

```python
from videopython.ai import AudioToText

# Constructor default — applies to every transcribe() call on this instance.
transcriber = AudioToText(vocabulary=["Klarna", "Allegro", "InPost"])
result = transcriber.transcribe(video)

# Per-call override — useful when one transcriber serves multiple tenants.
result = transcriber.transcribe(video, vocabulary=["Pyszne", "Wolt"])
```

The list is normalized at construction (whitespace stripped,
case-insensitive dedup, casing of the first occurrence preserved).
Whisper reserves ~224 tokens for the prompt; longer lists are trimmed
from the tail with a single `WARNING` log line naming the count
dropped.

`VideoDubber` and `LocalDubbingPipeline` accept the same `vocabulary`
kwarg; it threads through to the underlying transcriber. Within
`VideoAnalyzer`, pass it via `analyzer_params`:

```python
from videopython.ai import VideoAnalyzer
from videopython.ai.video_analysis import VideoAnalysisConfig

config = VideoAnalysisConfig(
    analyzer_params={"audio_to_text": {"vocabulary": ["Klarna", "Allegro"]}}
)
analysis = VideoAnalyzer(config=config).analyze_path("brand_review.mp4")
```

Recovers names Whisper *almost* heard correctly. It will not catch
zero-prior names; an LLM correction pass would close that gap.

### Per-segment confidence

`TranscriptionSegment` carries three optional confidence fields populated from
the raw Whisper output: `avg_logprob`, `no_speech_prob`, and
`compression_ratio`. They are `None` when not available (e.g. on the
diarization-only path that builds segments from words without overlap match,
or on transcripts loaded from formats that don't carry the metadata).

These signals feed the dubbing pipeline's transcript-quality gate (median
`avg_logprob` is one of three reject flags) and Qwen3's confidence-aware
translation prompt (segments below threshold get a `low_confidence` hint). They
are also useful for downstream callers that want to drop low-quality segments
before further processing.

```python
result = AudioToText().transcribe(video)
for segment in result.segments:
    if segment.avg_logprob is not None and segment.avg_logprob < -1.0:
        print(f"low confidence: {segment.text!r}")
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

`SceneVLM` supports Qwen3.5 dense vision-capable variants via the
`model_size` kwarg: `"4b"` (default, ~8 GB FP16), `"9b"` (~18 GB FP16),
`"27b"` (~54 GB FP16, needs ≥48 GB). Device selection is automatic by
default (`cuda` -> `mps` -> `cpu`).

`analyze_scene()` and `analyze_frame()` return a structured
[`SceneDescription`](#scenedescription) with three fields: a one-sentence
`caption`, an open-list `subjects`, and a closed-enum `shot_type`. The
class uses few-shot JSON prompting with one parse-retry; on persistent
parse failure, the raw text becomes the caption while `subjects` and
`shot_type` are returned empty / `None`.

```python
from videopython.ai import SceneVLM

vlm = SceneVLM(model_size="4b")  # default
description = vlm.analyze_frame(frame_array)

print(description.caption)     # "A man in a cap speaks into a microphone."
print(description.subjects)    # ["man", "microphone", "cap"]
print(description.shot_type)   # "medium"
```

`SceneVLM.unload()` releases the model + processor for `low_memory`
parity with the dubbing pipeline's translator backends.

::: videopython.ai.SceneVLM

## SemanticSceneDetector

ML-based scene boundary detection using TransNetV2. More accurate than histogram-based detection, especially for gradual transitions like fades and dissolves.

```python
from videopython.ai import SemanticSceneDetector

detector = SemanticSceneDetector(threshold=0.5, min_scene_length=1.0)
scenes = detector.detect_streaming("video.mp4")

for scene in scenes:
    print(f"Scene: {scene.start:.1f}s - {scene.end:.1f}s ({scene.duration:.1f}s)")
```

::: videopython.ai.SemanticSceneDetector

## FaceTracker

`FaceTracker` runs YOLOv8-face detection and stitches detections into
per-shot tracks via IoU association — no embedding re-id, so a track
does not survive across shot boundaries. Two surfaces:

- `track_shot(frames, frame_indices)` returns a list of
  [`FaceTrack`](#facetrack) objects with stable ids within the shot.
  This is the API the analyzer uses.
- `detect_and_track(frame, frame_index)` / `track_video(frames)` are
  the legacy single-subject smoothed-position APIs used by
  `FaceTrackingCrop` (see [AI Transforms](transforms.md)).

```python
from videopython.ai import FaceTracker

tracker = FaceTracker(backend="auto")
tracks = tracker.track_shot(frames)

for track in tracks:
    print(f"track #{track.track_id}: {track.length} frames, "
          f"first frame {track.frame_indices[0]}")
```

::: videopython.ai.FaceTracker

## ObjectDetector

`ObjectDetector` runs a YOLOv8-COCO model and returns a list of
[`DetectedObject`](#detectedobject) per frame, with normalized bounding boxes
sorted by confidence. It is the object-detection counterpart to `FaceTracker`
and powers [`ObjectDetectionOverlay`](effects.md#objectdetectionoverlay).

The Ultralytics weights auto-download on first use; class names come from the
loaded model. Detection is gated by `confidence_threshold` and optionally
restricted to `class_filter`.

```python
from videopython.ai import ObjectDetector
from videopython.base import Video

video = Video.from_path("street.mp4")

detector = ObjectDetector(model_name="yolov8n.pt", class_filter=("person", "car"))
for obj in detector.detect(video.frames[0]):
    print(f"{obj.label} {obj.confidence:.2f} @ {obj.bounding_box}")

# Batched detection over several frames.
per_frame = detector.detect_batch(video.frames[:16])
```

::: videopython.ai.ObjectDetector

## Scene Data Classes

These classes are used by scene and audio analyzers to represent analysis results:

### SceneBoundary

::: videopython.base.SceneBoundary

### SceneDescription

::: videopython.base.SceneDescription

### BoundingBox

::: videopython.base.BoundingBox

### DetectedObject

::: videopython.base.DetectedObject

### DetectedFace

::: videopython.base.DetectedFace

### DetectedText

::: videopython.base.DetectedText

### FaceTrack

::: videopython.base.FaceTrack

### AudioEvent

::: videopython.base.AudioEvent

### AudioClassification

::: videopython.base.AudioClassification

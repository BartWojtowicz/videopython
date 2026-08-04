# AI understanding

Transcribe audio, classify sounds, detect scenes and objects, describe shots, and track
faces. For one aggregate object across all of these, see [Video
analysis](video-analysis.md).

| Class | Local model family |
|---|---|
| `AudioToText` | Whisper (+ pyannote for diarization) |
| `AudioClassifier` | AST |
| `SemanticSceneDetector` | TransNetV2 |
| `SceneVLM` | Ollama vision model |
| `FaceShotTracker` / `FaceSmoothingTracker` | OpenCV YuNet |
| `ObjectDetector` | D-FINE (COCO) |

## AudioToText

```python
from videopython.ai import AudioToText

transcription = AudioToText().transcribe(video)
```

Model sizes: `tiny`, `base`, `small`, `medium`, `large`, `turbo` (default). Diarization is
opt-in with `enable_diarization=True`. VAD-gated language detection runs by default
(`enable_vad=False` to skip).

### Anti-hallucination knobs

Three Whisper decoder kwargs are surfaced for noisy or sparse-speech audio. Defaults:
`condition_on_previous_text=False` (the cascading-hallucination fix),
`no_speech_threshold=0.6`, `logprob_threshold=-1.0`.

```python
AudioToText(no_speech_threshold=0.85)       # tighter gate under heavy ambient music
AudioToText(condition_on_previous_text=True)  # Whisper's upstream default; helps on clean podcasts
```

### Brand-name vocabulary biasing

Biases Whisper's first-window decoder toward supplied proper nouns via the native
`initial_prompt` channel, recovering near-mishears (Klarna → "carna", InPost → "in
post") with no extra model dependency.

```python
transcriber = AudioToText(vocabulary=["Klarna", "Allegro", "InPost"])   # instance default
result = transcriber.transcribe(video, vocabulary=["Pyszne", "Wolt"])   # per-call override
```

The list is normalized at construction: whitespace stripped, case-insensitive dedup, the
casing of the first occurrence preserved. Whisper reserves ~224 tokens for the prompt;
longer lists are trimmed from the tail with one `WARNING` log line naming the dropped
count. It recovers names Whisper *almost* heard — it will not catch zero-prior names.

`VideoDubber` and `LocalDubbingPipeline` take the same `vocabulary` kwarg. Inside
`VideoAnalyzer`, pass it through `analyzer_params`:

```python
VideoAnalysisConfig(analyzer_params={"audio_to_text": {"vocabulary": ["Klarna"]}})
```

### Per-segment confidence

`TranscriptionSegment` carries `avg_logprob`, `no_speech_prob` and `compression_ratio`
from the raw Whisper output. They are `None` when unavailable — for example on the
diarization-only path that builds segments from words without an overlap match, or on
transcripts loaded from formats that do not carry the metadata.

These feed the dubbing transcript-quality gate and the translator's confidence-aware
prompt, and are useful for dropping low-quality segments downstream.

```python
for segment in result.segments:
    if segment.avg_logprob is not None and segment.avg_logprob < -1.0:
        print(f"low confidence: {segment.text!r}")
```

::: videopython.ai.AudioToText

## AudioClassifier

Sound, music and audio-event classification with timestamps, using an Audio Spectrogram
Transformer (0.485 mAP on AudioSet).

```python
from videopython.ai import AudioClassifier

result = AudioClassifier(confidence_threshold=0.3).classify(video)

for label, confidence in result.clip_predictions.items():
    print(f"{label}: {confidence:.2f}")

for event in result.events:
    print(f"{event.start:.1f}s - {event.end:.1f}s: {event.label} ({event.confidence:.2f})")
```

::: videopython.ai.AudioClassifier

## SceneVLM

Describes scenes with a local Ollama vision model. Needs a running Ollama server and a
vision model that supports structured output; `model` is any tag you have pulled (default
`qwen3.6:27b`).

`analyze_scene()` and `analyze_frame()` return a
[`SceneDescription`](#videopython.base.SceneDescription): a one-sentence `caption`, an open-list
`subjects`, and a closed-enum `shot_type`. The schema is handed to Ollama's `format`, so
the model returns valid JSON directly.

```python
from videopython.ai import SceneVLM

vlm = SceneVLM(model="llava")
description = vlm.analyze_frame(frame_array)

description.caption      # "A man in a cap speaks into a microphone."
description.subjects     # ["man", "microphone", "cap"]
description.shot_type    # "medium"
```

`SceneVLM.unload()` clears the Ollama client, for `low_memory` parity.

::: videopython.ai.SceneVLM

## SemanticSceneDetector

TransNetV2 scene-boundary detection — more accurate than histogram methods, especially on
fades and dissolves.

```python
from videopython.ai import SemanticSceneDetector

detector = SemanticSceneDetector(threshold=0.5, min_scene_length=1.0)
for scene in detector.detect_streaming("video.mp4"):
    print(f"{scene.start:.1f}s - {scene.end:.1f}s ({scene.duration:.1f}s)")
```

::: videopython.ai.SemanticSceneDetector

## Face tracking

Two YuNet-based trackers share one detector, one per use case:

- `FaceShotTracker.track_shot(frames, frame_indices)` returns [`FaceTrack`](#videopython.base.FaceTrack)
  objects with ids that are stable **within a shot**, associated by IoU. There is no
  embedding re-identification, so a track does not survive a shot boundary. This is what
  `VideoAnalyzer` uses.
- `FaceSmoothingTracker.detect_and_track(frame, frame_index)` / `track_video(frames)` are
  the single-subject smoothed-position APIs behind
  [`FaceTrackingCrop`](operations.md#facetrackingcrop).

```python
from videopython.ai import FaceShotTracker

for track in FaceShotTracker().track_shot(frames):
    print(f"track #{track.track_id}: {track.length} frames, first {track.frame_indices[0]}")
```

::: videopython.ai.FaceShotTracker

::: videopython.ai.FaceSmoothingTracker

## ObjectDetector

Runs a D-FINE COCO model and returns [`DetectedObject`](#videopython.base.DetectedObject) per frame, with
normalized bounding boxes sorted by confidence. Weights (Apache-2.0) download from
HuggingFace on first use; class names come from the model config.

D-FINE uses VOC-style COCO names, so `class_filter` must use the model's exact spellings
(`motorbike`, `tvmonitor`).

```python
from videopython.ai import ObjectDetector

detector = ObjectDetector(model_name="ustc-community/dfine-nano-coco",
                          class_filter=("person", "car"))

for obj in detector.detect(video.frames[0]):
    print(f"{obj.label} {obj.confidence:.2f} @ {obj.bounding_box}")

per_frame = detector.detect_batch(video.frames[:16])
```

::: videopython.ai.ObjectDetector

## Result types

Shared, AI-free data classes from `videopython.base`, produced by the analyzers above and
consumed by `videopython.editing`.

::: videopython.base.SceneBoundary

::: videopython.base.SceneDescription

::: videopython.base.BoundingBox

::: videopython.base.DetectedObject

::: videopython.base.DetectedFace

::: videopython.base.DetectedText

::: videopython.base.FaceTrack

::: videopython.base.MotionInfo

::: videopython.base.AudioEvent

::: videopython.base.AudioClassification

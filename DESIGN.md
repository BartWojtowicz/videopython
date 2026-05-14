# DESIGN — proposed package layout

## Problems with current layout
- `base/` is a 5.5k-LOC kitchen sink: data container *plus* every editing primitive (`operation`, `transforms`, `effects`, `streaming`), audio (~1.25k LOC), text rendering, scene detection. The name implies foundation; the contents are everything-but-AI.
- `editing/` holds only `VideoEdit`/`SegmentConfig`, while the things being edited with (`Operation`, `Effect`, transforms, effects, `streaming.py`) live in `base/`. Misleading split.
- `audio/` is large and self-contained, but hidden one level deep under `base/`.
- `TranscriptionOverlay` is an `Effect` but lives under `base/text/` away from siblings in `effects.py`.
- `streaming.py`'s only consumer is `video_edit.py`.
- `scene.py` (histogram `SceneDetector`, 456 LOC) duplicates `ai.SemanticSceneDetector` at lower quality. Drop it; `SceneBoundary` (the result dataclass) stays in `base/description.py` since the AI detector still returns it.

## Proposed layout

```
videopython/
  __init__.py
  base/                       # data containers + io primitives (no editing logic)
    video.py                  # Video, VideoMetadata, FrameIterator
    description.py            # BoundingBox, DetectedFace, FaceTrack, SceneBoundary, ...
    transcription.py          # MOVED from base/text/ — pure data class
    image_text.py             # MOVED from base/text/ — generic PIL text renderer
    exceptions.py
    _ffmpeg.py, _video_io.py, _dimensions.py
  audio/                      # PROMOTED to top level
    audio.py                  # Audio, AudioMetadata
    analysis.py               # AudioLevels, SilentSegment, AudioSegment
  editing/                    # all editing primitives + plan runner
    operation.py              # MOVED from base/ — Operation/Effect base + registry
    transforms.py             # MOVED from base/ — CutSeconds, Resize, Crop, ...
    effects.py                # MOVED from base/ — Blur, Zoom, Fade, ...
    transcription_overlay.py  # MOVED from base/text/overlay.py — it's an Effect
    streaming.py              # MOVED from base/ — only video_edit.py uses it
    video_edit.py             # VideoEdit, SegmentConfig
  ai/                         # unchanged
```

## Dependency direction (must stay acyclic)
`base` → (nothing in-package) · `audio` → `base` · `editing` → `base`, `audio` · `ai` → `base`, `audio`, optionally `editing`

## Re-export contract (one-release deprecation, except scene)
Top-level `videopython.base` re-exports moved symbols (`Operation`, `Effect`, transforms, effects, `Audio`, `Transcription`, ...) for one release, with `DeprecationWarning` pointing at the new home. README examples (`from videopython.base import Video, CutSeconds, Resize, Fade`) keep working through the shim. `SceneDetector` is dropped outright — point users at `ai.SemanticSceneDetector` in the release notes; also delete `tests/base/test_scene.py` and `docs/api/core/scene.md`, and trim the `SceneDetector` rows from `docs/api/index.md` and `docs/examples/large-videos.md`.

## Out of scope here
`description.py` is still a grab-bag; splitting it (e.g. `detection.py` / `scene_types.py`) is a follow-up, not part of this move. Tests mirror the new tree (`tests/audio/`, `tests/editing/`) — currently `test_video_edit.py` already sits in `tests/base/` and should follow its module.

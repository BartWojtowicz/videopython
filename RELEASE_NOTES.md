# Release Notes

## 0.12.0

### Breaking Changes

- **FrameDescription**: Removed `camera_motion` field - use `motion.motion_type` instead
- **FrameDescription**: Changed `detected_faces` from `int | None` to `list[DetectedFace] | None`
  - Use `len(detected_faces)` to get the count
  - Local backend now returns faces with bounding boxes
  - Cloud backends return faces without bounding boxes (count only)
- **DetectedFace**: Made `bounding_box` optional (can be `None` for cloud backends)
- **FaceDetector**: Removed `count()` method - use `len(detect(image))` instead

### Changed

- Extracted duplicate `_adjust_audio_duration` methods to `Audio.fit_to_duration()`
- Vectorized `Vignette` effect (removed per-frame loop)
- Added `MIN_FRAMES_FOR_MULTIPROCESSING` threshold (100 frames) to `Resize`, `Blur`, and `ColorGrading` to avoid multiprocessing overhead for short videos

## 0.11.1

### New Features

- **Exception hierarchy**: Proper exception classes for better error handling
  - Base module: `VideoPythonError`, `VideoError`, `VideoLoadError`, `VideoMetadataError`, `AudioError`, `AudioLoadError`, `TransformError`, `InsufficientDurationError`, `IncompatibleVideoError`, `TextRenderError`, `OutOfBoundsError`
  - AI module: `BackendError`, `MissingAPIKeyError`, `UnsupportedBackendError`, `GenerationError`, `LumaGenerationError`, `RunwayGenerationError`, `ConfigError`
  - All exceptions exported from `videopython.base` and `videopython.ai`

- **AI module tests in CI**: Added lightweight AI tests to CI pipeline
  - Tests for models <100MB run in CI (YOLO, PANNs, OpenCV)
  - Tests for models 100MB+ excluded via `@pytest.mark.requires_model_download` marker
  - New `ai_tests` job in CI workflow

### Fixed

- Replaced broad `except Exception` patterns with specific exception types
- Config file parsing errors now emit warnings instead of failing silently
- LLM summarization failures now emit warnings with fallback behavior

### Changed

- Moved `VideoMetadataError` from `video.py` to `exceptions.py`
- Moved `AudioLoadError` from `audio/audio.py` to `exceptions.py`
- Moved AI backend exceptions from `backends.py` to `exceptions.py`
- Transitions now raise `InsufficientDurationError` instead of `RuntimeError`
- Transitions now raise `IncompatibleVideoError` instead of `ValueError`
- Video loading now raises `VideoLoadError` instead of `ValueError`

## 0.11.0

### New Features

- **SpeedChange transform**: Change video playback speed
  - Constant speed changes (e.g., 2x faster, 0.5x slower)
  - Smooth speed ramping for cinematic effects

- **ColorGrading effect**: Adjust video color properties
  - Brightness, contrast, saturation adjustments
  - Color temperature control (warm/cool tones)

- **Vignette effect**: Add darkened edges to frames
  - Configurable strength and radius

- **KenBurns effect**: Cinematic pan-and-zoom effect
  - Animate between two regions using normalized BoundingBox coordinates
  - Easing functions: linear, ease_in, ease_out, ease_in_out
  - Fluent API: `video.ken_burns(start_region, end_region, easing="ease_in_out")`

- **PictureInPicture transform**: Overlay video on main video
  - Configurable position (normalized 0-1), scale, border, rounded corners, opacity
  - Overlay loops automatically if shorter than main video
  - Fluent API: `video.picture_in_picture(overlay, position=(0.7, 0.7), scale=0.25)`

- **FaceDetector bounding boxes**: Now returns detailed face information
  - Returns `list[DetectedFace]` with normalized bounding box coordinates
  - Use `.count()` for backward compatibility with previous API

- **SpeedChange audio synchronization**: Audio now adjusts with video speed
  - Pitch-preserving time stretch using FFmpeg atempo filter
  - New `adjust_audio: bool = True` parameter (enabled by default)
  - For speed ramps, uses average speed for audio adjustment

- **PictureInPicture audio mixing**: Configurable audio handling for overlays
  - New `audio_mode` parameter: `"main"` (default), `"overlay"`, or `"mix"`
  - New `audio_mix` parameter for volume factors in mix mode, e.g. `(0.8, 0.5)`
  - Overlay audio loops automatically if shorter than main video

- **FaceTrackingCrop transform** (in `videopython.ai`): Crop video to follow detected faces
  - Create vertical (9:16) content from horizontal (16:9) by tracking speaker
  - Configurable face selection: largest, centered, or by index
  - Smooth position tracking with exponential moving average
  - Fallback options when face not detected: center, last_position, full_frame

- **SplitScreenComposite transform** (in `videopython.ai`): Arrange multiple videos in grid layouts
  - Layouts: 2x1, 1x2, 2x2, 1+2, 2+1
  - Face tracking for each cell to keep subjects centered
  - Configurable gap, border, and colors

- **AutoFramingCrop transform** (in `videopython.ai`): Intelligent cropping with cinematographic rules
  - Framing rules: thirds, center, headroom, dynamic
  - Configurable headroom and lead room
  - Smooth camera movement with speed limiting

- **FaceTracker utility** (in `videopython.ai`): Frame-by-frame face tracking with smoothing
  - Selection strategies: largest, centered, by index
  - Detection interval to reduce processing load
  - Exponential moving average for jitter-free tracking

- **Audio.time_stretch()**: New method for pitch-preserving time stretching
  - Supports extreme speeds via chained atempo filters (e.g., 4x, 0.25x)

- **Audio.scale_volume()**: New method for volume adjustment

## 0.10.0

### Breaking Changes

- Removed `TransformationPipeline` class - use the new fluent API instead

### New Features

- **Fluent API for Video**: Chain transformations directly on Video objects
  - `video.cut(start, end)` - cut by time range
  - `video.cut_frames(start, end)` - cut by frame range
  - `video.resize(width, height)` - resize (aspect ratio preserved if only one dimension given)
  - `video.crop(width, height)` - center crop
  - `video.resample_fps(fps)` - change frame rate
  - `video.transition_to(other, transition)` - combine with another video

- **Fluent API for VideoMetadata**: Validate operations before execution
  - Same methods as Video, but only transforms metadata (fast, no frame processing)
  - Raises `ValueError` for invalid operations (e.g., incompatible dimensions for transitions)
  - Example: `video.metadata.cut(0, 10).resize(1280, 720)` validates the operation chain

- **Transcription**: Added `words` property to access all words across segments

### Fixed

- Fixed PyTorch 2.6+ compatibility for speaker diarization (omegaconf serialization)

### Migration Guide

```python
# Before (0.9.x)
from videopython.base import TransformationPipeline, CutSeconds, Resize
pipeline = TransformationPipeline([CutSeconds(0, 10), Resize(1280, 720)])
result = pipeline.run(video)

# After (0.10.0)
result = video.cut(0, 10).resize(1280, 720)

# Validate before executing (optional)
output_meta = video.metadata.cut(0, 10).resize(1280, 720)
```

## 0.9.1

- Re-release of 0.9.0 (PyPI publish failed)

## 0.9.0

### Breaking Changes

- `add_audio()` and `add_audio_from_file()` now return a new `Video` instance instead of mutating in place
- Reduced public API exports from `videopython.base` (items still importable, just not in `__all__`)

### Deprecated

- `Audio.from_file()` is deprecated, use `Audio.from_path()` instead

### Fixed

- Security: Replaced `eval()` with `json.loads()` when parsing ffprobe output in audio loading
- `Audio.is_silent` now returns Python `bool` instead of `np.bool`
- Exception handling now uses specific exceptions instead of generic `except Exception`

### Changed

- Large video loading (>10GB estimated RAM) now emits `ResourceWarning` suggesting `FrameIterator`

## 0.8.3

- Added `preset` and `crf` parameters to `Video.save()` for encoding control
  - `preset`: Speed/compression tradeoff (ultrafast to veryslow), default "medium"
  - `crf`: Quality control (0-51), default 23. Lower values = better quality, larger files
  - Changed default preset from "ultrafast" to "medium" for better compression
  - Removed `-tune zerolatency` for improved compression efficiency

## 0.8.2

- Fixed missing `MotionInfo` export from `videopython.base`
- Added documentation build check to CI pipeline

## 0.8.1

- Added `MotionAnalyzer` for motion detection via optical flow analysis (Farneback method)
  - Detects motion types: static, pan, tilt, zoom, complex
  - Returns normalized motion magnitude (0-1) and raw pixel displacement
  - Frame-level analysis with `analyze_frames()` and `analyze_frame_sequence()`
  - Memory-efficient `analyze_video_path()` for long videos
  - Scene-level aggregation via `aggregate_motion()`
- Added `analyze_motion` parameter to `VideoAnalyzer.analyze()` and `VideoAnalyzer.analyze_path()`
  - Motion info automatically distributed to scene descriptions
  - New `avg_motion_magnitude` and `dominant_motion_type` fields on `SceneDescription`
- New dataclass: `MotionInfo`

## 0.8.0

- Added `AudioClassifier` for sound event detection using PANNs (Pretrained Audio Neural Networks)
  - Detects 527 AudioSet sound classes (speech, music, animals, vehicles, alarms, etc.)
  - Returns timestamped `AudioEvent` objects with start/end times and confidence scores
  - Configurable confidence threshold and model selection (Cnn14, Cnn10, ResNet38, MobileNetV2)
  - Frame-level predictions (~10ms resolution) automatically merged into coherent events
- Added `classify_audio` parameter to `VideoAnalyzer.analyze()` and `VideoAnalyzer.analyze_path()`
  - Audio events are automatically distributed to scene descriptions
  - New `audio_events` field on `SceneDescription`
- New dataclasses: `AudioEvent`, `AudioClassification`
- Added `panns-inference` to AI dependencies
- Fixed audio slicing issue in Blur transition

## 0.7.3

- Fixed thread safety issue in `ImageToText.describe_frames()` that caused meta tensor errors during concurrent model initialization
- Removed thread parallelism from frame captioning (was causing race conditions with lazy model loading)
- BLIP captioning now defaults to CPU instead of MPS on Apple Silicon (benchmarks show CPU is ~2x faster for this model)
- Parallel processing remains available for scene detection via `SceneDetector.detect_parallel()` (histogram-based, no AI models)

## 0.7.2

- Fixed compatibility with transformers 4.52+ (updated BLIP import path)

## 0.7.1

- Added `SceneDetector.detect_parallel()` for multi-core scene detection (3.5x speedup on 8 cores)
- Added `SceneDetector.detect_streaming()` for memory-efficient frame-by-frame processing
- Added `FrameIterator` for streaming video frames without loading entire video into memory
- Added `VideoAnalyzer.analyze_path()` for memory-efficient analysis of long videos
- Scene detection now works on video files directly without loading all frames into RAM

## 0.7.0

- **Breaking change**: Removed async/await from all AI module APIs - all functions are now synchronous
- Simplified API: no more `asyncio.run()` boilerplate required
- `describe_frames()` uses `ThreadPoolExecutor` for parallel frame processing (maintains performance)
- Removed `pytest-asyncio` dependency
- Fixed whisper type stubs (renamed to `__init__.pyi`)

## 0.6.3

- Removed `VideoUpscaler` - MMagic/mmcv has compatibility issues with NumPy 2.x and is unmaintained
- Removed MPS support for CogVideoX models (TextToVideo, ImageToVideo) - these require CUDA due to 364GB+ memory requirements on MPS
- Fixed `TextToMusic` crash on MPS - added missing `.cpu()` call before numpy conversion
- Removed `mmagic`, `mmcv`, `mmengine` dependencies (reduces install size significantly)
- Added MPS backend testing documentation (`scripts/mps_tests/MPS_TESTING.md`)

## 0.6.2

- Added silence detection in audio analysis (`Audio.get_silence_intervals()`)
- Added scene type detection for videos (`SceneType` enum with OUTDOOR, INDOOR, CLOSEUP, etc.)
- Added comprehensive tests for audio and scene functionality
- Added import isolation tests to ensure optional dependencies don't break core imports

## 0.6.1

- Added video understanding capabilities including scene analysis, color histograms, and object/text/face detection
- Added speaker diarization support for audio/video transcription
- Added video upscaling with RealBasicVSR model
- Added new AI backends: Luma Dream Machine and Runway Gen-4 Turbo for video generation
- Added Polish and German text-to-speech support
- Dropped `soundpython` dependency - audio functionality now built into videopython
- Extended documentation with new examples and API reference


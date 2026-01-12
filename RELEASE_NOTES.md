# Release Notes

## 0.7.4

- Fixed audio slicing issue in Blur transition.

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


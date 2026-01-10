# Release Notes

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


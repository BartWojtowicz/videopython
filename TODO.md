# Video Understanding Features - TODO

## Overview
Extend `videopython.ai.understanding` module to support comprehensive video understanding beyond transcription, including scene detection and visual frame analysis.

## Current State
- `videopython.ai.understanding.transcribe`: Audio/video transcription using Whisper
- Base classes: `Transcription`, `TranscriptionSegment`, `TranscriptionWord` in `videopython.base.text.transcription`

## Proposed Features

### 1. Scene Change Detection
**Module**: `videopython.ai.understanding.scenes`

**Purpose**: Detect when scenes change in a video to intelligently sample frames

**Key components**:
- `Scene` dataclass (base class in `videopython.base.video` or new `videopython.base.scenes`)
  - `start: float` - scene start time in seconds
  - `end: float` - scene end time in seconds
  - `frame_indices: list[int]` - key frame indices in this scene

- `SceneDetector` class
  - `detect(video: Video) -> list[Scene]` - detect scene changes
  - Implementation: Use frame differencing or PySceneDetect library

### 2. Frame Understanding
**Module**: `videopython.ai.understanding.frames`

**Purpose**: Understand visual content in video frames using vision-language models

**Key components**:
- `FrameDescription` dataclass (base class)
  - `frame_index: int`
  - `timestamp: float`
  - `description: str`

- `FrameAnalyzer` class
  - `analyze_frame(video: Video, frame_index: int, prompt: str) -> FrameDescription`
  - `analyze_frames(video: Video, frame_indices: list[int], prompt: str) -> list[FrameDescription]`
  - Implementation: Use vision-language model (e.g., LLaVA, CLIP + captioning)

### 3. Combined Video Understanding
**Module**: `videopython.ai.understanding.video`

**Purpose**: Combine transcription and frame understanding for complete video analysis

**Key components**:
- `VideoUnderstanding` dataclass (base class)
  - `transcription: Transcription`
  - `scenes: list[Scene]`
  - `frame_descriptions: list[FrameDescription]`

- `VideoAnalyzer` class
  - `analyze(video: Video, frames_per_scene: int = 3, description_prompt: str = "Describe what you see") -> VideoUnderstanding`
  - Workflow:
    1. Detect scenes
    2. Sample N frames per scene
    3. Analyze sampled frames
    4. Generate transcription
    5. Return combined understanding

## Implementation Priority
1. Scene detection (foundational)
2. Frame understanding (core feature)
3. Combined video analyzer (integrates everything)

## Open Questions
- Which vision-language model to use? (performance vs quality tradeoff)
- Should scene detection be configurable? (sensitivity threshold)
- Frame sampling strategy within scenes (uniform, start/middle/end, key frames)?

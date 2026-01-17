# Potential Improvements

This document tracks potential code improvements and optimizations identified during code review.

## Performance

### Redundant Video Metadata Calculations

**Location:** `video.py:996-997`

The `metadata` property creates a new `VideoMetadata` object on every access:

```python
@property
def metadata(self) -> VideoMetadata:
    return VideoMetadata.from_video(self)
```

**Suggestion:** Consider caching the metadata or using `@functools.cached_property`. Invalidate cache when `frames` or `fps` change.

### Memory Inefficiency in Video Loading

**Location:** `video.py:591-600`

Large video memory warning only warns but proceeds anyway. For videos >10GB estimated RAM usage, this could cause system issues.

**Suggestion:** Consider adding an option to fail-fast or automatically switch to `FrameIterator` streaming mode for very large videos.

### Redundant Data Copying in Effects

**Location:** `effects.py:39-50`

`Effect.apply()` creates intermediate Video objects when applying effects to a time range:

```python
video_with_effect = self._apply(video[effect_start_frame:effect_end_frame])
video = Video.from_frames(
    np.r_["0,2", video.frames[:effect_start_frame], video_with_effect.frames, video.frames[effect_end_frame:]],
    fps=video.fps,
)
```

This copies frames multiple times unnecessarily.

**Suggestion:** Consider in-place modification for effects that only affect a subset of frames.

### Audio Resampling Inefficiency

**Location:** `audio.py:376-404`

Custom FFT-based resampling implementation. The per-channel loop adds overhead:

```python
for channel in range(self.metadata.channels):
    resampled_data[:, channel] = self._resample_channel(...)
```

**Suggestion:** Consider using `scipy.signal.resample` or `librosa.resample` for better quality and performance.

### Repeated File I/O in Audio Time Stretch

**Location:** `audio.py:613-658`

`time_stretch()` saves to temp file, runs FFmpeg, then reads back. For chained operations, this creates unnecessary disk I/O.

**Suggestion:** Consider pipe-based FFmpeg communication to keep data in memory.

## Code Quality

### Print Statements

Multiple `print()` statements scattered throughout:
- `transforms.py`: lines 138, 193, 196, 305, 315, 516, 529
- `effects.py`: lines 106, 182, 310, 352, 464
- `ai/transforms.py`: lines 318, 499, 721

**Suggestion:** Consider using Python's `logging` module for configurable output, or add a verbosity flag.

### Repetitive FFmpeg Command Construction

Similar FFmpeg command patterns appear in:
- `video.py:603-632` (loading)
- `video.py:851-890` (saving)
- `audio.py:171-184` (loading)
- `audio.py:626-638` (time stretching)
- `audio.py:761-773` (saving)

**Suggestion:** Extract common FFmpeg patterns into a shared utility module.

## Architecture

### Over-Engineered Data Classes

**Location:** `description.py`

Many optional fields that may never be populated together:
- `FrameDescription` has 10 fields, 7 optional
- `SceneDescription` has 14 fields, 10 optional

**Suggestion:** Consider splitting into smaller, purpose-specific classes or using composition.

### Scene Detection Parallel Merge

**Location:** `scene.py:339-366`

The parallel detection merge logic is complex and may miss scene boundaries at segment edges. The boundary checking between segments is incomplete.

**Suggestion:** Add explicit boundary frame comparison when merging parallel detection results.

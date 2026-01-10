# Audio

Core audio class for loading, manipulating, analyzing, and saving audio files.

## Audio

The `Audio` class handles audio data with numpy arrays, supporting operations like slicing, concatenation, overlay mixing, resampling, analysis, and format conversion.

```python
from videopython.base import Audio

# Load from file
audio = Audio.from_file("music.mp3")

# Create silent track
silent = Audio.create_silent(duration_seconds=5.0, stereo=True)

# Basic operations
mono = audio.to_mono()
resampled = audio.resample(16000)
segment = audio.slice(start_seconds=1.0, end_seconds=5.0)

# Combine audio
combined = audio1.concat(audio2, crossfade=0.5)
mixed = audio1.overlay(audio2, position=2.0)

# Save
audio.save("output.wav")
```

::: videopython.base.audio.Audio

## Audio Analysis

The `Audio` class includes methods for analyzing audio levels, detecting silence, classifying content, and normalizing.

### Level Analysis

```python
from videopython.base import Audio

audio = Audio.from_file("audio.mp3")

# Get overall levels
levels = audio.get_levels()
print(f"Peak: {levels.db_peak:.1f} dB, RMS: {levels.db_rms:.1f} dB")

# Get levels for a specific segment
segment_levels = audio.get_levels(start_seconds=1.0, end_seconds=3.0)

# Get levels over time (sliding window analysis)
levels_over_time = audio.get_levels_over_time(window_seconds=0.1)
for timestamp, levels in levels_over_time:
    print(f"{timestamp:.2f}s: {levels.db_rms:.1f} dB")
```

### Silence Detection

```python
from videopython.base import Audio

audio = Audio.from_file("podcast.mp3")

# Detect silent segments
silent_segments = audio.detect_silence(
    threshold_db=-40.0,  # dB threshold
    min_duration=0.5,    # minimum silence duration in seconds
)

for seg in silent_segments:
    print(f"Silence: {seg.start:.2f}s - {seg.end:.2f}s ({seg.duration:.2f}s)")
```

### Segment Classification

Classify audio segments as speech, music, noise, or silence using heuristic analysis (no ML required).

```python
from videopython.base import Audio

audio = Audio.from_file("mixed_content.mp3")

# Classify 2-second segments with 50% overlap
segments = audio.classify_segments(segment_length=2.0, overlap=0.5)

for seg in segments:
    print(f"{seg.start:.1f}-{seg.end:.1f}s: {seg.segment_type.value} ({seg.confidence:.0%})")
```

### Normalization

```python
from videopython.base import Audio

audio = Audio.from_file("quiet_audio.mp3")

# Peak normalization (default)
normalized = audio.normalize(target_db=-3.0, method="peak")

# RMS normalization
normalized = audio.normalize(target_db=-18.0, method="rms")

# Verify
print(f"New peak: {normalized.get_levels().db_peak:.1f} dB")
```

## Data Classes

### AudioMetadata

Stores metadata for audio files including sample rate, channels, duration, and frame count.

::: videopython.base.audio.AudioMetadata

### AudioLevels

Audio level measurements (RMS, peak, dB values).

::: videopython.base.audio.AudioLevels

### SilentSegment

Represents a detected silent segment with timestamps.

::: videopython.base.audio.SilentSegment

### AudioSegment

A classified segment of audio with type and confidence.

::: videopython.base.audio.AudioSegment

### AudioSegmentType

Enum for audio segment classification: `SILENCE`, `SPEECH`, `MUSIC`, `NOISE`.

::: videopython.base.audio.AudioSegmentType

## Exceptions

### AudioLoadError

Exception raised when there's an error loading or saving audio files.

::: videopython.base.audio.AudioLoadError

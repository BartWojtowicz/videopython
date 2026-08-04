# Audio

`videopython.audio` — audio loading, manipulation, analysis and saving. Numpy-backed, no
ML dependencies.

## Audio

```python
from videopython.audio import Audio

audio = Audio.from_path("music.mp3")
silent = Audio.create_silent(duration_seconds=5.0, stereo=True)

mono = audio.to_mono()
resampled = audio.resample(16000)
segment = audio.slice(start_seconds=1.0, end_seconds=5.0)

combined = audio_a.concat(audio_b, crossfade=0.5)
mixed = audio_a.overlay(audio_b, position=2.0)

audio.save("output.wav")
```

::: videopython.audio.Audio

## Manipulation

```python
quieter = audio.scale_volume(0.5)          # 1.0 = unchanged
faster = audio.time_stretch(2.0)           # pitch-preserving (ffmpeg atempo)
slower = audio.time_stretch(0.5)
extreme = audio.time_stretch(4.0)          # chained filters handle extreme factors
fitted = audio.fit_to_duration(10.0)       # slices if longer, pads with silence if shorter
```

## Analysis

### Levels

```python
levels = audio.get_levels()
print(levels.db_peak, levels.db_rms)

segment_levels = audio.get_levels(start_seconds=1.0, end_seconds=3.0)

for timestamp, levels in audio.get_levels_over_time(window_seconds=0.1):
    print(f"{timestamp:.2f}s: {levels.db_rms:.1f} dB")
```

### Silence

```python
for seg in audio.detect_silence(threshold_db=-40.0, min_duration=0.5):
    print(f"{seg.start:.2f}s - {seg.end:.2f}s ({seg.duration:.2f}s)")
```

### Segment classification

Heuristic — speech, music, noise, or silence — with no ML dependency.

```python
for seg in audio.classify_segments(segment_length=2.0, overlap=0.5):
    print(f"{seg.start:.1f}-{seg.end:.1f}s: {seg.segment_type.value} ({seg.confidence:.0%})")
```

### Normalization

```python
normalized = audio.normalize(target_db=-3.0, method="peak")     # default
normalized = audio.normalize(target_db=-18.0, method="rms")
```

## Data classes

::: videopython.audio.AudioMetadata

::: videopython.audio.AudioLevels

::: videopython.audio.SilentSegment

::: videopython.audio.AudioSegment

::: videopython.audio.AudioSegmentType

## Exceptions

::: videopython.base.AudioError

::: videopython.base.AudioLoadError

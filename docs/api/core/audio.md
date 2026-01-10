# Audio

Core audio class for loading, manipulating, and saving audio files.

## Audio

The `Audio` class handles audio data with numpy arrays, supporting operations like slicing, concatenation, overlay mixing, resampling, and format conversion.

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

## AudioMetadata

Stores metadata for audio files including sample rate, channels, duration, and frame count.

::: videopython.base.audio.AudioMetadata

## AudioLoadError

Exception raised when there's an error loading or saving audio files.

::: videopython.base.audio.AudioLoadError

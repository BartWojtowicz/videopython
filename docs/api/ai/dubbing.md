# AI Dubbing

Dub videos into different languages or replace speech with custom text using voice cloning.

## Local Pipeline

Video dubbing runs with a local pipeline combining Whisper, translation models, XTTS, and Demucs.

## VideoDubber

Main class for video dubbing and voice revoicing.

### Basic Dubbing

Translate speech to another language while preserving the original speaker's voice:

```python
from videopython.ai.dubbing import VideoDubber
from videopython.base import Video

video = Video.from_path("video.mp4")
dubber = VideoDubber()

# Dub to Spanish with voice cloning
result = dubber.dub(
    video=video,
    target_lang="es",
    source_lang="en",
    preserve_background=True,  # Keep music and sound effects
    voice_clone=True,          # Clone original speaker's voice
)

# Save dubbed video
dubbed_video = video.add_audio(result.dubbed_audio, overlay=False)
dubbed_video.save("dubbed_video.mp4")

# Or use convenience method
dubbed_video = dubber.dub_and_replace(video, target_lang="es")
dubbed_video.save("dubbed_video.mp4")
```

### Voice Revoicing

Replace speech with completely different text using the original speaker's voice:

```python
from videopython.ai.dubbing import VideoDubber
from videopython.base import Video

video = Video.from_path("video.mp4")
dubber = VideoDubber()

# Make the person say something different
result = dubber.revoice(
    video=video,
    text="Hello everyone! This is a completely different message.",
    preserve_background=True,
)

print(f"Original duration: {result.original_duration:.1f}s")
print(f"New speech duration: {result.speech_duration:.1f}s")

# Save revoiced video (trimmed to speech length)
revoiced_video = dubber.revoice_and_replace(
    video=video,
    text="Hello everyone! This is a completely different message.",
)
revoiced_video.save("revoiced_video.mp4")
```

### Progress Tracking

```python
def on_progress(stage: str, progress: float) -> None:
    print(f"[{progress*100:5.1f}%] {stage}")

result = dubber.dub(
    video=video,
    target_lang="es",
    progress_callback=on_progress,
)
```

::: videopython.ai.dubbing.VideoDubber

## DubbingResult

Result of a dubbing operation containing the dubbed audio and metadata.

```python
result = dubber.dub(video, target_lang="es")

print(f"Translated {result.num_segments} segments")
print(f"Source language: {result.source_lang}")
print(f"Target language: {result.target_lang}")

# Access translated segments
for segment in result.translated_segments:
    print(f"'{segment.original_text}' -> '{segment.translated_text}'")

# Access voice samples used for cloning
for speaker, sample in result.voice_samples.items():
    print(f"{speaker}: {sample.metadata.duration_seconds:.1f}s sample")
```

::: videopython.ai.dubbing.DubbingResult

## RevoiceResult

Result of a revoicing operation.

```python
result = dubber.revoice(video, text="New message here")

print(f"Text: {result.text}")
print(f"Speech duration: {result.speech_duration:.1f}s")
print(f"Voice sample: {result.voice_sample.metadata.duration_seconds:.1f}s")
```

::: videopython.ai.dubbing.RevoiceResult

## TranslatedSegment

Individual translated speech segment with timing information.

::: videopython.ai.dubbing.TranslatedSegment

## SeparatedAudio

Audio separated into vocals and background components.

::: videopython.ai.dubbing.SeparatedAudio

## Supported Languages

Get the list of supported languages:

```python
languages = VideoDubber.get_supported_languages()
# {'en': 'English', 'es': 'Spanish', 'fr': 'French', ...}
```

Supported languages include: English, Spanish, French, German, Italian, Portuguese, Polish, Hindi, Arabic, Czech, Danish, Dutch, Finnish, Greek, Hebrew, Indonesian, Japanese, Korean, Malay, Norwegian, Romanian, Russian, Slovak, Swedish, Tamil, Thai, Turkish, Ukrainian, Vietnamese, Chinese.

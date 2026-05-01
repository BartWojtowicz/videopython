# AI Dubbing

Dub videos into different languages or replace speech with custom text using voice cloning.

## Local Pipeline

Video dubbing runs with a local pipeline combining Whisper, MarianMT translation, Chatterbox Multilingual TTS, and Demucs.

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

### Memory-Efficient Dubbing

The default pipeline keeps all four models (Whisper, Demucs, MarianMT, Chatterbox)
resident in memory and operates on `Video` objects that hold every frame in RAM.
For long or high-resolution sources — or memory-constrained hardware — two flags
trade a modest amount of latency for a much lower memory ceiling.

**Unload models between stages with `low_memory=True`:**

```python
# Each stage's model is released after it runs, so only one is resident at a time.
# Recommended for GPUs with <=12GB VRAM or hosts with <32GB RAM.
dubber = VideoDubber(low_memory=True)
dubbed_video = dubber.dub_and_replace(video, target_lang="es")
```

**Skip frame loading with `dub_file()`:**

```python
# Operates on file paths; extracts audio via ffmpeg, runs the pipeline on the
# audio only, and muxes the dubbed audio back into the source video using
# ffmpeg stream-copy (no video re-encode). Peak memory is bounded by model
# weights and the audio track, independent of video length and resolution.
dubber = VideoDubber(low_memory=True)
result = dubber.dub_file(
    input_path="long_video.mp4",
    output_path="dubbed.mp4",
    target_lang="es",
)
```

Use `dub_file()` when you don't need frame-level access in Python. Combine with
`low_memory=True` for the smallest memory footprint. See
[Processing Large Videos](../../examples/large-videos.md#dubbing-large-videos)
for a worked example.

### Whisper Model Selection

Pick the Whisper model size used for transcription. Larger models are more
accurate but use more VRAM and run slower. Default is `small`.

```python
# Higher accuracy on noisy or accented audio
dubber = VideoDubber(whisper_model="large")

# Lower VRAM footprint for short clips
dubber = VideoDubber(whisper_model="tiny")
```

Supported sizes: `tiny`, `base`, `small`, `medium`, `large`, `turbo`.

### Supplying a Pre-Computed Transcription

`dub()`, `dub_and_replace()`, and `dub_file()` accept an optional `transcription`
argument. Pass a pre-computed `Transcription` to skip the internal Whisper step
— useful when you've already transcribed (and possibly hand-edited) the source.

**Per-speaker voice cloning is driven by speaker labels on the supplied
transcription.** Three cases:

| Supplied transcription | `enable_diarization` | Behavior |
|---|---|---|
| Has speaker labels | any | Use supplied speakers; `enable_diarization` ignored |
| No speakers | `True` | Run pyannote on the audio, attach speakers to supplied words |
| No speakers | `False` | Use as-is; all segments share a single voice clone |

The diarize-on-supplied path requires word-level timings on the supplied
transcription — transcriptions loaded from SRT (one synthetic word per block)
are rejected.

```python
# Workflow: transcribe, edit, then dub with per-speaker cloning
from videopython.ai.dubbing import VideoDubber
from videopython.ai.understanding.audio import AudioToText
from videopython.base import Video

video = Video.from_path("video.mp4")

# 1. Transcribe with diarization
transcriber = AudioToText(enable_diarization=True)
transcription = transcriber.transcribe(video)

# 2. Edit segment text in-place (correct misrecognitions, etc.)
for seg in transcription.segments:
    if "incorrect word" in seg.text:
        seg.text = seg.text.replace("incorrect word", "correct word")

# 3. Dub using the edited transcription. Speaker labels from step 1 are
#    preserved, so each speaker gets their own cloned voice.
dubber = VideoDubber()
dubbed_video = dubber.dub_and_replace(
    video=video,
    target_lang="es",
    transcription=transcription,
)
```

If you have a transcription without speakers and want per-speaker cloning, pass
`enable_diarization=True` — pyannote will run standalone (skipping the Whisper
re-transcription).

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

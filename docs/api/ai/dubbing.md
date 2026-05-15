# AI Dubbing

Dub videos into different languages or replace speech with custom text using voice cloning.

## Local Pipeline

Video dubbing runs with a local pipeline combining Whisper for transcription, MarianMT or Qwen3 for translation, Chatterbox Multilingual TTS for speech synthesis, and Demucs for source separation. Translation backend selection is automatic by default — see [Translation Backend](#translation-backend) for details.

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

The default pipeline keeps all four models (Whisper, Demucs, the translation
backend, Chatterbox) resident in memory and operates on `Video` objects that
hold every frame in RAM.
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
accurate but use more VRAM and run slower. Default is `turbo` — large-v3
quality at ~8x the speed of `large` (and ~2x faster than `small`), so the
out-of-the-box dubbing path is now both more accurate and faster.

```python
# Even higher accuracy on very noisy or heavily accented audio
dubber = VideoDubber(whisper_model="large")

# Lower VRAM footprint for short clips
dubber = VideoDubber(whisper_model="tiny")
```

Supported sizes: `tiny`, `base`, `small`, `medium`, `large`, `turbo`.

### Anti-Hallucination Knobs

`VideoDubber` forwards three Whisper decoder kwargs to `AudioToText` so dubbing
inherits the same defaults — most importantly `condition_on_previous_text=False`,
which prevents a single hallucinated filler from cascading through the whole
dubbed track on noisy or sparse-speech inputs.

```python
# Defaults already protect against the cascading-hallucination failure mode.
dubber = VideoDubber()

# Tighter no-speech gate for a film with heavy ambient music.
dubber = VideoDubber(no_speech_threshold=0.85)
```

See [`AudioToText`](understanding.md#audiototext) for the full rationale.

### Brand-Name Vocabulary

Pass a list of brand names, product names, or proper nouns that may appear in
the source audio. The list is forwarded to `AudioToText` and biases Whisper's
first-window decoder via `initial_prompt`, recovering near-mishears (e.g.
Klarna → "carna") on brand-monitoring inputs.

```python
dubber = VideoDubber(vocabulary=["Klarna", "Allegro", "InPost"])
```

See [Brand-name vocabulary
biasing](understanding.md#brand-name-vocabulary-biasing) for normalization
rules and the token-budget guard.

### Translation Backend

Two translation backends ship with the dubbing pipeline:

- **MarianMT** (Helsinki-NLP) — fast on CPU, segment-isolated translation. Covers
  ~30 high-resource language pairs out of the box.
- **Qwen3** — Qwen3-4B-Instruct via `llama-cpp-python` (Q4_K_M GGUF, ~2.4 GB,
  downloaded on first use). Context-aware: prompts include a per-segment
  character budget derived from source duration and a `low_confidence` hint
  sourced from Whisper `avg_logprob`. Per-segment fallback to Marian if Qwen
  parse-retries both fail and the language pair is supported.

```python
# Auto resolver: Qwen3 on GPU when supported, MarianMT on CPU.
dubber = VideoDubber(translator="auto")

# Force MarianMT (e.g. CPU machines where Qwen3 wall time is impractical).
dubber = VideoDubber(translator="marian")

# Force Qwen3. Logs a WARNING on CPU because Qwen3-4B Q4_K_M runs ~10-15x
# slower than Marian without GPU acceleration.
dubber = VideoDubber(translator="qwen3")
```

Hard failures from Qwen3 (both the primary call and the per-segment Marian
fallback fail) are surfaced on `DubbingResult.translation_failures` as a list
of segment indices; those segments land on the result with empty translated
text. Empty list under MarianTranslator.

If neither backend covers the requested pair the auto resolver raises
`UnsupportedLanguageError` (importable from `videopython.ai.dubbing`).

### Output Options for `dub_file`

`dub_file()` writes the dubbed video by stream-copying the source video and
muxing the new audio. Two extras carry through automatically and one is opt-in:

- **Subtitles pass-through (automatic).** Subtitle streams from the source
  video are stream-copied into the output by default. Sources without subtitles
  are tolerated.
- **Source loudness match (automatic).** The dubbed audio is gain-matched to
  the source via BS.1770 integrated-loudness measurement (`pyloudnorm`,
  BSD-3) so the dub lands within ~1 LU of the source on dialogue-heavy mixes.
  Falls back to peak-amplitude match for clips shorter than 400 ms; post-gain
  peaks are clamped to 0.99.
- **`keep_original_audio=True` (opt-in).** Retains the source audio as a
  secondary audio track behind the dubbed one. Useful for editorial A/B; the
  dubbed track stays the default-playback track.

```python
result = dubber.dub_file(
    input_path="interview.mp4",
    output_path="interview_es.mp4",
    target_lang="es",
    keep_original_audio=True,  # source audio rides along as track #2
)
```

### Transcript Quality Gating

Even with `condition_on_previous_text=False`, sufficiently degenerate input
(ambient music, mostly-silent windows misread as speech) can still produce
unusable transcripts. The pipeline runs a cheap heuristic over the Whisper
output and exposes the assessment on every result.

Three checks fire flags:

- **Dominant phrase** — one phrase covers ≥70% of segment characters
  (catches cascades like the Japanese YouTube outro `「ご視聴ありがとうございました」`).
- **Low decoder confidence** — median `avg_logprob` < `-1.5`.
- **Sparse speech** — speech-region duration is <5% of clip duration on
  inputs >30s.

The `recommendation` is `"reject"` when the dominance flag fires together
with at least one other flag, `"warn"` when any single flag fires, `"ok"`
otherwise. Single repetition alone (chants, song lyrics) only warns.

```python
result = dubber.dub(video, target_lang="es")

q = result.transcript_quality
if q is not None:
    print(q.recommendation)            # "ok" | "warn" | "reject"
    print(q.dominant_phrase_fraction)  # 0.0-1.0
    print(q.flags)                     # ["dominant_phrase", ...]
```

Use `strict_quality=True` to refuse low-quality transcripts before paying for
Demucs, translation, and TTS:

```python
from videopython.ai.dubbing import GarbageTranscriptError

dubber = VideoDubber(strict_quality=True)
try:
    dubber.dub(video, target_lang="es")
except GarbageTranscriptError as exc:
    print("Refused:", exc.quality.flags)
```

### Timing Summary

`DubbingResult.timing_summary` aggregates the per-segment timing adjustments
the synchronizer applied to fit translated speech into source durations. High
truncation rates indicate translation produced text that was too long for the
source's spoken regions — a quality red flag worth surfacing in eval harnesses
or product UI.

```python
result = dubber.dub(video, target_lang="es")

ts = result.timing_summary
if ts is not None:
    print(f"{ts.clean_count}/{ts.total_segments} clean")
    print(f"{ts.truncated_count} truncated, worst {ts.max_truncation_seconds:.2f}s")
    print(f"mean speed factor {ts.mean_speed_factor:.3f}")
```

### Source-Prosody Expressiveness

`ChatterboxMultilingualTTS.generate()` exposes `exaggeration`, `cfg_weight`,
and `temperature` knobs. The dubbing pipeline derives an `Expressiveness`
profile per segment from source vocals RMS (relative to whole-vocals baseline)
and forwards it to Chatterbox, so the dub tracks the source's loud/quiet shape
instead of using flat defaults on every segment.

Three buckets, picked by-ear on `cam1_1min.mp4`:

| RMS ratio vs baseline | `exaggeration` | `cfg_weight` |
|---|---|---|
| `< 0.7×` (calm) | `0.3` | `0.7` |
| `0.7×–1.3×` (normal) | Chatterbox default | Chatterbox default |
| `> 1.3×` (dramatic) | `0.85` | `0.35` |

The `Expressiveness` dataclass is exported from `videopython.ai.dubbing`.

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

## DubbingConfig

Knobs shared by `VideoDubber` and `LocalDubbingPipeline`. Accept either
`config=DubbingConfig(...)` or pass the same knobs as flat kwargs — the
constructor builds a `DubbingConfig` internally.

```python
from videopython.ai.dubbing import DubbingConfig, VideoDubber

# Flat kwargs (recommended for ad-hoc calls)
dubber = VideoDubber(device="cuda", low_memory=True, whisper_model="large")

# Explicit config (recommended for reusable presets)
config = DubbingConfig(
    device="cuda",
    low_memory=True,
    whisper_model="large",
    translator="qwen3",
    vocabulary=["Klarna", "Allegro"],
)
dubber = VideoDubber(config=config)
```

::: videopython.ai.dubbing.DubbingConfig

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

## Expressiveness

Per-segment Chatterbox `generate()` knobs (`exaggeration`, `cfg_weight`,
`temperature`). `None` on any field means "let Chatterbox use its default".
The dubbing pipeline derives this from source vocals RMS automatically; the
type is exposed for users who want to inspect or override per-segment values.

::: videopython.ai.dubbing.Expressiveness

## TimingSummary

Aggregate stats over per-segment timing adjustments applied by the
synchronizer. Surfaces truncation and speed-change counts that translation
quality eval harnesses can compare across backends.

::: videopython.ai.dubbing.models.TimingSummary

## TranscriptQuality

Heuristic quality assessment over a Whisper transcription. Surfaced on every
`DubbingResult`; drives the optional `strict_quality` reject path.

::: videopython.ai.dubbing.TranscriptQuality

## GarbageTranscriptError

Raised by the pipeline when `strict_quality=True` and the transcript-quality
heuristic returns `recommendation="reject"`. Carries the triggering
`TranscriptQuality` as `error.quality` for caller introspection.

::: videopython.ai.dubbing.GarbageTranscriptError

## UnsupportedLanguageError

Raised by the translator auto-resolver when neither MarianMT nor Qwen3 covers
the requested `(source_lang, target_lang)` pair. Carries both fields for
caller introspection without parsing the message.

::: videopython.ai.dubbing.UnsupportedLanguageError

## Supported Languages

Get the list of supported languages:

```python
languages = VideoDubber.get_supported_languages()
# {'en': 'English', 'es': 'Spanish', 'fr': 'French', ...}
```

Supported languages include: English, Spanish, French, German, Italian, Portuguese, Polish, Hindi, Arabic, Czech, Danish, Dutch, Finnish, Greek, Hebrew, Indonesian, Japanese, Korean, Malay, Norwegian, Romanian, Russian, Slovak, Swedish, Tamil, Thai, Turkish, Ukrainian, Vietnamese, Chinese.

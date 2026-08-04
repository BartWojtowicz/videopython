# AI dubbing

`videopython.ai.dubbing` — translate speech, clone the voice, and re-time the dub onto the
source. Whisper for transcription, a local Ollama model for translation, Chatterbox for
TTS, Demucs for source separation. Task recipes are in
[Dub a video into another language](../../how-to/dubbing.md).

## VideoDubber

Four entry points:

| Method | Input → output | Notes |
|---|---|---|
| `dub_file(input_path, output_path, ...)` | path → file | Never loads frames; video is stream-copied |
| `dub(video, ...)` | `Video` → `DubbingResult` | |
| `dub_and_replace(video, ...)` | `Video` → `Video` | Convenience over `dub` |
| `revoice(video, text, ...)` / `revoice_and_replace(...)` | `Video` → result / `Video` | New words, original voice |

`dub`, `dub_and_replace` and `dub_file` all accept a pre-computed `transcription`. Speaker
labels on it drive per-speaker voice cloning; the diarize-on-supplied path needs
word-level timings, so SRT-loaded transcriptions (one synthetic word per block) are
rejected.

`dub_file` copies subtitle streams through automatically and gain-matches the dub to the
source with BS.1770 integrated loudness (`pyloudnorm`; falls back to peak match under
400 ms, post-gain peaks clamped to 0.99). `keep_original_audio=True` retains the source
audio as a secondary track.

::: videopython.ai.dubbing.VideoDubber

## DubbingConfig

Knobs shared by `VideoDubber` and `LocalDubbingPipeline`. Pass `config=DubbingConfig(...)`
or the same knobs as flat kwargs — the constructor builds a `DubbingConfig` either way.

::: videopython.ai.dubbing.DubbingConfig

## Results

```python
result = dubber.dub(video, target_lang="es")

result.num_segments, result.source_lang, result.target_lang
result.translation_failures            # indices the model never returned

for segment in result.translated_segments:
    print(f"{segment.original_text!r} -> {segment.translated_text!r}")

for speaker, sample in result.voice_samples.items():
    print(f"{speaker}: {sample.metadata.duration_seconds:.1f}s sample")
```

::: videopython.ai.dubbing.DubbingResult

::: videopython.ai.dubbing.RevoiceResult

::: videopython.ai.dubbing.TranslatedSegment

::: videopython.ai.dubbing.SeparatedAudio

## Expressiveness

Per-segment Chatterbox `generate()` knobs (`exaggeration`, `cfg_weight`, `temperature`).
`None` on a field means "let Chatterbox use its default". The pipeline derives these from
source vocals RMS relative to the whole-vocals baseline, so the dub tracks the source's
loud/quiet shape instead of using flat defaults everywhere.

| RMS ratio vs baseline | `exaggeration` | `cfg_weight` |
|---|---|---|
| `< 0.7×` (calm) | `0.3` | `0.7` |
| `0.7×–1.3×` (normal) | Chatterbox default | Chatterbox default |
| `> 1.3×` (dramatic) | `0.85` | `0.35` |

::: videopython.ai.dubbing.Expressiveness

## TimingSummary

Aggregate stats over the per-segment timing adjustments the synchronizer applied. High
truncation counts mean the translation produced text too long for the source's spoken
regions.

::: videopython.ai.dubbing.models.TimingSummary

## TranscriptQuality

Heuristic assessment over the Whisper transcription, surfaced on every `DubbingResult` and
driving the optional `strict_quality` reject path. Flags: dominant phrase covering ≥70% of
segment characters, median `avg_logprob` < `-1.5`, or speech under 5% of a clip longer
than 30 s. `recommendation` is `"reject"` when dominance fires together with another flag,
`"warn"` for any single flag, `"ok"` otherwise.

::: videopython.ai.dubbing.TranscriptQuality

::: videopython.ai.dubbing.GarbageTranscriptError

## Supported languages

```python
VideoDubber.get_supported_languages()
# {'en': 'English', 'es': 'Spanish', 'fr': 'French', ...}
```

English, Spanish, French, German, Italian, Portuguese, Polish, Hindi, Arabic, Czech,
Danish, Dutch, Finnish, Greek, Hebrew, Indonesian, Japanese, Korean, Malay, Norwegian,
Romanian, Russian, Slovak, Swedish, Tamil, Thai, Turkish, Ukrainian, Vietnamese, Chinese.

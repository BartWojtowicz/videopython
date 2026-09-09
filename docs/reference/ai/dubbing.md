# AI dubbing

`videopython.ai.dubbing` — translate speech, clone the voice, and re-time the dub onto the
source. Whisper for transcription, an Ollama model for translation, Chatterbox for
TTS, Demucs for source separation. Task recipes are in
[Dub a video into another language](../../how-to/dubbing.md).

## VideoDubber

Entry points:

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

Settings shared by `VideoDubber` and `LocalDubbingPipeline`. Pass `config=DubbingConfig(...)`
or the same knobs as flat kwargs — the constructor builds a `DubbingConfig` either way.

::: videopython.ai.dubbing.DubbingConfig

## Results

```python
result = dubber.dub(video, target_lang="es")

result.num_segments, result.source_lang, result.target_lang
result.translation_failures            # original indices with missing/invalid translation parts
result.synthesis_failures              # original indices without generated speech

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

Aggregate stats over the per-segment timing adjustments. `excessive_speed_count`
counts turns exceeding the preferred maximum speed; `max_speed_factor` records the
fastest adjustment. The pipeline borrows following silence before speeding up and
preserves complete speech instead of clipping its tail. Small tempo-filter duration
errors are corrected by resampling the entire output, which can slightly shift pitch.
`clean_count` includes speed factors within 0.01 of 1.0; `stretched_count` includes
the remaining adjustments. See [Update timing consumers](../../how-to/update-dubbing.md)
when migrating callers or saved results.

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

The returned map names languages known to the translator. It is not a tested
language matrix for the complete dubbing pipeline. Translation attempts other codes;
actual translation and synthesis support depends on the selected models.

## OllamaTranslator

Import from `videopython.ai.dubbing.translation`. The default model is
`qwen3.6:27b`; `VideoDubber(translator_model=..., translator_host=...)` forwards a
model tag and host. Vision is not required.

`max_tokens` defaults to 4096 and must be at least 140. `n_ctx` defaults to 8192
and must be at least `max_tokens + 1020`. The `options` keys `num_predict` and
`num_ctx` override these values. Invalid effective budgets raise at construction.
These are allocation estimates, not tokenizer guarantees.

`keep_alive` defaults to five minutes between requests. `None` uses the server
policy. `unload()` requests release on the Ollama server. A failed release logs a
warning and clears the local client without discarding completed translations;
server memory can remain allocated. `VideoDubber(low_memory=True)` requests
release after translation and before speech synthesis.

::: videopython.ai.dubbing.translation.OllamaTranslator

## Phrase and synthesis limits

Tiny adjacent fragments can join within one speaker when the gap is at most
150 ms. A group contains at most four turns spanning at most ten seconds. The
longer fragment supplies the expression profile. Isolated groups shorter than
100 ms become synthesis failures. Original transcript entries remain separate.

Local synthesis retries invalid token outputs up to three attempts. Duration
checks can flag the backend's output limit but cannot verify spoken-word coverage.
For the relationship between turns, phrases, and source indices, see
[The dubbing pipeline](../../explanation/dubbing.md).

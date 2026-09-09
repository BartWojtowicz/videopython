# Dub a video into another language

`VideoDubber` transcribes the source, translates it, re-synthesizes the speech in the
original speaker's voice, and fits the result back onto the source timing. Every stage
runs with infrastructure you control: Whisper, an Ollama model for translation,
Chatterbox for TTS, and Demucs to keep the background music.

Needs `pip install "videopython[ai]"` and a running Ollama server
([Install](../install.md)).

## Dub a file

The path-based API is the one to reach for by default — it never loads frames:

```python
from videopython.ai.dubbing import VideoDubber

dubber = VideoDubber()
result = dubber.dub_file(
    input_path="interview.mp4",
    output_path="interview_es.mp4",
    target_lang="es",
    source_lang="en",        # omit to auto-detect
    voice_clone=True,        # keep the original speaker's voice
    preserve_background=True,  # keep music and effects under the dub
)

print(f"Translated {result.num_segments} segments")
```

Two things happen automatically on this path: subtitle streams are copied through from
the source, and the dubbed audio is gain-matched to the source with BS.1770 integrated
loudness (within ~1 LU on dialogue-heavy mixes). Add `keep_original_audio=True` to retain
the source audio as a second track for A/B review — the dub stays the default track.

## Dub an in-memory video

When you need the frames in Python anyway:

```python
from videopython.base import Video

video = Video.from_path("video.mp4")
result = dubber.dub(video=video, target_lang="es", preserve_background=True)

video.add_audio(result.dubbed_audio, overlay=False).save("dubbed.mp4")

# or, in one call
dubber.dub_and_replace(video, target_lang="es").save("dubbed.mp4")
```

## Replace what someone says

`revoice` keeps the voice and swaps the words:

```python
result = dubber.revoice(
    video=video,
    text="Hello everyone! This is a completely different message.",
    preserve_background=True,
)
print(result.original_duration, result.speech_duration)

dubber.revoice_and_replace(video, text="...").save("revoiced.mp4")
```

## Track progress

```python
def on_progress(stage: str, progress: float) -> None:
    print(f"[{progress * 100:5.1f}%] {stage}")

result = dubber.dub(video=video, target_lang="es", progress_callback=on_progress)
```

## Fit it in less memory

`low_memory=True` releases the in-process transcription, separation and speech
synthesis models after their stages — recommended for GPUs with ≤12 GB VRAM or
hosts under 32 GB RAM:

```python
dubber = VideoDubber(low_memory=True)
```

Translation requests keep the Ollama model resident for five minutes between calls,
including when the server defaults to immediate eviction. `low_memory=True`
explicitly unloads it after translation, before speech synthesis loads. Standalone
`OllamaTranslator` users can override `keep_alive` (`None` uses the server policy)
and call `unload()` to release residency requested by the translator.

Combine it with `dub_file()` for the smallest footprint; see
[Process hour-long videos](long-videos.md#dub-without-loading-frames).

## Tune the transcription

```python
dubber = VideoDubber(whisper_model="large")        # tiny|base|small|medium|large|turbo
dubber = VideoDubber(no_speech_threshold=0.85)     # tighter gate under heavy music
dubber = VideoDubber(vocabulary=["Klarna", "Allegro", "InPost"])  # brand-name biasing
```

`turbo` is the default: large-v3 quality at ~8× the speed. `condition_on_previous_text`
defaults to `False`, which stops one hallucinated filler from cascading through the whole
track. Details in [AI understanding](../reference/ai/understanding.md#audiototext).

## Reject garbage input before paying for it

Degenerate audio (ambient music, near-silence read as speech) produces unusable
transcripts. Every result carries a heuristic assessment:

```python
q = result.transcript_quality
if q is not None:
    print(q.recommendation)            # "ok" | "warn" | "reject"
    print(q.flags)                     # ["dominant_phrase", ...]
    print(q.dominant_phrase_fraction)
```

Three checks fire flags: one phrase covering ≥70% of segment characters, a median
`avg_logprob` below `-1.5`, or speech covering <5% of a clip longer than 30 s. The
recommendation is `reject` when the dominance flag fires together with another, `warn`
for any single flag. Repetition alone (chants, lyrics) only warns.

To refuse before Demucs, translation and TTS run:

```python
from videopython.ai.dubbing import GarbageTranscriptError

dubber = VideoDubber(strict_quality=True)
try:
    dubber.dub(video, target_lang="es")
except GarbageTranscriptError as exc:
    print("Refused:", exc.quality.flags)
```

## Check the timing fit

The pipeline first uses available silence before the next turn. If complete speech
still cannot fit at the preferred 1.1× maximum, it goes faster instead of cutting
words from the end. Excessive speed can sound unnatural and warrants review:

```python
ts = result.timing_summary
if ts is not None:
    print(f"{ts.clean_count}/{ts.total_segments} clean")
    print(f"{ts.excessive_speed_count} above preferred speed; maximum {ts.max_speed_factor:.2f}×")
    print(f"mean speed factor {ts.mean_speed_factor:.3f}")
```

## Give each speaker their own cloned voice

Per-speaker cloning is driven by speaker labels on the transcription. `dub()`,
`dub_and_replace()` and `dub_file()` all accept a pre-computed `transcription`, which also
lets you correct the text before it is translated:

```python
from videopython.ai import AudioToText

transcription = AudioToText(enable_diarization=True).transcribe(video)

for seg in transcription.segments:
    seg.text = seg.text.replace("incorrect word", "correct word")

dubber.dub_and_replace(video=video, target_lang="es", transcription=transcription)
```

On CUDA, compatible local speech decoder operations use graph replay to reduce
launch overhead. Model weights, precision and generation settings stay unchanged.
Other input layouts use the original execution path, as does CPU synthesis; if
graph capture is unavailable, synthesis falls back automatically. Graph buffers
are released with the model.

| Supplied transcription | `enable_diarization` | Behavior |
|---|---|---|
| Has speaker labels | any | Supplied speakers are used; the flag is ignored |
| No speakers | `True` | pyannote runs on the audio and attaches speakers to the supplied words |
| No speakers | `False` | Used as-is; all segments share one voice clone |

The diarize-on-supplied path needs word-level timings, so transcriptions loaded from SRT
(one synthetic word per block) are rejected.

## Pick the translation model

Translation goes through `OllamaTranslator`, a single Ollama text model. Each request
translates one bounded source part, with neighboring text marked as context only.
Long turns split at sentence, clause or word boundaries, falling back to character
boundaries for text without spaces, and reassemble under their original segment
and speaker. A duration-derived character target encourages concise speech while
preserving meaning. Requests remain sequential to isolate segment identities; this
adds request overhead compared with batching. Invalid identities,
duplicate entries, empty responses and output-budget exhaustion trigger a retry.
If any part remains unavailable, the whole parent appears in `translation_failures`.

When constructing `OllamaTranslator` directly, set `max_tokens` to at least 140 and
`n_ctx` to at least `max_tokens + 1020`. Defaults are 4096 and 8192 respectively.
The equivalent `options` keys, `num_predict` and `num_ctx`, override these values;
invalid effective budgets fail at construction. For example, a 4096-token context
can use `max_tokens=1024`. These are allocation estimates, not tokenizer guarantees.

Unloading requests release of the model on the Ollama server. A failed release
request logs a warning and clears the local client without discarding translations
or masking an exception from the caller. Server memory may remain allocated.

Select a model already downloaded in Ollama. For example, run
`ollama pull translategemma:12b`, then configure the dubber:

```python
dubber = VideoDubber(
    translator_model="translategemma:12b",
    translator_host="http://localhost:11434",
    low_memory=True,
)
```

Any language pair is attempted — the pipeline does not reject a target language up
front. An empty `translation_failures` list establishes response availability, not
semantic accuracy. Review meaning, numbers, names and speaker alignment before publishing.

Local Chatterbox synthesis splits long translations into bounded calls, preserving
voice and expression settings, and joins their audio before synchronizing the parent
turn. Calls approaching the backend's speech-token ceiling retry with smaller text
units. Sampling excludes invalid vocoder token IDs while preserving end-of-speech.
Invalid token outputs are also rejected before GPU indexing and retried up
to three attempts. The required `videopython-chatterbox>=0.1.7.post2` fixes the empty alignment
reduction for short text in the backend itself. The duration check can flag a likely cap but cannot prove
every word was spoken.
Tiny adjacent fragments can join within one speaker when the gap is at most 150 ms;
groups contain at most four turns spanning at most ten seconds. The longer fragment
supplies the expression profile. Isolated groups shorter than 100 ms are reported
as synthesis failures. Original transcript entries remain available separately.

Check `result.synthesis_failures` for original segment indices whose speech could not
be generated or timed. Also inspect `result.timing_summary`: fitting the complete
speech into the source window can require excessive speed. Verify the final audio,
including the ends of long turns, rather than relying on success counts alone.

Speaker diarization turns are not necessarily good dubbing units. Before translation,
long turns are split into phrases at sentence ends, then clauses or pauses, using
validated word timestamps. Phrase text is sliced from the source transcript to retain
its internal spacing. Complete sentences take priority over earlier commas;
roughly eight-second phrases are preferred without creating tiny word fragments.
Missing, partial or inconsistent word alignment leaves the source segment intact;
we do not invent timestamps by dividing text proportionally.

`source_transcription` remains unchanged. Translated phrases expose
`source_segment_index` to trace them to that transcript; translation and synthesis
failure lists still refer to original source segment indices. Speaker reference
extraction uses the full original turns, preserving voice identity across phrases.

Each phrase starts at its source-word anchor. Shorter generated speech keeps its
natural speed and pauses until the next phrase, instead of filling the whole window
with a slowdown. Available gaps can absorb longer speech before acceleration.
Assembly applies a 5 ms fade at each phrase edge to reduce clicks at silence
boundaries. It keeps the sample count and source anchor unchanged.
See [Check the timing fit](#check-the-timing-fit) for speed limits and result checks.

Dubbing prefers FFmpeg's [Rubber Band filter](https://ffmpeg.org/ffmpeg-filters.html#rubberband)
when stretching is necessary and the filter is available, with an explicit warning
and `atempo` fallback on builds without it. General `Audio.time_stretch()` calls
retain the `atempo` default; callers may select `method="rubberband"` explicitly.
Phrase alignment reduces accumulated timing drift; it is not phoneme-level lip sync,
and translation or TTS quality still requires listening review.

## Update timing consumers

Remove the `min_speed` argument from `TimingSynchronizer` calls. Shorter speech
keeps its natural speed. Use `max_speed` to set the preferred acceleration limit.

Remove uses of `TimingAdjustment.was_truncated`, `truncation_seconds`, and
`excessive_slowdown`. Remove uses of `TimingSummary.truncated_count`,
`max_truncation_seconds`, `excessive_slowdown_count`, and `min_speed_factor`.
Regenerate saved timing summaries from the current pipeline. Use
`excessive_speed_count` and `max_speed_factor` to select output for listening review.

## Swap the TTS backend

Synthesis sits behind a `runtime_checkable` `SpeechBackend` protocol. Inject your own to
keep chatterbox out of the process entirely:

```python
from videopython.ai.dubbing import VideoDubber
from videopython.audio import Audio

class RemoteTTS:
    def generate_audio(self, text, voice_sample=None, voice_sample_path=None,
                       exaggeration=None, cfg_weight=None, temperature=None) -> Audio:
        ...   # call your remote synthesizer, return an Audio

dubber = VideoDubber(tts_backend=RemoteTTS())
```

videopython ships the protocol and the local backend only — there is no reference
remote/HTTP implementation.

## Reusable presets

Flat kwargs and `DubbingConfig` are equivalent; the constructor builds a config either
way.

```python
from videopython.ai.dubbing import DubbingConfig, VideoDubber

dubber = VideoDubber(device="cuda", low_memory=True, whisper_model="large")

config = DubbingConfig(
    device="cuda",
    low_memory=True,
    whisper_model="large",
    translator_model="qwen3.6:27b",
    vocabulary=["Klarna", "Allegro"],
)
dubber = VideoDubber(config=config)
```

Full field lists, result types, and supported languages: [AI dubbing
reference](../reference/ai/dubbing.md).

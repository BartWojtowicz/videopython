# Dub a video into another language

`VideoDubber` transcribes the source, translates it, re-synthesizes the speech in the
original speaker's voice, and fits the result back onto the source timing. Every stage
runs locally: Whisper, a local Ollama model for translation, Chatterbox for TTS, and
Demucs to keep the background music.

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

The default pipeline keeps all four models resident. `low_memory=True` releases each
one after its stage — recommended for GPUs with ≤12 GB VRAM or hosts under 32 GB RAM:

```python
dubber = VideoDubber(low_memory=True)
```

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

Translated speech that does not fit the source's spoken gaps gets time-stretched or
truncated. High truncation rates are a translation-quality red flag worth surfacing:

```python
ts = result.timing_summary
if ts is not None:
    print(f"{ts.clean_count}/{ts.total_segments} clean")
    print(f"{ts.truncated_count} truncated, worst {ts.max_truncation_seconds:.2f}s")
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

| Supplied transcription | `enable_diarization` | Behavior |
|---|---|---|
| Has speaker labels | any | Supplied speakers are used; the flag is ignored |
| No speakers | `True` | pyannote runs on the audio and attaches speakers to the supplied words |
| No speakers | `False` | Used as-is; all segments share one voice clone |

The diarize-on-supplied path needs word-level timings, so transcriptions loaded from SRT
(one synthetic word per block) are rejected.

## Pick the translation model

Translation goes through `OllamaTranslator`, a single local Ollama text model. It sends
segments under a structured-output schema and reads back length-budgeted translations —
the prompt carries a per-segment character budget derived from the source duration and a
`low_confidence` hint sourced from Whisper's `avg_logprob`. Long sources are chunked to
fit the context window, with one parse-retry for segments the first pass misses.

```python
dubber = VideoDubber(
    translator_model="qwen3.6:27b",
    translator_host="http://localhost:11434",
)
```

Segments the model never returns land on `result.translation_failures` as indices, with
empty translated text. Any language pair is attempted — the pipeline does not reject a
target language up front.

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

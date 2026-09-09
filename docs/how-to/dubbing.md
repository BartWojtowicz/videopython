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
the source, and the dubbed audio is gain-matched to the source; see the
[file-output reference](../reference/ai/dubbing.md#videodubber). Add `keep_original_audio=True` to retain
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
dubber = VideoDubber(no_speech_threshold=0.4)      # lower the no-speech probability cutoff
dubber = VideoDubber(vocabulary=["Klarna", "Allegro", "InPost"])  # brand-name biasing
```

`turbo` is the default. `condition_on_previous_text=False` reduces propagation of
incorrect text from one decoder window to the next. Details in [AI understanding](../reference/ai/understanding.md#audiototext).

## Reject poor transcripts before synthesis

Degenerate audio (ambient music, near-silence read as speech) produces unusable
transcripts. Every result carries a heuristic assessment:

```python
q = result.transcript_quality
if q is not None:
    print(q.recommendation)            # "ok" | "warn" | "reject"
    print(q.flags)                     # ["dominant_phrase", ...]
    print(q.dominant_phrase_fraction)
```

See [TranscriptQuality](../reference/ai/dubbing.md#transcriptquality) for flag thresholds.

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
lets you inspect speaker labels before translation:

```python
from videopython.ai import AudioToText

transcription = AudioToText(enable_diarization=True).transcribe(video)

for seg in transcription.segments:
    print(seg.speaker, seg.start, seg.end, seg.text)

dubber.dub_and_replace(video=video, target_lang="es", transcription=transcription)
```

| Supplied transcription | `enable_diarization` | Behavior |
|---|---|---|
| Has speaker labels | any | Supplied speakers are used; the flag is ignored |
| No speakers | `True` | pyannote runs on the audio and attaches speakers to the supplied words |
| No speakers | `False` | Used as-is; all segments share one voice clone |

If you correct the transcript, keep segment text and timed word text consistent.
Phrase splitting relies on their alignment.

The diarize-on-supplied path needs word-level timings, so transcriptions loaded from SRT
(one synthetic word per block) are rejected.

## Pick the translation model

Use an Ollama text model that supports structured output. The translator needs no
vision capability. See [OllamaTranslator](../reference/ai/dubbing.md#ollamatranslator)
for request budgets and model residency.

Select a model already downloaded in Ollama. For example, run
`ollama pull translategemma:12b`, then configure the dubber:

```python
dubber = VideoDubber(
    translator_model="translategemma:12b",
    translator_host="http://localhost:11434",
    low_memory=True,
)
```

Check both `result.translation_failures` and `result.synthesis_failures`. Their indices
refer to the original source transcription. No failures means that outputs were
available; it does not prove correct translation or speech.

Review meaning, numbers, names, speaker alignment, and the ends of long phrases.
Use `result.timing_summary` to find excessive speedups. Current measured limitations
are in the [verification record](../reference/verification.md#final-dubbing-review-0612).
The [pipeline explanation](../explanation/dubbing.md) describes phrase timing and
why successful generation can still need listening review.

## Swap the TTS backend

Supply an object with the `generate_audio` method below to
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

The method returns `Audio`. Voice sample paths take precedence over in-memory
samples. Expression arguments may be `None`. A backend can also provide `unload()`
for cleanup in low-memory mode. The example is an interface sketch; implement its
method before use. The `ai` extra still installs the local backend dependencies.

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

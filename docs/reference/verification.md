# Verification records

These point-in-time measurements support release checks and implementation decisions.
Older entries include candidate designs that were later replaced. The
[final dubbing review](#final-dubbing-review-0612) describes the latest recorded
release result; API behavior belongs in the reference pages. These records
describe the tested environment and are not performance guarantees for other
hardware, inputs, or dependency versions.

For the interfaces covered by the AI checks, see [AI generation](ai/generation.md),
[AI understanding](ai/understanding.md), and [Dubbing](ai/dubbing.md). For the design
decision supported by the effects profile, see [The streaming
engine](../explanation/streaming-engine.md#why-pixel-effects-are-not-ffmpeg-filters).

## Catalog keyframe extraction

On 2026-09-09, catalog construction was checked on `cam1_10min.mp4`: 599.8 s,
1280×720, 25 fps, SHA256
`deeaa2055a9061ea04fdddafdcc846be0454c677cabc5e1d5c546b595d4e1e7b`.
The saved analysis used 24 fixed 25-second ranges to exercise a large catalog;
these were test ranges, not model-detected scene boundaries. No analyzer ran.
Environment: Linux under WSL2, Python 3.12.12, FFmpeg 6.1.1.

The baseline was commit `60142cf`. Each run built the MCP catalog, then requested
its final scene twice. Process-tree RSS was sampled every 20 ms across the Python
process and its children. Metadata was warm in each run: zero `ffprobe` processes.
The initial response contained 12 images and an omitted-ID note.

| Measurement | Baseline | Batched, no image cache | Batched, 12-image cache |
|---|---:|---:|---:|
| Initial response | 415.00 s | 9.17 s | 9.14 s |
| Initial FFmpeg invocations | 24 | 1 | 1 |
| Peak process-tree RSS over all requests | 262.81 MiB | 238.16 MiB | 236.45 MiB |
| Retained image arrays after initial response | 63.28 MiB | 0 MiB | 11.39 MiB |
| First final-scene request | 0.089 s | 16.25 s | 16.86 s |
| Repeated final-scene request | 0.088 s | 16.48 s | 0.089 s |

Catalog JSON and every initial PNG payload had identical SHA256 values across
all three runs. The synthetic RGB test also compared unsorted and duplicate
requests with a full decode, and the MCP test retrieved an omitted scene, checked
the image-size/cache bounds, and repeated the request without extraction.

The 12-image cache was selected because an uncached request near the end still
requires a long sequential decode. Cache entries are downscaled and independently
owned, so they do not retain the full extraction batch. The local planner still
receives full-resolution frames. No sparse-seek strategy was added.

These are single observations. Focused tests overlapped part of the baseline run;
the changed measurements ran alone. The timings are not a controlled speedup
estimate. RGB equality establishes unchanged keyframes, not editorial selection
quality. The local scripts, saved analysis, payload hashes, and logs are under
`.cache/catalog-keyframes/` and are not distributed.

## Editing recipes

On 2026-09-09, the caption and branding examples rendered seconds 0–6 of
`cam1_1min.mp4` at 640×360 and 360×640. The caption example used the saved Polish
word transcription. The title was “Początek roku: rozmowa o biznesie, planach i
nowych możliwościach”, with a 24-pixel font and a caller-supplied SVG mark. Frames
at 3.2 s were inspected: Polish text was readable and wrapped within the margins,
and the mark stayed at the top left in both orientations. Center cropping fit
this shot; it is not a general subject-framing guarantee.

The two-pass summary rendered ranges 8–12 s then 0–4 s from the same source at
both sizes. It used a local synthetic chord bed with gain 0.490981 and duck 0.8.
Both final files were 8 s long with the requested dimensions. The saved transcript
was mapped to cut order before the second pass. All plans passed JSON round trips
and validation. Outputs and review frames remain local in `.cache/editing-recipes/`.

The automated two-pass check used red/blue video, a 440 Hz tone, and two timed
words. It verified reversed visual order, copied source words, mapped word times,
a nonzero bed, and speech-window RMS below 40% of the pause level. This measures
ducking behavior. The user listened to the real-clip sample at gain 0.35 and found
the music only slightly too quiet. After a first increase to 0.39, the user requested
another 2 dB. The final gain is `0.39 * 10 ** (2 / 20)`, or 0.490981, with the same
ducking settings. The user accepted the latest sample. The mix had peaks above full scale, so
the final review exports also use an audio-only finishing pass with overall gain
0.89 and video stream copy. Float PCM decode measured a peak of 0.90625 in both
orientations, below full scale.
This finishing gain preserves the adjusted music-to-speech ratio.

The recipe is limited to frame-aligned cuts from one source with no transitions
or retiming. It requires an intermediate file and a second encode. No model ran,
and these checks do not assess transcription accuracy or music suitability.

## Speech candidate selection

On 2026-09-09, speech selection used fresh audio from seconds 300–390 of
`all_in_30min.mp4`, outside the translation comparison range at 750–1050 s.
The audio was paired with a static 640×360, 25 fps video for render checks.
The resulting source SHA256 was
`74f29ad82c870627ccd40dc1dd7404d871647168329709d0b5ff1ad44b3d5250`.
No scene detector, planner, translator, or synthesis model ran.

Transcription used the cached Whisper turbo revision
`0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf` on CUDA, with VAD disabled and the
library's other defaults. The recorded transcription-plus-catalog interval was
10.78 s. The analysis supplied one visual scene; speech settings were 5 s minimum,
20 s maximum, and a 0.8 s pause. This was a boundary check, not a speed comparison.

Nine candidates were produced, from 5.08 to 13.60 s long. All nine resolved,
validated, rendered, and passed full FFmpeg decode. Their boundaries did not pass
through any supplied word interval. Ranges did not overlap, and the full retrieved
transcripts had no exact duplicates. Source timestamps remain ASR estimates;
these checks do not establish acoustic alignment accuracy.

Transcript review found a complete response on company innovation, several units
that depend on preceding context, a candidate that changes topic between complete
sentences, and a final unit that introduces an explanation without including it.
The first candidate starts mid-question because the supplied 90-second excerpt
starts there. Punctuation and pause rules do not establish standalone meaning.
There was no listening score or claim of automatic editorial quality.

The synthetic suite additionally checks multiple candidates from a single shot,
passages crossing visual cuts, overlapping words, zero-duration word ownership,
missing alignment, impossible limits, repeated builds, identical file stems,
changed-setting ID rejection, full MCP transcript retrieval, and a selected render.

Two preliminary checks were retained separately. `cam1_10min.mp4` repeated the
previously reviewed minute, so it was not counted as fresh content. The selected
`dreams_15min.mp4` excerpt mixed sparse Japanese speech with an English language
detection result and zero-duration words; its transcript was not used as a quality
baseline. Zero-duration words retain their supplied times without invented length.
Scripts, transcripts, exports, and per-candidate notes remain local under
`.cache/speech-candidates/`. No model downloads or paid APIs were used.

## MCP workflow verification

The stdio workflow passed on 2026-09-06 at commit
`8af4f71754a0d16859475093dcdfea8d31fe2b72`, using Python 3.13.5 on a 16 GB M1 Mac
mini with macOS 14.8.9 and Ollama 0.33.3. It used the representative Polish clip from
the AI check and the locally available `gemma3:12b` caption model.

| MCP call | Elapsed | Result |
|---|---:|---|
| `analyze_video` | 140.224 s | One scene; speech, scene detection, captioning, and face tracking completed |
| `build_catalog` | 2.337 s | One scene with a caption, Polish transcript, speech flag, and face flag |
| `validate_edit` | 0.039 s | The one-scene plan was valid with no errors |
| `run_edit` | 10.350 s | 60.08-second 1280×720 H.264/AAC MP4 with 1,502 frames |

The full session took 154.153 seconds. Manual review confirmed the caption and rendered
shot; FFmpeg decoded both output streams without an error. Audio classification was
disabled and reported as such.

The model files were present before the measured run. Reproduce it with the real stdio
client and server harness:

```bash
OLLAMA_HOST=127.0.0.1:11434 uv run python scripts/verify_mcp_workflow.py \
  --source verification-input/cam1_1min.mp4 \
  --workdir verify-results/mcp-gemma3-12b \
  --vision-model gemma3:12b
```

## AI model verification

### Final dubbing review, 0.61.2

On 2026-09-09, the listener approved the one-minute Polish-to-English demo with
TranslateGemma 12B and 5 ms phrase-boundary fades. The fades preserve phrase
anchors, sample counts and interior samples. The demo used two cloned voices,
background preservation, seed 1777 and `low_memory=True` on an RTX 2060 SUPER
with 8 GB VRAM. It reused the saved diarized transcription.

Generation and export took 176.54 seconds. All ten phrases had speed factor 1.0,
with no reported translation or synthesis failures. The H.264 video and single
default AAC audio track fully decoded. Independent ASR recognized 141 words in
both raw and faded speech, matching the expected count. Both checks recognized
“health” as “hell”; equal word counts do not establish exact pronunciation.
The listener accepted the recording, rather than providing a phonetic audit.

The model comparison also covered 72 timed phrases from seconds 750–1050 of an
English podcast, translated into Polish. Assistant text review favored the 12B
library configuration among the tested Qwen3.5 9B, Hy-MT2 7B and TranslateGemma
4B/12B configurations. Remaining errors include units, names, financial terms
and duplicated context. This is a local qualitative comparison, not an independent
human ranking. The default `qwen3.6:27b` was not compared and remains unchanged.

The 12B library translation run took 535.34 seconds with no structural failures.
It used CPU offloading; these settings are a compatibility check, not a default-model
performance baseline. No new ten-minute performance comparison was completed.
Further tuning needs fresh validation material because the podcast cut was also
used for prompt diagnostics. The [dubbing guide](../how-to/dubbing.md#pick-the-translation-model)
shows the tested model configuration.

The maintained real-model harness then passed `env`, `imports` (38 entrypoints),
`ollama` and `dub` with the same 12B override. This fresh run included transcription
and diarization, produced two voice samples and ten translated phrases, and reported
no translation/synthesis failures or excessive speeds (maximum 1.0×). The dub check
took 249.31 seconds; all four checks took 263.04 seconds. This run used the harness
seed behavior and did not reuse the approved demo's frozen transcription.

```bash
OLLAMA_HOST=127.0.0.1:11435 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python scripts/verify_ai_models.py \
  --only env,imports,ollama,dub --video cam1_1min.mp4 \
  --workdir verify-results/release-0.61.2 \
  --ollama-model translategemma:12b --low-memory \
  --source-lang pl --target-lang en --enable-diarization
```

Final release preparation passed pre-commit, lock validation, strict documentation
build, wheel/sdist builds and clean-wheel public-import/render/MCP smoke checks.
The functional code had passed the full 1,285-test suite. The final source cleanup
removed a module docstring; all 17 translator tests passed afterward. Package
contents contain no experiment scripts, media, model caches or private instructions.

#### Fresh two-language sanity checks

After the review fixes, both recordings ran through fresh transcription, diarization,
per-speaker cloning and background preservation with TranslateGemma 12B, seed 1777
and `low_memory=True`. Both exports decoded and reported no translation or synthesis
failures. These checks show that generation success does not establish speech quality.

| Input | Direction | Elapsed | Phrases | Flagged speedups | Maximum speed |
|---|---|---:|---:|---:|---:|
| One-minute conversation | Polish to English | 171.04 s | 10 | 0 | 1.04× |
| Five-minute podcast excerpt | English to Polish | 1,142.58 s | 67 | 49 | 8.4× |

The conversation's translations matched the approved demo, but independent ASR
flagged possible extra speech around 37 seconds in raw and mixed audio. In the
podcast, ASR recognized a property-sale sentence in raw speech but missed it after
3.16× acceleration and in the final mix. The maximum 8.4× adjustment fit a 2.52-second
“Tak.” output into a 0.30-second source window. Raw Polish speech also had number
recognition errors. These differences require listening to distinguish synthesis
and intelligibility errors from recognition errors.

The podcast text review found unit substitutions, duplicated neighboring content,
and changes to financial actions. The fresh excerpt produced three speaker labels;
an earlier full-recording run assigned four labels within that same interval. Neither
count is a verified count of people. The listener also reported inconsistent voices
and dramatic speedups. This remains a difficult quality case, not a clean quality
pass or evidence of a regression caused by the review fixes.

### Dubbing decoder optimization, 0.61.2

On 2026-09-08, full Polish-to-English dubbing of `cam1_10min.mp4` was measured
on an RTX 2060 SUPER (8 GB), Python 3.12.12, PyTorch 2.13.0+cu130,
videopython-chatterbox 0.1.7.post1 and pyannote-audio 4.0.7. Both variants include
the 0.61.1 diarization optimization. The isolated candidate adds only CUDA graph replay
for the existing decoder's feed-forward and normalization operations.

The comparison calls `VideoDubber.dub_file` with diarization, per-speaker voice
cloning, background preservation, original-audio retention and `low_memory=True`.
It includes decoding, model loading, transcription, diarization, separation,
translation, synthesis, synchronization, mixing and final video output. Imports
and downloaded model files were warmed before timing. Local instrumentation saves
intermediate outputs in both variants and adds some overhead.

Translation uses the same Ollama 0.33.3 `qwen3.5:4b` model and seed in both runs.
This smaller model makes the benchmark practical on this GPU; it does not replace
the library default or establish translation parity with that larger model.
The dedicated Ollama server uses `OLLAMA_KEEP_ALIVE=0`. Python, NumPy and PyTorch
seeds are reset identically before each synthesis call.

| Measurement | Before | Decoder graphs |
|---|---:|---:|
| Full dubbing, including final video output | 656.77 s | 568.12 s |
| Speech synthesis calls, including model loading and reference preparation | 508.00 s | 417.86 s |
| Peak GPU memory (whole device, sampled every second) | 7,338 MiB | 7,806 MiB |
| Peak process RSS | 6,993,460 KiB | 6,992,940 KiB |

Full time fell by 13.5%; synthesis time fell by 17.7%. The additional GPU memory
is a material tradeoff on small GPUs. Capture failures restore the original path.
Ninety decoder graphs remained active throughout all 21 synthesis calls.

Both runs used the same two speakers, selected reference samples, translated text,
per-segment expression and seeds. All 21 synthesis calls succeeded with identical
sample counts. Maximum absolute waveform difference was 4.04e-6 on normalized
floating-point audio; outputs are not bit-identical. Translation had no failures,
and synchronization summaries matched, with no truncated segments.

Input SHA-256:
`deeaa2055a9061ea04fdddafdcc846be0454c677cabc5e1d5c546b595d4e1e7b`.

These are single full-pipeline runs on one recording, supported by repeated short
synthesis probes. They do not establish universal speedups or bitwise equivalence
on other hardware, recordings or dependency versions. Reference-cache and attention-
observer cleanup experiments improved full time by only 0.6% together and were
excluded from the release because their benefit did not justify the extra code.
Experimental benchmark helpers and intermediate media are kept outside the commit.

### Dubbing reliability, 0.61.2

These entries record successive candidates. Truncation and slowdown behavior below
was superseded by the final source-word phrase scheduling. Measurements remain
attached to the candidate that produced them.

Follow-up checks on 2026-09-08 used the same GPU environment and explicitly selected
`qwen3.5:4b`; the library default remains unchanged. These checks include bounded
translation and synthesis and are separate from the decoder-only comparison above.

| Check | Result |
|---|---|
| Full `cam1_10min.mp4` dubbing | 747.608 s; 21/21 turns generated, no translation or synthesis failures, no timing truncation |
| First 10 minutes of `all_in_30min.mp4` | Recovery generated 42/42 positive-duration turns; one zero-duration ASR fragment reported as a synthesis failure |
| Long Polish synthesis | 235 words, ten bounded calls, 105.44 s of raw audio; normalized ASR matched every source word, including the closing sentence |
| Export validation | Both final MP4 files fully decoded with FFmpeg; cam1 retained video, and the audio-only All-In input retained dubbed and original audio |

The cam1 run preceded the final short-alignment and vocabulary guards. The final
long-speech probe exercised those guards. All-In's first full run failed on short
alignment and invalid vocoder tokens; recovery reused successful transcription,
translation and speech and regenerated failed or unfinished turns. Its final
127.093 s recovery/reassembly time is **not** a full-pipeline benchmark. A clean
full All-In run with every final guard has not been measured.

This is a robustness improvement, not a translation or timing quality pass.
The tested 4B model still reversed meanings and mishandled idioms. All-In timing
adjustment truncated 33 of 42 generated turns; final ASR confirmed missing closing
speech. Raw long-turn synthesis retained its endings. That candidate's reported
truncation-seconds metric also included time-stretch savings, so it overstates
actual tail removal. The review follow-up below records the next changes.

The implementation suite passed 1,265 tests with CUDA hidden, plus lint, typing
and strict documentation checks. Experimental scripts, intermediate transcripts,
and media remain local and are not shipped in the release.

#### Review follow-up

The initial reliability candidate was 13.8% slower than the 656.77 s baseline and
31.6% slower than the graph-only candidate. Saved stage timings attribute the
increase to both translation (40.00 → 115.24 s) and synthesis (417.86 → 518.37 s).
These are different generated workloads; the figures do not isolate individual
code changes or measure the final branch.

A fresh full cam1 run with the review fixes completed in **702.922 s**: **7.0%
slower than the 656.77 s baseline**, and 6.0% faster than the initial reliability
candidate. Translation took 47.119 s; synthesis took 541.291 s across
51 bounded backend calls. All 21 turns generated successfully, with no translation
or synthesis failures, no timing truncation and no excessive speed. Peak sampled
whole-device memory was 7,161 MiB, versus 7,338 MiB in the baseline.

These remain **4B-model compatibility measurements**, not default-model performance
baselines. The final cam1 output contains 378.64 s of raw speech and 1,326 translated
words, versus 332.48 s and 1,299 words in the baseline. Changed generated workloads
prevent attributing the net difference solely to execution overhead. Cam1 translation
still mishandles the idiom “sezon ogórkowy” as “pickling/cucumber season”.

The fresh full ten-minute All-In run completed in **1,189.936 s**. All 41 usable
turns generated successfully across 81 bounded calls. Original indices 34 (20 ms)
and 37 (zero duration) were reported in `synthesis_failures`. There were no
structural translation failures or hard timing truncations. Translation took
88.473 s; synthesis took 957.863 s and produced 637.48 s of raw speech. Peak sampled
whole-device GPU memory was 7,157 MiB. This is a complete run, unlike the earlier
recovery/reassembly measurement.

**All-In still fails quality.** Twenty-one of 41 turns exceeded the preferred speed;
mean speed was 2.568× and maximum speed was 16.840×. Inspecting the worst short
outputs found extra speech in raw TTS output, not merely silence: “Ale” generated
5.94 s, “Tak.” 4.74 s, and “to.” 3.44 s, with ASR detecting unrequested words. Faster
timing preserves the generated waveform but cannot repair hallucinated speech or
make those extreme speeds intelligible. The 4B translator still reverses “the market
is ripping” into a falling market and confuses valuation multiples with revenue growth.

Both final MP4s fully decoded with every video/audio stream explicitly mapped.
Cam1 contains H.264 video plus dubbed/original AAC; All-In contains two AAC tracks.
Final-audio ASR retained both closing sentences, including All-In's previously lost
“trochę szalony”. Raw longest and closing turns were also transcribed. This checks
selected coverage risks, not every word or overall listening quality.

The release verification script's `dub` check also passed on a 68.28 s cam1 extract
with `qwen3.5:4b`, CUDA, low-memory mode, diarization and cloning: three turns, two
speakers, no translation/synthesis failures, no truncation or excessive speed. The
script now honors the model override, exposes low-memory mode, reports excessive
speeds and fails on synthesis failures. This is an operational compatibility check;
it does not contradict the semantic failures above.

The full suite passed **1,275 tests** with CUDA hidden; typing passed all 144 source
and test files. The final translator/Ollama checks passed another 27 tests after
test-typing cleanup. The live Ollama residency probe confirmed the model was loaded
between calls and absent after explicit unload.

The review fixes bound tiny groups to four turns and ten seconds, reject isolated
sub-100 ms groups with original-index failure reporting, split unbroken text, and
restore soft spoken-length hints. Translation explicitly retains Ollama between
requests and unloads it at the low-memory stage boundary. Requests still isolate
one source part at a time; this trades batching throughput for segment ownership.

That candidate borrowed following silence and preserves the entire generated utterance
by allowing speeds above its preferred 1.3× maximum. Excessive speeds are reported
in the timing summary and logs. This avoids deliberate tail clipping, but does not
guarantee natural delivery or correct translation. Residual tempo-filter duration
errors are fitted by resampling the entire output, with a possible small pitch shift.

Two fresh-process CUDA pool probes each generated “za”, “Ja” and a full Polish
sentence twice. Shared pools reserved 178 MiB less than separate pools on all six
calls, with matching sample counts and maximum waveform difference below 8e-7.
Timings varied and do not establish a pool-sharing speedup. A separate GPU probe
verified replay under `no_grad` after capture in inference mode.

At this stage the short-alignment dependency fix was local Chatterbox commit `57fb312`
(version 0.1.7.post2). Seven direct PyTorch boundary tests and the GPU probes above
passed without the alignment monkeypatch. At that stage publication was deferred:
the review candidate retained `>=0.1.7.post1` and the compatibility guard. The full
cam1 and All-In measurements above used that configuration.

#### Published Chatterbox integration

On 2026-09-08, the published `videopython-chatterbox==0.1.7.post2` wheel was
installed and the dependency minimum and lock updated. The local alignment
monkeypatch and its two wrapper tests were removed; vocabulary masking and vocoder
token validation remain separate protections.

Seven direct PyTorch tests passed against the installed dependency's alignment
implementation, covering text widths 1–6 and 20. Seeded CUDA voice cloning produced
“za” (0.96 s) and “Ja” (0.64 s), with the native alignment method unchanged after
model initialization. A 235-word Polish synthesis produced 105.44 s across ten
backend calls; all 235 normalized words matched independent ASR, including the
closing sentence. These examples do not establish that the short-utterance
hallucinations in the All-In record are fixed.

The English TTS harness initially exhausted GPU memory when loading Whisper while
the synthesis model was still referenced. Explicitly unloading TTS before ASR
resolved this: the rerun passed with 0.818 word-set overlap and generated the cloned
voice sample successfully. The recognized sentence differed in “riverbank” versus
“river bank”; this overlap metric is not a complete semantic or cloning-quality audit.
The core suite passed all 1,273 tests (the two obsolete wrapper tests were removed).
Ruff, formatting, mypy, lock validation, wheel/sdist builds and strict documentation
build passed. The wheel contains the token protections, excludes the alignment
monkeypatch and requires the published post2 dependency.

These are functional integration checks, not new full-pipeline benchmarks. The
cam1 and All-In performance and quality results above still describe the earlier
dependency configuration.

#### Comparison against merged 0.61.1

On 2026-09-08, the 0.61.2 branch was rebased onto merged `main` at `3ca57b9`.
`git range-diff` confirmed all four release patches were unchanged; the candidate
tested here is `f9b9c37`. Separate processes imported each checkout, using the same
RTX 2060 SUPER, PyTorch 2.13.0+cu130, published Chatterbox 0.1.7.post2, reference WAVs,
expression settings and per-case seeds. Using post2 on both sides isolates the
videopython changes; this is not a comparison of the two historical lockfiles.

| Matched check | Merged main | 0.61.2 candidate |
|---|---:|---:|
| 235-word Polish passage: generated duration | 30.22 s | 105.44 s |
| Passage: normalized ASR word edit distance | 216 | 0 |
| Known short-utterance hallucinations reproduced | 3/3 | 3/3 |
| Same 41 All-In raw utterances: timing reports tail removal | 31/41 | 0/41 |

The long passage retained all 235 words on the candidate. Main's ASR returned 112
words, including repeated invented phrases and an unrequested closing, so that
word count does not represent 112 correctly retained source words. The short cases
requested “Ale”, “Tak.” and “to.” with original All-In seeds 1793, 1805 and 1812.
Both versions produced 5.94 s, 4.74 s and 3.44 s, respectively, with the same
unrequested speech in ASR. Saved PCM samples differed by at most one 16-bit step.

The timing comparison reused identical cached speech in the original turn windows,
without pipeline gap borrowing. The candidate reported 23 excessive speeds, peaking
at 35.63×; this is not the earlier full-pipeline maximum of 16.84×. Preserving all
samples at such speeds does not establish intelligibility. In a separate closing-turn
check (22.56 s fitted to 17 s), both versions retained the closing phrase in ASR;
word edit distances were 8 on main and 10 on the candidate. That example does not
show a transcription-quality improvement from faster timing.

Paired translation checks used the existing `qwen3.5:4b`, seed 1777 and default
translator settings on cam1 segments 0–2 and All-In segments 30–33. Both versions
returned all seven segments without structural failures. Both reversed “the market
is ripping” into “rynek się wali” (the market is collapsing). For “sezon ogórkowy”
in the January pizzeria discussion, main returned “summer season” and the candidate
returned “pickling season”; neither preserved the quiet-business-period meaning.
These examples do not demonstrate better semantic translation on the candidate,
and do not evaluate the default 27B model. No model downloads were needed.

These targeted checks establish improved long-speech fidelity and avoidance of hard
tail cuts, not uniformly better dubbing quality. They are not new full-video latency
benchmarks or listening tests. Post-rebase checks passed 160 dubbing/speech tests
and 17 translator tests. Raw WAVs, recognized text, settings and per-turn timing
records are retained locally under `.cache/dubbing/main-quality-comparison/`.

#### Listener follow-up: early finishes on cam1_1min

The listener reported stretch artifacts and a long silence near 38 seconds in both
one-minute dubs. Both had clamped all three turns to 0.8×. Capturing the candidate's
raw speech reproduced durations of 4.76 s, 25.08 s and 6.60 s against source windows
of 7.16 s, 39.64 s and 11.48 s. The second turn begins at 8.02 s: the old minimum
speed left roughly eight seconds of unused time before its 47.66 s boundary.

That candidate treated the minimum as a preference and reported excessive slowdowns,
as it did for excessive speedups. A corrected mix reused identical raw
speech, translations, source audio and background; only timing changes. Rubber Band
at 0.665×, 0.633× and 0.575× fills all three windows. The second turn's ASR ending
moves from 23.82 s in the raw clip to 37.66 s in the corrected clip, or approximately
45.68 s on the video timeline. Some natural trailing silence remains before 47.66 s.

This fixes the early-finish mechanism but does not establish artifact-free speech.
Normalized ASR word edit counts for raw versus corrected turns were 2/2, 3/9 and
0/0. The old 0.8× atempo version of the middle turn produced 7 edits while leaving
the long unused tail. At the full required duration, atempo produced 11 edits;
Rubber Band's default and long-window settings produced 10 and 11. The selected
smooth-transient setting produced 9. ASR is an imperfect proxy for perceptual
quality; the new listening sample still requires review. Existing translation
errors and short-input hallucinations are unaffected.

Pitch/duration tests cover slowing and accelerating a 220 Hz signal, including
activity near the output ending. The focused audio/dubbing suite passed 220 tests;
lint and mypy passed. Both output streams fully decoded. Samples and comparison
records are retained under `.cache/dubbing/cam1_1min_listen/`, including
`cam1_1min-0.61.2-pacing-fixed-en.mp4`.

#### Source-word phrase scheduling replaces paragraph slowdowns

The listener rejected the full-window slowdown as unnatural. The next design keeps
speaker turns for diarization/reference extraction, but derives dubbing phrases from
validated word timestamps before translation. Sentence boundaries take priority over
commas; pauses and clauses help bound longer runs. Tiny tails are kept with their
neighbors. Original source segments are preserved and phrase failures map back to
them through `source_segment_index`. Missing or inconsistent word timing retains
the original segment rather than inventing proportional timestamps.

On `cam1_1min.mp4`, three original turns became ten phrases; the middle turn became
six sentences anchored at 8.02, 13.34, 19.30, 26.44, 33.64 and 39.68 seconds. Shorter
generated phrases keep their natural pace; available gaps absorb overruns before
acceleration. That experiment used a preferred range of 0.9–1.1×. Larger necessary speedups
remain reported to preserve complete generated speech. Forced slowdowns are removed.

The new full-pipeline review video used the same existing 4B translator, post2
Chatterbox, seed policy, diarization, speaker cloning and background preservation.
All ten phrases ran at exactly 1.0×: zero stretches, truncations, excessive speeds,
translation failures or synthesis failures. A separate check under the new 0.9/1.1
defaults confirmed every synchronized waveform was sample-identical to raw TTS.
Both exported streams fully decoded. This removes time-stretch artifacts from this
sample by avoiding time stretching entirely, rather than selecting another filter.

Final ASR returned 145 words against 145 translated words, with three word edits:
the negation in “can't say” became “can say”, “health” became “hell”, and an extra
“Umm” appeared. These require listening review; successful scheduling is not proof
of exact spoken meaning. The 4B translation still mistranslates the quiet-season
idiom. Late phrases remain anchored throughout the final part of the video, with
speech recognized through approximately 59 seconds.

Validation passed 184 focused dubbing, phrase, translation and speech tests, plus
lint, typing and strict documentation checks. Artifacts are retained under
`.cache/dubbing/cam1_1min_phrases/`; the review output is
`cam1_1min-phrases-en.mp4`. This is a pacing/quality check, not a controlled new
full-video performance benchmark or phoneme-level lip-sync claim.

The listener accepted this phrase-based version for 0.61.2 with the remaining
sentence pauses. A subsequent placement experiment was discarded: release code
keeps the original phrase anchors and introduces no accumulated start-time shifts.
Final local release checks passed 1,283 tests, Ruff/formatting, mypy, lock validation
and strict documentation build. Wheel and sdist builds passed; a clean wheel install
passed public-import, render and MCP smoke checks. The wheel requires published
Chatterbox post2 and excludes the discarded placement experiment.

### Diarization optimization, 0.61.1

On 2026-09-08, the diarization embedding path was measured on an NVIDIA GeForce
RTX 2060 SUPER, Python 3.12.12, pyannote-audio 4.0.7, and PyTorch 2.13.0+cu130. The patch skips inactive
speaker/chunk pairs and shares frame extraction when the pyannote embedding backend
provides the compatible split-frame interface in evaluation mode with zero dither.
This is an internal optimization; videopython does not select an identity model or
depend on the downstream `wespeakerruntime` package.

Both versions used the same `Audio.from_path(..., sample_rate=16000, channels=1)`
decode. Precision, segmentation overlap, and clustering settings were unchanged.
Each variant ran three times per loaded model. The following averages use runs 2
and 3, exclude loading and decoding, and include final annotation construction.
"Before" already skips inactive pairs; "after" additionally shares frame extraction.

| Recording | Before | After | Time reduction | Speakers / exclusive turns |
|---|---:|---:|---:|---:|
| `cam1_10min.mp4` | 6.46s | 5.22s | 19.3% | 2 / 146 |
| `all_in_30min.mp4` | 21.18s | 16.16s | 23.7% | 5 / 640 |

Exclusive and overlap-aware labels and exact unrounded timestamps matched in every
comparison run on both recordings, and their RTTM files were byte-identical. Maximum
absolute embedding differences were 3.04e-6 and 1.73e-6 respectively. This establishes
unchanged output on these inputs, not accuracy against human annotations or a guarantee
for other recordings. The 30-minute clip had 686 overlap-aware turns.

Smaller embedding batches (16 and 24), convolution/batch-normalization fusion, and
channels-last layout gave no useful improvement over shared frame extraction on the
10-minute clip. Reduced segmentation overlap was not adopted because it changes output.

These measurements used a local experimental harness, not a maintained repository
tool. It timed the complete pyannote call with CUDA synchronization and recorded
exclusive and overlap-aware RTTM, unrounded turn timestamps, and embeddings.
For an equivalent comparison, disable frame sharing while retaining inactive-pair
skipping in the reference run, use identical decoded audio, and compare exact
annotations as well as RTTM files. Input SHA-256 values:

| Recording | SHA-256 |
|---|---|
| `cam1_10min.mp4` | `deeaa2055a9061ea04fdddafdcc846be0454c677cabc5e1d5c546b595d4e1e7b` |
| `all_in_30min.mp4` | `b95a8afe03c39529939ddd0343b3490d665b745ebfb758fee73ef61bb1ee91a6` |

A single stage-timed run on the 30-minute clip with default Whisper turbo (float32),
VAD, automatic language detection, and diarization produced 5,655 English words and
five speakers in 105.24s: decode 10.18s, VAD including initialization 16.03s, language
detection 0.90s, transcription 61.66s, diarization 16.44s, and word processing/speaker
assignment 0.03s. Whisper and diarization loading added 9.92s, for 115.16s including
those loads. Python startup/imports and output writing are outside these totals.
The experimental harness also omitted final transcript regrouping and confidence
reattachment, so these are sums of the measured stages, not exact public-API
end-to-end wall times.

The real-model verification harness also passed `env` and `imports` (38/38 public
AI entrypoints) on this machine. Speaker identity matching remains downstream:
anonymous diarization labels do not make internal cluster embeddings compatible with
an application's enrolled voice embeddings.

The dependency range is restricted to pyannote-audio `>=4.0.7,<4.1` because these
optimizations override private pipeline steps. Before widening it, recheck inactive
pair filtering, clustering exclusion, the split-frame computation, and exact output
comparisons. Method availability alone does not establish those semantics.

### CPU diarization reconstruction comparison

On 2026-09-07, pyannote's original reconstruction and the `0.60.1` implementation at
commit `1ae2b789c5c91e64bd419614208c3ab578ec7163` ran against the same 60.024-second
audio and 121 timed words on an M1 CPU.

| | Original | `0.60.1` |
|---|---:|---:|
| Reconstruction workspace | 478.6 KiB (`float64`) | 239.3 KiB (`float32`) |
| Wall time | 43.708 s | 42.347 s |
| Process peak RSS | 3,036.9 MB | 3,049.3 MB |

Both runs produced the same two speakers, four segments, and word labels. The exact
workspace is 50% smaller. Process RSS is model-dominated at this input length.

Environment: macOS 14.8.9, Python 3.13.5, pyannote-audio 4.0.7, PyTorch 2.13.0,
NumPy 2.4.6. Input SHA-256:
`472540f20091958d5283f26701927e0cf0ea193f35c4d5a3e70ac0ae905d8d66`.

### Earlier full AI verification

The real-model harness in `scripts/verify_ai_models.py` passed with public defaults on
2026-09-06. It used library commit
`3023da1ffc20254abd91c4f8b3005925f0ca4e19` and a representative 60.08-second Polish
clip. The input was 1280×720 H.264 video with AAC audio. Its SHA-256 was
`4a258bf9eb50a120485399a60768479bec8b72fae2e98de21361c751eff350f0`.

The environment used Python 3.12.3, an NVIDIA RTX PRO 6000 Blackwell Workstation
Edition with 97,887 MiB VRAM and compute capability 12.0, driver 595.71.05, PyTorch
2.13.0+cu130, Diffusers 0.39.0, Transformers 5.14.1, Safetensors 0.8.0, Ollama server
0.33.3, and Ollama Python client 0.6.2.

### Default-setting results and timings

| # | Check | Elapsed | Result | Semantic evidence |
|---|---|---:|---|---|
| 1 | `env` — Environment and package versions | 1.109 s | **PASS** | PyTorch 2.13.0, RTX PRO 6000 Blackwell |
| 2 | `imports` — Public AI entrypoints resolve | 0.328 s | **PASS** | 38/38 entrypoints |
| 3 | `ollama` — Schema-constrained output | 75.632 s | **PASS** | `qwen3.6:27b` returned a valid Spanish translation |
| 4 | `t2i` — Prompt conditions image output | 200.657 s | **PASS** | Same-seed cross-prompt correlation 0.067 |
| 5 | `t2v` — Video renders and moves | 2,112.681 s | **PASS** | 81 frames, 97.1% of pixels moved |
| 6 | `i2v` — Input conditions frame 0 | 691.838 s | **PASS** | Input correlation 0.992, 91.2% of pixels moved |
| 7 | `tts` — Speech is intelligible | 40.995 s | **PASS** | Round-trip word overlap 91%; cloned sample also written |
| 8 | `separation` — Voice reaches vocals stem | 10.795 s | **PASS** | Vocals correlation 0.990 with voice, -0.146 with music |
| 9 | `music` — Output is audible and conditioned | 9.863 s | **PASS** | Peak 0.334, cross-prompt envelope correlation -0.214 |
| 10 | `detect` — Known objects are detected | 2.815 s | **PASS** | Five objects: cat, remote, sofa |
| 11 | `dub` — Full Polish-to-Spanish dub | 73.272 s | **PASS** | 17/17 translated; cloned voice; worst truncation 1.760 seconds |

The complete run took 3,219.985 seconds. Text-to-image generated two 50-step
1328×1328 images. Text-to-video generated 81 frames at 1280×720 and 16 fps with 40
steps. Image-to-video generated 81 frames at 832×480 and 16 fps with 40 steps. The
default dub used voice cloning without speaker diarization and grouped segments under
one `speaker_0` clone.

Manual review confirmed that both images matched their prompts. The text-to-video clip
kept a coherent mountain-lake scene while mist and water moved. The image-to-video clip
kept the bicycle from its source image while a camera push moved it partly out of frame.

### Reproduction

Use [Verify local AI models](../how-to/verify-models.md) for the maintained harness
and cache-warming procedure. The historical runs above used the stated commits,
dependencies, and inputs; running today's harness does not reproduce older code.

### Dubbing synchronization threshold

Speech synthesis is nondeterministic. Four earlier A100 runs with the same one-minute
input established the failure limit.

| Run | Truncated segments | Mean speed factor | Worst truncation |
|---|---:|---:|---:|
| Baseline 1 | 9/17 | 1.072 | 2.460 s |
| Baseline 2 | 8/17 | 1.054 | 1.680 s |
| Baseline 3 | 10/17 | 1.085 | 2.000 s |
| Complete model run | 7/17 | 1.102 | 2.400 s |

These historical runs used the former truncation threshold. The current dub
verification fails on missing timing measurements or translation/synthesis failures.
It reports excessive speeds for listening review; synchronization preserves complete
generated speech.

## Channel layout observations

An earlier local review of a two-person, ten-minute recording reported source-channel
correlation of 0.009 and about 33 dB separation between channel-energy ranges. Two
mono decodes made with different resamplers differed by 0.63% RMS; their diarization
outputs differed by a reported 10.5% diarization error rate, mostly missed speech.

The original notes did not record the source hash, model configuration, reference
annotation, or measurement method. These observations motivate investigation of
channel-aware diarization; they do not establish speaker accuracy or a reproducible
performance result.

## 4K effects performance

This profile measures median processing time for one warmed 3840×2160 frame on an
Apple M1. Each result is the median of seven samples, repeated in three independent
runs. Effects use their defaults except for an active animation frame and a
representative non-default strength or geometry where the default would not exercise
the effect.

The non-default inputs were a 0.6-alpha full overlay, five blur iterations, 1.5× zoom,
`color_adjust` at 0.1 brightness/temperature and 1.1 contrast/1.2 saturation, a
full-to-80% Ken Burns crop, a 0.15-scale/0.8-opacity image overlay, 12-pixel shake, 1.4×
punch-in, 6-pixel chromatic shift, and 16-pixel blocks. `flash` used its active peak and
`fade` its midpoint.

| Effect | ms/frame |
|---|---:|
| `full_image_overlay` | 41.11 |
| `blur_effect` | 4.98 |
| `zoom_effect` | 3.98 |
| `color_adjust` | 16.17 |
| `vignette` | 6.35 |
| `ken_burns` | 4.25 |
| `fade` | 29.14 |
| `image_overlay` | 2.30 |
| `shake` | 5.41 |
| `punch_in` | 3.63 |
| `flash` | 106.20 |
| `chromatic_aberration` | 19.62 |
| `glitch` | 18.00 |
| `film_grain` | 11.62 |
| `sharpen` | 8.78 |
| `pixelate` | 3.28 |
| `mirror_flip` | 3.26 |
| `kaleidoscope` | 5.86 |

The reference `libx264` encode took 34.3 ms/frame on the same machine. The active
`flash` peak, full-frame overlay, and some effect combinations can therefore become the
bottleneck at 4K. Per-frame costs remained additive: `color_adjust` + `vignette` +
`film_grain` took 34.22 ms/frame, compared with 34.13 ms/frame for the sum of their
individual measurements.

An end-to-end one-second `run_to_file` cross-check, including decode and `libx264`
medium/CRF 23 encode, took 44.3 ms/frame with no operations and 324.8 ms/frame for that
three-effect plan. Its incremental cost was 1.03× the sum of the three individual plan
increments, so the scheduler did not materially compound framewise overhead. These
wall-clock results include content-dependent encoding work; grain makes frames harder
to compress, which is why its end-to-end cost is much larger than its isolated pixel
cost.

`FilmGrain` kept a 51.95 MiB padded noise pool at 4K. Its offset table for 60 frames was
960 bytes, and initialization peaked at 69.27 MiB of traced Python memory. These
measurements used macOS 14.0, Python 3.13.5, NumPy 2.4.6, and OpenCV 5.0.0.

# Verification records

These point-in-time measurements support release checks and implementation decisions.
They describe the tested environment and are not performance guarantees for other
hardware, inputs, or dependency versions.

For the interfaces covered by the AI checks, see [AI generation](ai/generation.md),
[AI understanding](ai/understanding.md), and [Dubbing](ai/dubbing.md). For the design
decision supported by the effects profile, see [The streaming
engine](../explanation/streaming-engine.md#why-pixel-effects-are-not-ffmpeg-filters).

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

Run the harness once to populate model caches, then again from a fresh process into a
new output directory. The reported time includes cached model loading, inference,
output writing, semantic validation, and weight cleanup.

```bash
uv sync --all-extras --group ai --frozen
HF_HOME=/workspace/.hf_home TOKENIZERS_PARALLELISM=false OLLAMA_HOST=127.0.0.1:11434 \
  uv run python scripts/verify_ai_models.py \
  --all \
  --video verification-input/cam1_1min.mp4 \
  --workdir verify-results/measured
```

### Dubbing synchronization threshold

Speech synthesis is nondeterministic. Four earlier A100 runs with the same one-minute
input established the failure limit.

| Run | Truncated segments | Mean speed factor | Worst truncation |
|---|---:|---:|---:|
| Baseline 1 | 9/17 | 1.072 | 2.460 s |
| Baseline 2 | 8/17 | 1.054 | 1.680 s |
| Baseline 3 | 10/17 | 1.085 | 2.000 s |
| Complete model run | 7/17 | 1.102 | 2.400 s |

The dub verification fails if the timing summary is missing or if one segment loses
more than 3.0 seconds during synchronization.

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

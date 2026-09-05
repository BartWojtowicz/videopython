# Verification records

These point-in-time measurements support release checks and implementation decisions.
They describe the tested environment and are not performance guarantees for other
hardware, inputs, or dependency versions.

For the interfaces covered by the AI checks, see [AI generation](ai/generation.md),
[AI understanding](ai/understanding.md), and [Dubbing](ai/dubbing.md). For the design
decision supported by the effects profile, see [The streaming
engine](../explanation/streaming-engine.md#why-pixel-effects-are-not-ffmpeg-filters).

## AI model verification

The real-model harness in `scripts/verify_ai_models.py` passed on 2026-09-05. It used
library commit `c65639080d7369322b4bd5383af0a8db35f7e689` and a representative
60.08-second Polish clip. The input was 1280×720 H.264 video with AAC audio. Its SHA-256
was `4a258bf9eb50a120485399a60768479bec8b72fae2e98de21361c751eff350f0`.

The environment used Python 3.11.16, an NVIDIA RTX PRO 6000 Blackwell Workstation
Edition with 97,887 MiB VRAM and compute capability 12.0, driver 595.71.05, PyTorch
2.13.0+cu130, Diffusers 0.39.0, Transformers 5.14.1, Safetensors 0.8.0, Ollama server
0.33.3, and Ollama Python client 0.6.2.

### Timing protocol and reproduction

The download caches were populated before the measured run. The harness then ran from a
fresh Python process and wrote to a new output directory. Ollama used
`OLLAMA_KEEP_ALIVE=0`, and `ollama stop` confirmed that the model was not loaded before
the run. Each check time includes cached model loading, inference, output writing,
semantic validation, and weight cleanup. It excludes dependency installation and model
downloads.

Start the Ollama server separately with
`OLLAMA_HOST=127.0.0.1:11434 OLLAMA_KEEP_ALIVE=0 ollama serve` before these commands.

Use the same input and run the first command once to populate all caches. Discard its
times. Stop Ollama, then run the same checks in a new process and record the second
report.

```bash
uv sync --all-extras --group ai --frozen
ollama pull qwen3.6:27b

HF_HOME=/workspace/.hf_home TOKENIZERS_PARALLELISM=false \
  uv run python scripts/verify_ai_models.py \
  --all \
  --video verification-input/cam1_1min.mp4 \
  --workdir verify-results/warm

OLLAMA_HOST=127.0.0.1:11434 ollama stop qwen3.6:27b

HF_HOME=/workspace/.hf_home TOKENIZERS_PARALLELISM=false \
  uv run python scripts/verify_ai_models.py \
  --all \
  --video verification-input/cam1_1min.mp4 \
  --workdir verify-results/measured
```

### Results and timings

| # | Check | Elapsed | Result | Semantic evidence |
|---|---|---:|---|---|
| 1 | `env` — Environment and package versions | 1.042 s | **PASS** | PyTorch 2.13.0, RTX PRO 6000 Blackwell |
| 2 | `imports` — Public AI entrypoints resolve | 0.179 s | **PASS** | 38/38 entrypoints |
| 3 | `ollama` — Schema-constrained output | 4.111 s | **PASS** | `qwen3.6:27b` returned a valid Spanish translation |
| 4 | `t2i` — Prompt conditions image output | 84.273 s | **PASS** | Same-seed cross-prompt correlation 0.017 |
| 5 | `t2v` — Video renders and moves | 524.665 s | **PASS** | 49 frames, 71.1% of pixels moved |
| 6 | `i2v` — Input conditions frame 0 | 245.381 s | **PASS** | Input correlation 0.993, 88.7% of pixels moved |
| 7 | `tts` — Speech is intelligible | 15.176 s | **PASS** | Round-trip word overlap 82%; cloned sample also written |
| 8 | `separation` — Voice reaches vocals stem | 6.481 s | **PASS** | Vocals correlation 0.989 with voice, -0.225 with music |
| 9 | `music` — Output is audible and conditioned | 5.318 s | **PASS** | Peak 0.260, cross-prompt envelope correlation 0.074 |
| 10 | `detect` — Known objects are detected | 0.897 s | **PASS** | Five objects: cat, remote, sofa |
| 11 | `dub` — Full Polish-to-Spanish dub | 40.493 s | **PASS** | 17/17 translated; cloned voice; worst truncation 1.440 seconds |

The complete run took 928.017 seconds. The default dub uses voice cloning without
speaker diarization. It groups segments under one `speaker_0` clone.

The optional diarization path was timed separately so that it does not replace the
default baseline. Run it once to populate the pyannote cache, unload Ollama, then repeat
it from a fresh process:

```bash
HF_HOME=/workspace/.hf_home TOKENIZERS_PARALLELISM=false \
  uv run python scripts/verify_ai_models.py \
  --only dub --enable-diarization \
  --video verification-input/cam1_1min.mp4 \
  --workdir verify-results/diarized-warm

OLLAMA_HOST=127.0.0.1:11434 ollama stop qwen3.6:27b

HF_HOME=/workspace/.hf_home TOKENIZERS_PARALLELISM=false \
  uv run python scripts/verify_ai_models.py \
  --only dub --enable-diarization \
  --video verification-input/cam1_1min.mp4 \
  --workdir verify-results/diarized-measured
```

The measured diarized dub passed in 37.017 seconds. Pyannote found two speakers, and
voice cloning produced one sample for each speaker. All three speaker-aligned segments
were translated and audible, with no truncation. The segment count differs from the
default dub because diarization changes transcription segmentation, so the two dub
times are separate baselines.

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

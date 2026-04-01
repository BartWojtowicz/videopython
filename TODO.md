# Streaming Video Processing Pipeline - Design

## Problem

A 5-minute 1080p 30fps video with simple effects OOM-kills a 16GB T4 instance.
Root cause: the entire video is decoded into a single numpy array (~56GB for 5min 1080p),
and every effect creates full copies of that array via `np.r_` concatenation.

Peak memory for 5min + 3 effects: ~170GB. Even an A100 (80GB) fails.

## Baseline Benchmark (cam1_1min.mp4)

Measured on cam1_1min.mp4 (1280x720, 25fps, 60s, 1498 frames).
Plan: color_adjust (grayscale) + volume_adjust + fade in/out.
Script: `benchmark_baseline.py`.

| Metric | Value |
|--------|-------|
| Frame data in memory | 3.86 GB (1498 frames x 1280x720x3) |
| Peak RSS | 9,187 MB (~2.4x frame data) |
| Parse + validate | 0.07s |
| Run (load + effects) | 78.28s |
| Save | 11.61s |
| Total | 89.96s |
| Output file | 14.0 MB |

The 2.4x multiplier confirms intermediate copies from effects. For the 5-min
1080p case (56GB frames), this would extrapolate to ~134GB peak RSS.

## Current Architecture

```
Video.from_path()  -->  np.ndarray (ALL frames)  -->  Effect.apply() x N  -->  Video.save()
                        ^^^^^^^^^^^^^^^^^^^^^^^^
                        9000 frames x 1920x1080x3 = 56GB
```

Each `Effect.apply()`:
1. Slices frames into 3 parts (pre, target, post)
2. Calls `_apply()` on target -- many effects create another copy internally
3. `np.r_` concatenates all 3 parts back into a new array

`VideoEdit.run()`:
1. For each segment: `Video.from_path()` (full decode) -> transforms -> effects
2. Concatenate segments with `Video.__add__()` (another `np.r_` copy)
3. Apply post-transforms and post-effects (more copies)

## Design Goals

1. Process videos of arbitrary length with bounded memory (tens of MB, not GB)
2. Preserve the existing public API -- `VideoEdit.from_dict(plan).run().save()` still works
3. Automatic: streaming kicks in when possible, eager fallback when not
4. General: works for all combinations of operations, not just the reported case
5. No performance regression on short videos

## Operation Streamability Analysis

The design must handle every operation in the system. Each needs classification.

### Effects (frame count and shape invariant)

| Effect | Per-frame independent? | Streamable? | Notes |
|--------|----------------------|-------------|-------|
| ColorGrading | Yes | Yes | Already has `_grade_frame()` |
| Blur | Yes (per-frame) | Yes | Already has `_blur_frame()`, mode affects sigma per frame but index-computable |
| Vignette | Yes | Yes | Mask depends only on resolution, precompute once |
| Fade | Yes | Yes | Alpha depends on frame index, precompute array |
| FullImageOverlay | Yes | Yes | Overlay image + fade precomputable, blend per frame |
| TextOverlay | Yes | Yes | Render overlay once, blend per frame |
| Zoom | Yes | Yes | Crop region per frame is a function of index |
| KenBurns | Yes | Yes | Region interpolation per frame, precompute keyframes |
| VolumeAdjust | Audio-only | Yes (no-op for frames) | Already handled by AudioEffect base |

All current effects are per-frame independent. None require cross-frame state
(no temporal filtering, no optical flow, no multi-frame accumulation).

### Transforms (may change frame count, dimensions, or ordering)

| Transform | Streamable via ffmpeg? | Strategy | Fallback needed? |
|-----------|----------------------|----------|-----------------|
| CutFrames / CutSeconds | Yes | `-ss` / `-t` on decode (already implemented) | No |
| Resize | Yes | `-vf scale=W:H` on decode (already implemented) | No |
| ResampleFPS | Yes | `-vf fps=N` on decode (already implemented) | No |
| Crop | Yes | `-vf crop=W:H:X:Y` on decode | No |
| SpeedChange (constant) | Yes | `-vf setpts=PTS*{1/speed}` + audio time_stretch | No |
| SpeedChange (ramp) | No | Non-uniform time mapping, needs full frame access | Yes |
| Reverse | No | Must read all frames to reverse order | Yes |
| FreezeFrame | Partially | Insert/replace at known timestamp, but requires frame duplication in stream | Yes (complex) |
| PictureInPicture | No | Needs overlay video frames aligned with main video | Yes |
| SilenceRemoval | No | Requires transcript + non-contiguous cuts | Yes |
| FaceTrackingCrop (AI) | No | Needs face detection across all frames first | Yes |
| SplitScreenComposite (AI) | No | Needs multiple video streams + face tracking | Yes |

### Key Observation

The common production pipeline (cut + resize + color + audio + fade) is fully
streamable. The non-streamable transforms are either rare in automated pipelines
(Reverse, FreezeFrame) or inherently require full materialization (PictureInPicture,
AI transforms).

## Proposed Architecture

### Core Idea: Dual-Mode Execution

`VideoEdit` checks each segment's operation chain. If every operation in the chain
is streamable, it runs as a streaming ffmpeg-to-ffmpeg pipeline. Otherwise, it
falls back to the current eager path. This decision is per-segment, not per-pipeline,
so a multi-segment plan can stream some segments and eager-load others.

```
                        +-- streamable? --> StreamingPipeline (ffmpeg pipe)
segment.process() --+
                        +-- not streamable? --> eager path (current code)
```

### Phase 1: Eliminate Unnecessary Copies (no API changes)

Safe, isolated fixes that reduce memory for the eager path. Worth doing first
because: (a) they help even without streaming, (b) they make the eager fallback
less painful, (c) they're low risk.

#### 1a. In-place frame mutation in effects

Effects that process frames independently should mutate `video.frames` in-place.

**ColorGrading._apply()** currently:
```python
new_frames = [self._grade_frame(f) for f in video.frames]  # Python list of copies
video.frames = np.array(new_frames)                         # another full copy
```

Should become:
```python
for i in range(len(video.frames)):
    video.frames[i] = self._grade_frame(video.frames[i])
```

Affected effects: `ColorGrading`, `Blur`, `FullImageOverlay`, `KenBurns`, `TextOverlay`.

**Multiprocessing concern**: current code uses `multiprocessing.Pool.map()` for
100+ frames, which inherently copies frames across process boundaries. Options:
- Use `multiprocessing.shared_memory` with index-based writes (complex but zero-copy)
- Switch to `concurrent.futures.ThreadPoolExecutor` (GIL released during numpy/cv2 C
  extensions, so threading gives real parallelism for these workloads)
- Drop multiprocessing for the in-place path (serial is still faster than OOM)

Recommendation: switch to threading. The heavy ops (cv2.GaussianBlur, np arithmetic,
cv2.cvtColor) all release the GIL. Avoids the IPC serialization overhead too.

#### 1b. Skip np.r_ rebuild for full-range effects

`Effect.apply()` always slices + concatenates, even when the effect covers the
entire video (start=None, stop=None). When full-range, skip directly to `_apply()`:

```python
def apply(self, video, start=None, stop=None):
    if start is None and stop is None:
        original_shape = video.video_shape
        video = self._apply(video)
        if video.video_shape != original_shape:
            raise RuntimeError(...)
        return video
    # ... existing slice + np.r_ path for partial ranges
```

This is the common case -- effects in VideoEdit plans rarely use start/stop.
For partial-range effects, the np.r_ path is correct and necessary.

Note: `Fade.apply()` and `AudioEffect.apply()` override `apply()` directly, so
this change only affects the `Effect` base class. Fade already modifies frames
in-place (good). AudioEffect already skips frames (good).

#### 1c. Effect fusion

When consecutive effects in a segment's pipeline are all per-frame and cover
the same range (full video), fuse them into a single pass:

```python
# In SegmentConfig.apply_operations or VideoEdit execution:
# Group consecutive full-range, per-frame effects
# Apply them as: for each frame, run all effects on that frame

for i in range(len(video.frames)):
    frame = video.frames[i]
    for effect in fused_group:
        frame = effect.process_frame(frame, i, total_frames)
    video.frames[i] = frame
```

This requires the `process_frame` interface from Phase 2b, so 1c depends on 2b.
Alternatively, implement fusion as a simpler optimization: if all effects in a
group have in-place `_apply`, just call them sequentially (no copies anyway).

**Memory impact of Phase 1 on 5-min example:**
- Before: ~170 GB peak
- After: ~56 GB (initial load only, effects are in-place)
- Still exceeds 16GB T4, but now bounded by video size, not video x effects

### Phase 2: Streaming Execution

The core architectural change. Frames flow through the pipeline without full
materialization.

#### 2a. Per-frame effect interface

Extend `Effect` with an optional streaming protocol:

```python
class Effect(ABC):
    # Existing batch interface (unchanged)
    @abstractmethod
    def _apply(self, video: Video) -> Video: ...

    # New streaming interface
    def supports_streaming(self) -> bool:
        """Override to return True when process_frame is implemented."""
        return False

    def init_streaming(self, metadata: VideoMetadata, start_frame: int, end_frame: int) -> None:
        """Precompute per-frame parameters before streaming begins.

        Called once with the video metadata and the frame range this effect
        applies to. Use this to build lookup tables, render overlays, etc.

        Args:
            metadata: Resolution, fps, etc of the frames that will arrive.
            start_frame: First frame index this effect applies to.
            end_frame: Last frame index (exclusive).
        """
        pass

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Process a single frame during streaming.

        Args:
            frame: RGB uint8 array of shape (H, W, 3). May be modified in place.
            frame_index: Absolute index within the effect's range (0-based from
                start_frame passed to init_streaming).

        Returns:
            Processed frame, same shape and dtype.
        """
        raise NotImplementedError
```

Implementation per effect:

- **ColorGrading**: `process_frame` = existing `_grade_frame`. No init needed.
- **Blur**: `init_streaming` precomputes sigma per frame (for ascending/descending
  modes). `process_frame` = `_blur_frame(frame, sigma[index])`.
- **Vignette**: `init_streaming` creates the mask array (depends on resolution).
  `process_frame` = `(frame * mask).astype(uint8)`.
- **Fade**: `init_streaming` precomputes alpha array from curve/mode/duration.
  `process_frame` = `(frame * alpha[index]).astype(uint8)`.
- **FullImageOverlay**: `init_streaming` loads + resizes overlay, precomputes
  per-frame opacity for fade_time. `process_frame` = alpha composite.
- **TextOverlay**: `init_streaming` renders text to RGBA overlay once.
  `process_frame` = alpha composite (same every frame).
- **Zoom**: `init_streaming` precomputes crop region per frame.
  `process_frame` = crop + resize.
- **KenBurns**: `init_streaming` precomputes interpolated region per frame.
  `process_frame` = crop + resize.
- **VolumeAdjust**: `supports_streaming()` returns True, `process_frame` is
  identity (audio handled separately). Could skip calling it entirely.

#### 2b. StreamingPipeline

The executor that connects ffmpeg decode -> effects -> ffmpeg encode:

```python
class StreamingPipeline:
    """Streams frames from ffmpeg decode through effects to ffmpeg encode.

    Memory usage is O(1) with respect to video length -- only one frame
    (or a small batch) is in memory at any time.
    """

    def __init__(
        self,
        source: Path,
        start_second: float,
        end_second: float,
        ffmpeg_filters: list[str],     # spatial/temporal transforms as ffmpeg -vf
        frame_effects: list[Effect],    # per-frame effects (supports_streaming=True)
        audio_effects: list[AudioEffect],
        output_settings: OutputSettings, # crf, preset, format
    ): ...

    def run(self, output_path: Path) -> Path:
        metadata = VideoMetadata.from_path(self.source)

        # 1. Audio pipeline (separate, small memory footprint)
        #    Load audio for [start, end] range (~50MB for 5min stereo)
        #    Apply audio effects sequentially
        #    Save to temp WAV
        audio = Audio.from_path(self.source, start=self.start_second, end=self.end_second)
        for effect in self.audio_effects:
            audio = effect._apply_audio(audio, ...)
        temp_audio = audio.save(temp_path, format="wav")

        # 2. Initialize effects for streaming
        for effect in self.frame_effects:
            effect.init_streaming(metadata, start_frame=0, end_frame=total_frames)

        # 3. Build ffmpeg decode command
        #    Includes: -ss, -t, -vf (scale, crop, fps, setpts for speed)
        decode_cmd = self._build_decode_cmd()

        # 4. Build ffmpeg encode command
        #    Includes: -i pipe:0 (video), -i temp.wav (audio), encoding params
        encode_cmd = self._build_encode_cmd(temp_audio, output_path)

        # 5. Stream frames
        decoder = subprocess.Popen(decode_cmd, stdout=PIPE)
        encoder = subprocess.Popen(encode_cmd, stdin=PIPE)

        frame_size = metadata.width * metadata.height * 3
        for frame_idx in range(total_frames):
            raw = decoder.stdout.read(frame_size)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)

            # Apply all effects to this single frame
            for effect in self.frame_effects:
                frame = effect.process_frame(frame, frame_idx)

            encoder.stdin.write(frame.tobytes())

        # 6. Cleanup
        encoder.stdin.close()
        decoder.stdout.close()
        ...

        return output_path
```

**Frame batching optimization**: Instead of one frame at a time, read/write in
small batches (e.g., 8-16 frames). This reduces syscall overhead and allows
vectorized numpy operations (e.g., Vignette mask broadcast over a batch).
Memory cost of 16 frames at 1080p = ~95MB. Still trivial.

#### 2c. Transform-to-ffmpeg-filter compilation

Streamable transforms compile to ffmpeg `-vf` filter expressions rather than
operating on loaded frames:

```python
def compile_to_ffmpeg_filter(transform, segment_config) -> str | None:
    """Return an ffmpeg filter string, or None if not streamable."""

    if isinstance(transform, Resize):
        w, h = transform.target_width, transform.target_height
        return f"scale={w}:{h}"

    if isinstance(transform, Crop):
        return f"crop={w}:{h}:{x}:{y}"

    if isinstance(transform, ResampleFPS):
        return f"fps={transform.target_fps}"

    if isinstance(transform, SpeedChange) and transform.mode == "constant":
        return f"setpts=PTS*{1/transform.speed}"

    if isinstance(transform, (CutFrames, CutSeconds)):
        return None  # handled by -ss/-t, not -vf

    return None  # not streamable
```

Multiple filters chain: `-vf "scale=1280:720,fps=30,setpts=PTS*0.5"`.

**Cut transforms** are special: `CutSeconds` maps directly to ffmpeg's `-ss` and `-t`
parameters on the decode command. `CutFrames` converts to seconds via fps.
These are already handled by `SegmentConfig.load_segment()` which passes
`start_second`/`end_second` to `Video.from_path()`.

**SpeedChange with ramp**: Not expressible as a single ffmpeg filter. Falls back
to eager mode. This is fine -- speed ramps are rare and inherently complex.

#### 2d. VideoEdit streaming integration

```python
class VideoEdit:
    def run(self, context=None):
        video = self._assemble_segments(context)
        # post-transforms and post-effects unchanged
        ...
        return video

    def _assemble_segments(self, context=None):
        segment_results = []
        for segment in self.segments:
            if self._segment_can_stream(segment, context):
                # Stream to temp file, then load only if needed for post-ops
                path = self._stream_segment(segment, context)
                segment_results.append(path)
            else:
                # Eager: current behavior
                video = segment.process_segment(context)
                segment_results.append(video)
        return self._combine_results(segment_results)
```

**Per-segment streaming** is the right granularity because:
- Segment 1 might be a simple cut+color (streamable)
- Segment 2 might use PictureInPicture (not streamable)
- No reason to penalize segment 1 because of segment 2

**Combining streamed segments**: If all segments streamed to temp files and
there are no post-transforms/post-effects, use ffmpeg concat demuxer:
```
ffmpeg -f concat -safe 0 -i list.txt -c copy output.mp4
```
This is zero-copy -- no re-encoding, no frame loading.

If there ARE post-effects, we need to either:
- Stream the concatenated output through another pipeline (chain two pipelines)
- Load the concatenated result for post-processing (acceptable if result is short)

The chained approach is better for long videos. Concat demuxer output can be
piped directly into a second StreamingPipeline for post-effects.

**Post-transforms**: These are rare but tricky. A post-transform like Resize
after assembly could be an ffmpeg filter on the concat output. But a post-transform
like Reverse would require full materialization. Same streamability classification
applies -- check and fall back as needed.

**Return type**: `VideoEdit.run()` currently returns a `Video` (frames in memory).
For streaming mode, options:
1. Return a `Video` with frames loaded from the temp output file (defeats the
   purpose for very long videos, but preserves API compatibility)
2. Return a `Video` with a path reference and lazy loading
3. Add `VideoEdit.run_to_file(output_path)` that streams directly

Option 1 for backward compatibility, option 3 as the recommended path for large
videos. Document that `run()` is for interactive use and `run_to_file()` is for
production pipelines.

#### 2e. Handling effects with start/stop in streaming mode

Effects can have time ranges (start/stop). In streaming mode, the pipeline
needs to know which effects apply to each frame:

```python
# During streaming:
for frame_idx in range(total_frames):
    frame = read_frame()
    for effect, start_frame, end_frame in effect_schedule:
        if start_frame <= frame_idx < end_frame:
            frame = effect.process_frame(frame, frame_idx - start_frame)
    write_frame(frame)
```

Each effect's `init_streaming` receives its own (start_frame, end_frame) so it
can precompute parameters for exactly that range.

#### 2f. Handling multi-segment audio

Audio across segments needs careful handling:

- Each segment's audio is loaded independently (bounded memory)
- Per-segment audio effects applied
- Segment audio concatenated (with optional crossfade for transitions)
- Post audio effects applied to the concatenated result
- Final audio saved to temp WAV for muxing with video

Audio is always small relative to video frames, so eager loading is fine.
A 30-minute video at 44.1kHz stereo float32 = ~300MB.

### Phase 3: Lazy Video (future)

For programmatic use outside VideoEdit. Deferred -- Phase 2 covers the
production use case. Described briefly for completeness.

A `LazyVideo` wraps a file path + a chain of deferred operations. Calling
`.save()` triggers streaming execution. Calling `.frames` triggers eager
loading. This gives users the best of both worlds.

## Implementation Order

```
Phase 1a: In-place frame mutation in effects     [low risk, high impact]
Phase 1b: Skip np.r_ for full-range effects      [trivial change]
Phase 1c: Threading instead of multiprocessing    [medium, helps memory + speed]
Phase 2a: Per-frame streaming interface on Effect [ABC extension]
Phase 2b: StreamingPipeline class                 [core new code]
Phase 2c: Transform-to-ffmpeg-filter compiler     [ffmpeg integration]
Phase 2d: VideoEdit streaming integration         [orchestration]
Phase 2e: run_to_file() for direct streaming      [API addition]
Phase 2f: Multi-segment concat via ffmpeg         [ffmpeg concat demuxer]
Phase 3:  Lazy Video                              [future]
```

Phase 1 ships independently. Phase 2a-2b can be developed in isolation (unit
testable without VideoEdit). Phase 2c-2f wires everything together.

## Memory Budget Comparison

5-minute 1080p 30fps video, 3 effects (color_adjust + volume_adjust + fade):

| Stage | Current | Phase 1 | Phase 2 |
|-------|---------|---------|---------|
| Frame load | 56 GB | 56 GB | ~6 MB (1 frame) |
| Effect 1 (color) | +56 GB copy | 0 (in-place) | 0 (per-frame) |
| Effect 2 (volume) | 0 (audio-only) | 0 | 0 |
| Effect 3 (fade) | +56 GB copy | 0 (in-place) | 0 (per-frame) |
| Audio | ~50 MB | ~50 MB | ~50 MB |
| **Peak** | **~170 GB** | **~56 GB** | **~56 MB** |

Phase 2 makes memory independent of video length.

## Edge Cases and Fallback Strategy

### When streaming is not possible

The pipeline falls back to eager mode (current behavior) when any operation
in the chain requires full materialization. Fallback triggers:

| Operation | Why it needs eager mode |
|-----------|----------------------|
| Reverse | Must read all frames before outputting first |
| SpeedChange (ramp mode) | Non-uniform time mapping across frames |
| FreezeFrame | Inserts/duplicates frames at arbitrary points |
| PictureInPicture | Requires overlay video synchronized frame-by-frame |
| SilenceRemoval | Non-contiguous cuts based on transcript timing |
| FaceTrackingCrop | Needs face detection pass across all frames first |
| SplitScreenComposite | Multiple synchronized video streams + face tracking |

**Fallback is per-segment**, not per-pipeline. A 10-segment plan where segment 3
uses Reverse still streams the other 9 segments.

**Phase 1 optimizations still apply to fallback segments.** Even when a segment
must eager-load, in-place mutation and skipped np.r_ copies reduce its memory
from N * frames to 1 * frames.

### Effect start/stop with streaming

When an effect has a partial time range (e.g., blur from 10s to 20s), the
streaming pipeline applies it conditionally per frame. The `init_streaming`
call tells the effect its range, and the pipeline tracks frame indices.

### Post-effects spanning the full assembled video

Post-effects apply after segment concatenation. If all post-effects are
streamable, they run as a second pipeline stage on the concatenated output.
If any post-effect is not streamable, the concatenated video is loaded for
eager processing. In practice, post-effects are usually simple (fade, color)
and streamable.

### Post-transforms

Post-transforms (applied after assembly) follow the same logic:
- Streamable post-transforms (Resize, Crop, ResampleFPS) compile to ffmpeg
  filters on the concat output.
- Non-streamable post-transforms trigger eager fallback for the post stage.

### Single segment with no effects

Common case: just a cut. Streaming pipeline degenerates to:
`ffmpeg -ss X -t Y -i input.mp4 -c:v libx264 ... output.mp4`
This is effectively a transcode with seek, which ffmpeg handles natively.
Even simpler: if no re-encoding is needed (same codec, no effects), use
`-c copy` for zero-processing extraction.

### Error handling in streaming mode

If an effect's `process_frame()` raises, the streaming pipeline must clean up
both ffmpeg subprocesses. The pipeline should catch exceptions, terminate both
processes, clean up temp files, and re-raise with context about which frame
and effect failed.

## Open Questions

1. **Batch size for frame streaming**: 1 frame is simplest but high syscall
   overhead. 8-16 frames amortizes I/O while keeping memory trivial (~100MB).
   Need to benchmark.

2. **Threading for process_frame**: Within a batch, could we apply
   process_frame to multiple frames in parallel using threads? The GIL-free
   numpy/cv2 ops make this viable. Worth benchmarking.

3. **Progress reporting**: Streaming needs frame-count-based progress (known
   from metadata). Replace per-effect tqdm with pipeline-level progress.

4. **run() vs run_to_file() API**: Should `run()` automatically detect large
   videos and stream to a temp file? Or should users explicitly call
   `run_to_file()`? Explicit is probably better -- less magical behavior.

5. **Transition support in VideoEdit plans**: Currently VideoEdit doesn't
   support transitions in the plan dict format. If added later, streaming
   would need to buffer overlap frames between segments (bounded by
   transition duration, typically < 2s = ~60 frames at 30fps). The design
   accommodates this without architectural changes.

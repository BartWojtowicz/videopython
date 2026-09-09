# The dubbing pipeline

Dubbing must preserve meaning, speaker identity, and timing across models that can
fail independently. The pipeline keeps source turns for voice references and
uses smaller phrases to place translated speech. For executable steps, see
[Dub a video](../how-to/dubbing.md); fields and limits are in the
[dubbing reference](../reference/ai/dubbing.md).

## Turns and phrases serve different purposes

Diarization groups words by speaker. Long speaker turns can contain several
sentences and pauses, so they are often poor units for speech synthesis.
The pipeline splits turns at sentence ends, clauses, or pauses using validated
word timestamps. It prefers phrases of roughly eight seconds and avoids tiny tails.
Text comes from slices of the source segment, which preserve internal spacing.
Missing or inconsistent word alignment leaves the segment intact.

`source_transcription` keeps the original turns. Translated phrases carry a
`source_segment_index`, and failure lists refer to original source indices.
Voice reference extraction uses the full original turns so that splitting a turn
does not select a new voice reference for every sentence.

## Translation and synthesis use bounded inputs

Translation sends one source part per request with nearby text marked as context.
It asks for concise speech without dropping meaning. Sequential requests isolate
segment identities but add request overhead. Invalid identities, missing output,
or exhausted output budgets trigger retries. A failed part makes its parent a
translation failure.

Local speech synthesis splits long text into bounded calls and joins their audio
before timing the parent phrase. Calls near the speech-token limit retry with
smaller text units. Invalid vocoder tokens are rejected. These checks can detect
incomplete generation, but cannot prove that every requested word was spoken.

Tiny adjacent fragments can join within one speaker. Isolated fragments below the
minimum duration are reported as synthesis failures. The reference documents these
limits and how failures map to source entries.

## Timing preserves source anchors

Each phrase starts at its source-word timestamp. Shorter speech keeps its natural
speed and leaves silence before the next phrase. Longer speech uses available gaps
before acceleration. The preferred speed limit is not a hard ceiling: the pipeline
can exceed it to keep complete generated speech, and reports that in the timing
summary. This is phrase alignment, not phoneme-level lip synchronization.

Dubbing prefers FFmpeg's Rubber Band filter when available and uses `atempo` with a
warning otherwise. Small residual duration differences are corrected by resampling
all generated samples, which can slightly change pitch. Assembly applies 5 ms fades
at phrase edges without changing their sample counts or start timestamps.

## Runtime improvements do not establish speech quality

On compatible CUDA inputs, local speech decoder operations use graph replay to
reduce launch overhead. Model weights, precision, and generation settings remain
the same. Other layouts and CPU synthesis use the original execution path. Graph
buffers are released with the model.

Translation success counts establish response availability. Synthesis success counts
establish audio availability. Neither proves meaning, pronunciation, voice consistency,
or intelligibility after acceleration. Listen to the final mix and inspect excessive
speeds separately. The [verification record](../reference/verification.md#final-dubbing-review-0612)
keeps the measured quality limitations and performance comparisons.

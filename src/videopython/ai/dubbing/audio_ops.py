"""Audio DSP helpers for the dubbing pipeline.

Merges the former ``expressiveness`` and ``loudness`` leaf modules: RMS,
source-prosody -> Chatterbox expressiveness mapping, and LUFS/peak loudness
matching. These are the small numpy/DSP helpers the pipeline composes; keeping
them in one file (rather than two ~50-line files) makes the audio-side toolkit
legible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from videopython.ai.dubbing.models import Expressiveness

if TYPE_CHECKING:
    from videopython.audio import Audio


def rms(data: np.ndarray) -> float:
    """RMS over samples; ``0.0`` for empty input. float64 reduction so a
    long slice can't overflow the squared accumulator."""
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))


# --- Source-prosody-driven expressiveness (Chatterbox TTS knobs) -----------
# Source-segment RMS / whole-vocals RMS below CALM lands in the calm bucket;
# above DRAMATIC in the dramatic bucket; in between gets Chatterbox's defaults.
# Knob values picked by-ear on cam1_1min.mp4 -- see RELEASE_NOTES 0.29.0.
CALM_RATIO_THRESHOLD = 0.7
DRAMATIC_RATIO_THRESHOLD = 1.3
_CALM = Expressiveness(exaggeration=0.3, cfg_weight=0.7)
_DRAMATIC = Expressiveness(exaggeration=0.85, cfg_weight=0.35)


def expressiveness_for(source_slice: Audio, baseline_rms: float) -> Expressiveness:
    """Map a source vocals slice to a Chatterbox expressiveness profile
    by RMS ratio. Falls back to the no-knobs default for empty or silent
    inputs."""
    if baseline_rms <= 0.0:
        return Expressiveness()
    segment_rms = rms(source_slice.data)
    if segment_rms <= 0.0:
        return Expressiveness()
    ratio = segment_rms / baseline_rms
    if ratio < CALM_RATIO_THRESHOLD:
        return _CALM
    if ratio > DRAMATIC_RATIO_THRESHOLD:
        return _DRAMATIC
    return Expressiveness()


# --- LUFS / peak loudness matching -----------------------------------------
# BS.1770 integrated-loudness measurement requires at least 400 ms of audio
# (one gating block). Below this, fall back to peak match -- pyloudnorm
# returns -inf or warns, neither of which gives a usable gain.
_LUFS_MIN_DURATION_SECONDS = 0.4


def peak_match(target: Audio, reference: Audio) -> Audio:
    """Scale ``target`` so its peak amplitude matches ``reference``.

    Used as the fallback when LUFS measurement isn't viable (clip < 0.4s
    or silent input). The new ``Audio`` shares no buffer with ``target``.
    """
    from videopython.audio import Audio as _Audio

    target_peak = float(np.max(np.abs(target.data))) if target.data.size else 0.0
    reference_peak = float(np.max(np.abs(reference.data))) if reference.data.size else 0.0

    if target_peak <= 0.0 or reference_peak <= 0.0:
        return target

    scale = reference_peak / target_peak
    if abs(scale - 1.0) < 1e-3:
        return target

    return _Audio(target.data * scale, target.metadata)


def loudness_match(target: Audio, reference: Audio) -> Audio:
    """Scale ``target`` so its integrated loudness (BS.1770 / LUFS) matches ``reference``.

    Demucs background normalization and the timing-assembler peak guard
    each clamp at 1.0 instead of restoring perceived loudness, so a
    dubbed mix lands perceptually "thinner" than the source even after
    peak match. LUFS captures the ear-weighted envelope that peak ratio
    misses on dialogue-heavy material.

    Falls back to :func:`peak_match` when either clip is shorter than
    the BS.1770 gating block (400 ms) or when measurement returns -inf
    (silent or near-silent gated content). After gain is applied, peaks
    are clamped to 0.99 -- BS.1770 has no peak ceiling and a sufficiently
    quiet source can demand gain that would otherwise clip.
    """
    from videopython.audio import Audio as _Audio

    target_dur = target.metadata.duration_seconds
    ref_dur = reference.metadata.duration_seconds
    if target_dur < _LUFS_MIN_DURATION_SECONDS or ref_dur < _LUFS_MIN_DURATION_SECONDS:
        return peak_match(target, reference)

    if not target.data.size or not reference.data.size:
        return target

    from videopython.ai._optional import require

    pyloudnorm = require("pyloudnorm", feature="loudness matching")

    target_lufs = pyloudnorm.Meter(target.metadata.sample_rate).integrated_loudness(target.data)
    reference_lufs = pyloudnorm.Meter(reference.metadata.sample_rate).integrated_loudness(reference.data)

    # Either clip's gated content was below -70 LUFS (effectively silent
    # under BS.1770). Gain would be undefined -- fall back to peak match,
    # which has its own silent-input no-op.
    if not np.isfinite(target_lufs) or not np.isfinite(reference_lufs):
        return peak_match(target, reference)

    gain_db = reference_lufs - target_lufs
    if abs(gain_db) < 0.1:
        return target
    scale = float(10 ** (gain_db / 20.0))

    scaled = target.data * scale
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.99:
        scaled = scaled * (0.99 / peak)

    return _Audio(scaled, target.metadata)

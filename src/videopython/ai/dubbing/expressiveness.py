"""Source-prosody-driven expressiveness knobs for Chatterbox TTS."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from videopython.ai.dubbing.models import Expressiveness

if TYPE_CHECKING:
    from videopython.audio import Audio


# Prosody-conditioning thresholds. Source-segment RMS / whole-vocals RMS
# below CALM lands in the calm bucket; above DRAMATIC in the dramatic
# bucket; in between gets Chatterbox's defaults. Knob values picked
# by-ear on cam1_1min.mp4 -- see RELEASE_NOTES 0.29.0.
CALM_RATIO_THRESHOLD = 0.7
DRAMATIC_RATIO_THRESHOLD = 1.3
_CALM = Expressiveness(exaggeration=0.3, cfg_weight=0.7)
_DRAMATIC = Expressiveness(exaggeration=0.85, cfg_weight=0.35)


def rms(data: np.ndarray) -> float:
    """RMS over samples; ``0.0`` for empty input. float64 reduction so a
    long slice can't overflow the squared accumulator."""
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))


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

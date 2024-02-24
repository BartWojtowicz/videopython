from typing import Callable, Literal, TypeVar

from .audio_segment import AudioSegment

_AudioSegmentT = TypeVar("_AudioSegmentT", bound=AudioSegment)

def apply_mono_filter_to_each_channel(
    seg: _AudioSegmentT, filter_fn: Callable[[_AudioSegmentT], _AudioSegmentT]
) -> _AudioSegmentT: ...
def normalize(seg: _AudioSegmentT, headroom: float = ...) -> _AudioSegmentT: ...
def speedup(
    seg: _AudioSegmentT,
    playback_speed: float = ...,
    chunk_size: int = ...,
    crossfade: int = ...,
) -> _AudioSegmentT: ...
def strip_silence(
    seg: _AudioSegmentT,
    silence_len: int = ...,
    silence_thresh: int = ...,
    padding: int = ...,
) -> _AudioSegmentT: ...
def compress_dynamic_range(
    seg: _AudioSegmentT,
    threshold: float = ...,
    ratio: float = ...,
    attack: float = ...,
    release: float = ...,
) -> _AudioSegmentT: ...
def invert_phase(
    seg: _AudioSegmentT,
    channels: tuple[Literal[1], Literal[1]] | tuple[Literal[1], Literal[0]] | tuple[Literal[0], Literal[1]] = ...,
) -> _AudioSegmentT: ...
def low_pass_filter(seg: _AudioSegmentT, cutoff: float) -> _AudioSegmentT: ...
def high_pass_filter(seg: _AudioSegmentT, cutoff: float) -> _AudioSegmentT: ...
def pan(seg: _AudioSegmentT, pan_amount: float) -> _AudioSegmentT: ...
def apply_gain_stereo(seg: _AudioSegmentT, left_gain: float = ..., right_gain: float = ...) -> _AudioSegmentT: ...

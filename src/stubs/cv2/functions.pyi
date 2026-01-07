from typing import TypeVar, overload

import numpy as np
import numpy.typing as npt

from .constants import INTER_LINEAR

_TImg = TypeVar("_TImg", np.uint8, np.float64)
_TSize = tuple[int, int]

def resize(
    src: npt.NDArray[_TImg],
    dsize: _TSize,
    dst: npt.NDArray[_TImg] = ...,
    fx: float = 0,
    fy: float = 0,
    interpolation: int = INTER_LINEAR,
) -> npt.NDArray[_TImg]: ...
def GaussianBlur(
    src: npt.NDArray[_TImg],
    ksize: _TSize,
    sigmaX: float,
    dst: npt.NDArray[_TImg] = ...,
    sigmaY: float = ...,
    borderType: int = ...,
) -> npt.NDArray[_TImg]: ...
def imwrite(filename: str, img: npt.NDArray[_TImg]) -> bool: ...
@overload
def cvtColor(src: npt.NDArray[np.uint8], code: int) -> npt.NDArray[np.uint8]: ...
@overload
def cvtColor(src: npt.NDArray[np.float64], code: int) -> npt.NDArray[np.float64]: ...
def calcHist(
    images: list[npt.NDArray[np.uint8]],
    channels: list[int],
    mask: npt.NDArray[np.uint8] | None,
    histSize: list[int],
    ranges: list[int] | list[float],
) -> npt.NDArray[np.float32]: ...
def normalize(
    src: npt.NDArray[np.float32],
    dst: npt.NDArray[np.float32],
    alpha: float = ...,
    beta: float = ...,
    norm_type: int = ...,
    dtype: int = ...,
) -> npt.NDArray[np.float32]: ...
def compareHist(
    H1: npt.NDArray[np.float32],
    H2: npt.NDArray[np.float32],
    method: int,
) -> float: ...
def calcOpticalFlowFarneback(
    prev: npt.NDArray[np.uint8],
    next: npt.NDArray[np.uint8],
    flow: npt.NDArray[np.float32] | None,
    pyr_scale: float,
    levels: int,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float,
    flags: int,
) -> npt.NDArray[np.float32]: ...

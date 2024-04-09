from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .constants import *

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

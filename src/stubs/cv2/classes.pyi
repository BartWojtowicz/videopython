import builtins
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

retval: TypeAlias = Any

class VideoCapture(builtins.object):
    def __init__(self, filename: str) -> None: ...
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, npt.NDArray[np.uint8]]: ...
    def release(self) -> None: ...
    def get(self, propId) -> retval: ...

class CascadeClassifier(builtins.object):
    def __init__(self, filename: str) -> None: ...
    def detectMultiScale(
        self,
        image: npt.NDArray[np.uint8],
        scaleFactor: float = ...,
        minNeighbors: int = ...,
        flags: int = ...,
        minSize: tuple[int, int] = ...,
        maxSize: tuple[int, int] = ...,
    ) -> npt.NDArray[np.int32]: ...

class _Data:
    haarcascades: str

data: _Data

class _Dnn:
    def readNetFromCaffe(self, prototxt: str, caffeModel: str) -> Any: ...

dnn: _Dnn

class VideoWriter(builtins.object):
    def __init__(
        self,
        filename: str,
        fourcc: int,
        fps: float,
        frameSize: tuple[int, int],
        isColor: bool = ...,
    ) -> None: ...
    def write(self, image: npt.NDArray[np.uint8]) -> None: ...
    def release(self) -> None: ...
    def isOpened(self) -> bool: ...

import builtins
from typing import Any, TypeAlias

retval: TypeAlias = Any

class VideoCapture(builtins.object):
    def __init__(self, filename: str) -> None: ...
    def get(self, propId) -> retval:
        """
        @brief Returns the specified VideoCapture property

        @param propId Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...) or one from @ref videoio_flags_others @return Value for the specified property. Value 0 is returned when querying a property that is not supported by the backend used by the VideoCapture instance.  @note Reading / writing properties involves many layers. Some unexpected result might happens along this chain. @code{.txt} VideoCapture -> API Backend -> Operating System -> Device Driver -> Device Hardware @endcode The returned value might be different from what really used by the device or it could be encoded using device dependent rules (eg. steps or percentage). Effective behaviour depends from device driver and API Backend
        """

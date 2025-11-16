from typing import Iterator

import numpy as np

class KPipeline:
    def __init__(self, lang_code: str = "a") -> None: ...
    def __call__(self, text: str, voice: str = "af_heart") -> Iterator[tuple[int, int, np.ndarray]]: ...

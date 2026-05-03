from typing import Any, Literal

class _Dims:
    n_mels: int

class Whisper:
    dims: _Dims
    device: Any
    def transcribe(self, **kwargs: Any) -> dict[str, Any]: ...
    def detect_language(self, mel: Any) -> tuple[Any, dict[str, float] | list[dict[str, float]]]: ...

def load_model(
    name: Literal["tiny", "base", "small", "medium", "large", "turbo"],
    device: str | None = ...,
) -> Whisper: ...

class audio:
    SAMPLE_RATE: int
    N_SAMPLES: int

    @staticmethod
    def load_audio(file: str) -> Any: ...
    @staticmethod
    def resample(audio: Any, sample_rate: int) -> Any: ...
    @staticmethod
    def pad_or_trim(array: Any, length: int = ...) -> Any: ...
    @staticmethod
    def log_mel_spectrogram(audio: Any, n_mels: int = ...) -> Any: ...

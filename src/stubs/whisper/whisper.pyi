from typing import Any, Dict, List, Literal

class Whisper:
    def transcribe(self, audio: Any) -> Dict[str, Any]: ...

def load_model(name: Literal["tiny", "base", "small", "medium", "large", "turbo"]) -> WhisperModel: ...

class audio:
    SAMPLE_RATE: int

    @staticmethod
    def load_audio(file: str) -> Any: ...
    @staticmethod
    def resample(audio: Any, sample_rate: int) -> Any: ...

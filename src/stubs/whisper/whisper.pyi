from typing import Any, Dict, Literal

class Whisper:
    def transcribe(self, audio: Any, word_timestamps: bool) -> Dict[str, Any]: ...

def load_model(name: Literal["tiny", "base", "small", "medium", "large", "turbo"]) -> Whisper: ...

class audio:
    SAMPLE_RATE: int

    @staticmethod
    def load_audio(file: str) -> Any: ...
    @staticmethod
    def resample(audio: Any, sample_rate: int) -> Any: ...

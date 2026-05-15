"""Configuration model for the dubbing pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

TranslatorChoice = Literal["auto", "marian", "qwen3"]
WhisperModel = Literal["tiny", "base", "small", "medium", "large", "turbo"]


class DubbingConfig(BaseModel):
    """Knobs shared by :class:`VideoDubber` and :class:`LocalDubbingPipeline`.

    Replaces the nine constructor kwargs that used to be duplicated across
    ``VideoDubber.__init__``, ``LocalDubbingPipeline.__init__``, and their
    init-log lines. Adding a knob is now one edit.
    """

    model_config = ConfigDict(frozen=True)

    device: str | None = None
    low_memory: bool = False
    whisper_model: WhisperModel = "turbo"
    condition_on_previous_text: bool = False
    no_speech_threshold: float = 0.6
    logprob_threshold: float | None = -1.0
    vocabulary: list[str] | None = None
    strict_quality: bool = False
    translator: TranslatorChoice = "auto"

    @property
    def requested_device(self) -> str:
        """Lowercased device string for log lines (``"auto"`` when unset)."""
        return self.device.lower() if isinstance(self.device, str) else "auto"

    def init_log_fields(self) -> dict[str, object]:
        """Subset of fields surfaced in the init-log line.

        Hand-picked so log noise stays bounded as the config grows.
        """
        return {
            "device": self.requested_device,
            "low_memory": self.low_memory,
            "whisper_model": self.whisper_model,
            "translator": self.translator,
        }

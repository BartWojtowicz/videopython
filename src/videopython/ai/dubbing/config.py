"""Configuration model for the dubbing pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

WhisperModel = Literal["tiny", "base", "small", "medium", "large", "turbo"]


class DubbingConfig(BaseModel):
    """Knobs shared by :class:`VideoDubber` and :class:`LocalDubbingPipeline`.

    Accepted as either ``config=DubbingConfig(...)`` or flat kwargs on the
    two constructors; the flat path builds a ``DubbingConfig`` internally.

    Attributes:
        device: Execution device (``cpu``, ``cuda``, ``mps``, or ``None`` for auto).
        low_memory: When True, each pipeline stage (Whisper, Demucs, translation,
            Chatterbox TTS) is unloaded from memory after it runs, so only one
            model is resident at a time. Trades per-run latency (~10-30s of
            extra model loads) for a much lower memory ceiling. Recommended
            for GPUs with <=12GB VRAM or hosts with <32GB RAM. Default False.
        whisper_model: Whisper model size used for transcription. Larger
            models give better accuracy at the cost of VRAM and latency. One
            of ``tiny``, ``base``, ``small``, ``medium``, ``large``, ``turbo``.
            Default ``turbo``.
        condition_on_previous_text: Forwarded to ``AudioToText``. Defaults to
            ``False`` (Whisper's own default is ``True``). With conditioning
            on, a single hallucinated filler phrase cascades through the rest
            of the file. See ``AudioToText`` for the full rationale.
        no_speech_threshold: Forwarded to ``AudioToText``. Whisper's
            no-speech gate; raise to drop more low-confidence windows.
        logprob_threshold: Forwarded to ``AudioToText``. Whisper's average
            log-probability gate.
        vocabulary: Forwarded to ``AudioToText``. Optional list of brand
            names, product names, or proper nouns to bias Whisper's
            first-window decoder via ``initial_prompt``. Recovers
            near-mishears (e.g. Klarna -> "carna") on brand-monitoring
            inputs without new model deps.
        strict_quality: When True, the pipeline raises
            :class:`GarbageTranscriptError` before Demucs/translation/TTS
            run if the transcript-quality heuristic returns ``"reject"``.
            When False (default), low-quality transcripts are logged at
            WARNING but processing continues. Either way the
            :class:`TranscriptQuality` is exposed on ``DubbingResult`` for
            inspection.
        translator_model: Ollama tag for the translation model (``None`` uses the
            translator's default). ``translator_host`` sets the server URL.
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
    translator_model: str | None = None
    translator_host: str | None = None

    def init_log_fields(self) -> dict[str, object]:
        """Subset of fields surfaced in the init-log line.

        Hand-picked so log noise stays bounded as the config grows.
        """
        return {
            "device": self.device.lower() if isinstance(self.device, str) else "auto",
            "low_memory": self.low_memory,
            "whisper_model": self.whisper_model,
            "translator_model": self.translator_model,
        }

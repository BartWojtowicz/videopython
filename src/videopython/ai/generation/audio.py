"""Audio generation using local models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from videopython.ai._device import log_device_initialization, select_device
from videopython.ai._predictor import ManagedPredictor
from videopython.ai._revisions import pinned
from videopython.ai._text_chunks import split_text
from videopython.audio import Audio, AudioMetadata

if TYPE_CHECKING:
    from pathlib import Path


class TextToSpeech(ManagedPredictor):
    """Generates speech audio from text using Chatterbox Multilingual.

    Backed by Chatterbox Multilingual (Resemble AI). When ``voice_sample`` is
    provided to ``generate_audio``, the model clones that voice; otherwise it
    falls back to Chatterbox's built-in default speaker.
    """

    SAMPLE_RATE: int = 24000

    def __init__(
        self,
        voice: Audio | None = None,
        device: str | None = None,
        language: str = "en",
    ):
        self.voice = voice
        self.device = device
        self.language = language
        self._model: Any = None
        self._speech_graphs: Any = None

    def unload(self) -> None:
        if self._speech_graphs is not None:
            self._speech_graphs.close()
            self._speech_graphs = None
        super().unload()

    def _init_local(self) -> None:
        from videopython.ai._optional import require

        ChatterboxMultilingualTTS = require("chatterbox.mtl_tts", feature="TextToSpeech").ChatterboxMultilingualTTS
        from videopython.ai.generation._speech_tokens import (
            guard_speech_tokens,
            restrict_speech_vocabulary,
        )

        requested_device = self.device
        device = select_device(self.device, mps_allowed=False)

        # No repo id to key a revision on: Chatterbox resolves its own repo +
        # revision internally, so there is nothing to pass revision= to.
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        self._model.s3gen.inference = guard_speech_tokens(
            self._model.s3gen.inference, self._model.s3gen.flow.input_embedding.num_embeddings
        )
        restrict_speech_vocabulary(
            self._model.t3.speech_head,
            self._model.s3gen.flow.input_embedding.num_embeddings,
            self._model.t3.hp.stop_speech_token,
        )
        if device == "cuda":
            from videopython.ai.generation._speech_graphs import SpeechGraphs

            self._speech_graphs = SpeechGraphs(self._model)
        self.device = device
        log_device_initialization(
            "TextToSpeech",
            requested_device=requested_device,
            resolved_device=device,
        )

    def generate_audio(
        self,
        text: str,
        voice_sample: Audio | None = None,
        voice_sample_path: str | Path | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        temperature: float | None = None,
    ) -> Audio:
        """Generate speech audio from text.

        Args:
            text: Text to synthesize.
            voice_sample: Optional voice sample to clone. Falls back to the
                instance's ``voice`` and then to Chatterbox's default speaker.
            voice_sample_path: Optional pre-encoded WAV path to use directly as
                the speaker prompt. Skips the per-call temp-WAV encode that
                ``voice_sample`` would otherwise trigger. When set, takes
                precedence over ``voice_sample`` and ``self.voice``. Used by
                the dubbing pipeline to encode each speaker's sample once and
                reuse it across all of that speaker's segments.
            exaggeration: Chatterbox emotional-intensity knob (default
                ``0.5``). ``None`` (default) means do not pass the kwarg —
                Chatterbox uses its own default and we stay forward-compatible
                with changes to it. ``0.7+`` produces dramatic output.
            cfg_weight: Chatterbox classifier-free-guidance weight (default
                ``0.5``). ``None`` means do not pass. Lower values (~``0.3``)
                slow pacing.
            temperature: Chatterbox sampling temperature (default ``0.8``).
                ``None`` means do not pass.
        """
        import tempfile
        from pathlib import Path

        import numpy as np

        parts = split_text(text, 200)
        if not parts:
            raise ValueError("Speech text must not be empty")
        if self._model is None:
            self._init_local()

        speaker_wav_path: Path | None = None
        cleanup_path = False

        if voice_sample_path is not None:
            speaker_wav_path = Path(voice_sample_path)
        else:
            effective_sample = voice_sample or self.voice
            if effective_sample is not None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    effective_sample.save(f.name)
                    speaker_wav_path = Path(f.name)
                    cleanup_path = True

        # Only forward knobs the caller explicitly set. Passing nothing
        # for a knob lets Chatterbox use its own default — important so a
        # future Chatterbox default change doesn't get pinned by us.
        knobs: dict[str, float] = {}
        if exaggeration is not None:
            knobs["exaggeration"] = exaggeration
        if cfg_weight is not None:
            knobs["cfg_weight"] = cfg_weight
        if temperature is not None:
            knobs["temperature"] = temperature

        try:
            # Chatterbox caps generation at 1,000 speech tokens (~40 seconds).
            # Keep individual calls well below that cap, then synchronize the
            # complete parent utterance once in the dubbing pipeline.
            def synthesize(part: str, budget: int) -> list[np.ndarray]:
                from videopython.ai.generation._speech_tokens import InvalidSpeechTokens

                for attempt in range(3):
                    try:
                        wav = self._model.generate(
                            text=part,
                            language_id=self.language,
                            audio_prompt_path=str(speaker_wav_path) if speaker_wav_path else None,
                            **knobs,
                        )
                        break
                    except InvalidSpeechTokens:
                        if attempt == 2:
                            raise
                data = wav.cpu().float().numpy().reshape(-1)
                if not len(data) or not np.isfinite(data).all():
                    raise ValueError("Speech generation returned empty or non-finite audio")
                # Duration is a conservative cap warning, not proof of coverage.
                # Discard suspect audio and regenerate smaller text units.
                if len(data) >= 38 * self.SAMPLE_RATE:
                    smaller = split_text(part, max(1, budget // 2))
                    if len(smaller) < 2 or budget <= 40:
                        raise RuntimeError("Speech generation may have reached its token limit")
                    return [audio for chunk in smaller for audio in synthesize(chunk, budget // 2)]
                return [data]

            arrays = [audio for part in parts for audio in synthesize(part, 200)]
            # Keep natural leading/trailing pauses; do not crossfade phonemes.
            audio_data = np.concatenate(arrays) if len(arrays) > 1 else arrays[0]

            metadata = AudioMetadata(
                sample_rate=self.SAMPLE_RATE,
                channels=1,
                sample_width=2,
                duration_seconds=len(audio_data) / self.SAMPLE_RATE,
                frame_count=len(audio_data),
            )
            return Audio(audio_data, metadata)
        finally:
            if cleanup_path and speaker_wav_path is not None:
                speaker_wav_path.unlink(missing_ok=True)


class TextToMusic(ManagedPredictor):
    """Generates music from text descriptions using MusicGen."""

    _model_attrs = ("_model", "_processor")

    def __init__(self, device: str | None = None):
        self.device = device
        self._processor: Any = None
        self._model: Any = None

    def _init_local(self) -> None:
        """Initialize local MusicGen model."""
        import os

        from videopython.ai._optional import require

        _transformers = require("transformers", feature="TextToMusic")
        AutoProcessor = _transformers.AutoProcessor
        MusicgenForConditionalGeneration = _transformers.MusicgenForConditionalGeneration

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        requested_device = self.device
        device = select_device(self.device, mps_allowed=True)

        model_name = "facebook/musicgen-small"
        self._processor = AutoProcessor.from_pretrained(model_name, revision=pinned(model_name))
        self._model = MusicgenForConditionalGeneration.from_pretrained(model_name, revision=pinned(model_name))
        self._model.to(device)
        self.device = device
        log_device_initialization(
            "TextToMusic",
            requested_device=requested_device,
            resolved_device=device,
        )

    def generate_audio(self, text: str, max_new_tokens: int = 256) -> Audio:
        """Generate music audio from text description."""
        if self._model is None:
            self._init_local()

        inputs = self._processor(text=[text], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        audio_values = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        sampling_rate = self._model.config.audio_encoder.sampling_rate

        audio_data = audio_values[0, 0].cpu().float().numpy()

        metadata = AudioMetadata(
            sample_rate=sampling_rate,
            channels=1,
            sample_width=2,
            duration_seconds=len(audio_data) / sampling_rate,
            frame_count=len(audio_data),
        )
        return Audio(audio_data, metadata)

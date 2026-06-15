"""Pluggable speech-synthesis backend protocol for the dubbing pipeline.

Mirrors :class:`videopython.ai.generation.translation.TranslationBackend`: a
dependency-free, ``runtime_checkable`` Protocol that the dubbing pipeline
depends on instead of binding directly to the local
:class:`videopython.ai.generation.audio.TextToSpeech` implementation.

This is the seam that lets dubbing run WITHOUT chatterbox in the process. The
local ``TextToSpeech`` (which pulls ``chatterbox-tts`` via the ``[tts]`` extra)
satisfies this protocol structurally — no changes needed there. A consumer that
can't or won't install chatterbox (e.g. a service running synthesis in a
separate process or on a remote/Modal function) supplies its own object
implementing :meth:`SpeechBackend.synthesize` and injects it; the pipeline never
imports chatterbox in that case. videopython ships ONLY this protocol plus the
local backend — no reference remote/HTTP backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from videopython.audio import Audio


@runtime_checkable
class SpeechBackend(Protocol):
    """Pipeline-facing text-to-speech interface.

    The local :class:`videopython.ai.generation.audio.TextToSpeech` satisfies
    this structurally. The dubbing pipeline only depends on
    :meth:`generate_audio`; the keyword-only knobs mirror the local backend's
    Chatterbox knobs so a remote backend can accept and honour (or ignore)
    them. Implementations that need teardown may also expose ``unload()``; the
    pipeline calls it opportunistically (``getattr`` guarded) in low-memory mode.
    """

    def generate_audio(
        self,
        text: str,
        voice_sample: Audio | None = ...,
        voice_sample_path: str | Path | None = ...,
        exaggeration: float | None = ...,
        cfg_weight: float | None = ...,
        temperature: float | None = ...,
    ) -> Audio:
        """Synthesize ``text`` into an :class:`~videopython.audio.Audio`.

        ``voice_sample`` / ``voice_sample_path`` carry the speaker prompt for
        voice cloning (path takes precedence and skips the per-call WAV encode);
        the remaining knobs are optional expressiveness controls where ``None``
        means "use the backend's own default".
        """
        ...

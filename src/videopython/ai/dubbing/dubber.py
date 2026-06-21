"""Main local video dubbing interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from videopython.ai.dubbing.config import DubbingConfig
from videopython.ai.dubbing.models import DubbingResult, RevoiceResult

if TYPE_CHECKING:
    from videopython.ai.generation._tts_backend import SpeechBackend
    from videopython.base.video import Video

logger = logging.getLogger(__name__)


class VideoDubber:
    """Dubs videos into different languages using the local pipeline.

    Accepts either a :class:`DubbingConfig` or the same knobs as flat kwargs
    (``device``, ``low_memory``, ``whisper_model``, ``translator``, etc.) --
    the flat path builds a ``DubbingConfig`` internally. See
    :class:`DubbingConfig` for the full knob list and defaults.
    """

    def __init__(
        self,
        config: DubbingConfig | None = None,
        *,
        tts_backend: SpeechBackend | None = None,
        **kwargs: Any,
    ):
        if config is not None and kwargs:
            raise TypeError("Pass either `config=` or knob kwargs, not both")
        self.config = config or DubbingConfig(**kwargs)
        # Optional injected speech backend. None -> the pipeline lazily builds
        # the local chatterbox-backed TextToSpeech (requires the [tts] extra).
        # Inject a SpeechBackend to dub with only [dub] installed.
        self._tts_backend = tts_backend
        self._local_pipeline: Any = None
        logger.info(
            "VideoDubber initialized with %s",
            " ".join(f"{k}={v}" for k, v in self.config.init_log_fields().items()),
        )

    def _init_local_pipeline(self) -> None:
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        self._local_pipeline = LocalDubbingPipeline(config=self.config, tts_backend=self._tts_backend)

    def dub(
        self,
        video: Video,
        target_lang: str,
        source_lang: str | None = None,
        preserve_background: bool = True,
        voice_clone: bool = True,
        enable_diarization: bool = False,
        progress_callback: Callable[[str, float], None] | None = None,
        transcription: Any = None,
    ) -> DubbingResult:
        """Dub a video into a target language.

        Args:
            enable_diarization: Enable speaker diarization to clone each speaker's
                voice separately. With ``transcription=None``, runs alongside Whisper.
                With a supplied ``transcription`` that has no speakers, runs pyannote
                standalone and overlays speakers onto the supplied words. Ignored when
                the supplied transcription already has speaker labels.
            transcription: Optional pre-computed ``Transcription`` to skip the Whisper
                step. Speaker labels on the supplied transcription drive per-speaker
                voice cloning. If it has no speakers, pass ``enable_diarization=True``
                to add them via pyannote (requires word-level timings).
        """
        if self._local_pipeline is None:
            self._init_local_pipeline()

        return self._local_pipeline.process(
            source_audio=video.audio,
            target_lang=target_lang,
            source_lang=source_lang,
            preserve_background=preserve_background,
            voice_clone=voice_clone,
            enable_diarization=enable_diarization,
            progress_callback=progress_callback,
            transcription=transcription,
        )

    def dub_and_replace(
        self,
        video: Video,
        target_lang: str,
        source_lang: str | None = None,
        preserve_background: bool = True,
        voice_clone: bool = True,
        enable_diarization: bool = False,
        progress_callback: Callable[[str, float], None] | None = None,
        transcription: Any = None,
    ) -> Video:
        """Dub a video and return a new video with the dubbed audio.

        Args:
            transcription: Optional pre-computed ``Transcription`` to skip the Whisper
                step. Speaker labels on the supplied transcription drive per-speaker
                voice cloning. See ``dub()`` for the interaction with
                ``enable_diarization``.
        """
        result = self.dub(
            video=video,
            target_lang=target_lang,
            source_lang=source_lang,
            preserve_background=preserve_background,
            voice_clone=voice_clone,
            enable_diarization=enable_diarization,
            progress_callback=progress_callback,
            transcription=transcription,
        )
        return video.add_audio(result.dubbed_audio, overlay=False)

    def dub_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_lang: str,
        source_lang: str | None = None,
        preserve_background: bool = True,
        voice_clone: bool = True,
        enable_diarization: bool = False,
        progress_callback: Callable[[str, float], None] | None = None,
        transcription: Any = None,
        keep_original_audio: bool = False,
    ) -> DubbingResult:
        """Dub a video file in place on disk without loading video frames into memory.

        Extracts the audio track via ffmpeg, runs the dubbing pipeline on the
        audio only, then muxes the dubbed audio back into the source video
        using ffmpeg stream-copy (no video re-encode). Peak memory is bounded
        by model weights and the audio track — independent of video length and
        resolution.

        Use this instead of ``dub_and_replace`` when the source video is long
        or high-resolution and you don't need frame-level access in Python.

        Args:
            input_path: Path to the source video file.
            output_path: Path to write the dubbed video. Overwritten if it exists.
            target_lang: Target language code (e.g. ``"es"``, ``"fr"``).
            source_lang: Source language code, or ``None`` to auto-detect.
            preserve_background: Preserve background music/effects via source separation.
            voice_clone: Clone the source speaker's voice for the dubbed track.
            enable_diarization: Enable speaker diarization for per-speaker voice cloning.
                See ``dub()`` for the interaction with ``transcription``.
            progress_callback: Optional callback ``(stage: str, progress: float) -> None``.
            transcription: Optional pre-computed ``Transcription`` to skip the Whisper
                step. Speaker labels on the supplied transcription drive per-speaker
                voice cloning. If it has no speakers, pass ``enable_diarization=True``
                to add them via pyannote (requires word-level timings).
            keep_original_audio: If True, retain the source audio in the output
                as a secondary track behind the dubbed one (editorial A/B).

        Returns:
            ``DubbingResult`` with the dubbed audio, translated segments, and
            source transcription. The output video is written to ``output_path``.
        """
        from videopython.ai.dubbing.remux import replace_audio_stream_from_audio
        from videopython.audio import Audio

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        logger.info("dub_file: loading audio from %s", input_path)
        source_audio = Audio.from_path(input_path)

        if self._local_pipeline is None:
            self._init_local_pipeline()

        result = self._local_pipeline.process(
            source_audio=source_audio,
            target_lang=target_lang,
            source_lang=source_lang,
            preserve_background=preserve_background,
            voice_clone=voice_clone,
            enable_diarization=enable_diarization,
            progress_callback=progress_callback,
            transcription=transcription,
        )

        # Stream the dubbed Audio directly into ffmpeg via stdin instead of
        # going through a temp WAV on disk. For a 2h dub the temp file would
        # be ~10 GB written-then-read; the streaming path drops both copies.
        replace_audio_stream_from_audio(
            video_path=input_path,
            audio=result.dubbed_audio,
            output_path=output_path,
            keep_original_audio=keep_original_audio,
        )

        return result

    def revoice(
        self,
        video: Video,
        text: str,
        preserve_background: bool = True,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> RevoiceResult:
        """Replace speech in a video with new text using voice cloning."""
        if self._local_pipeline is None:
            self._init_local_pipeline()

        return self._local_pipeline.revoice(
            source_audio=video.audio,
            text=text,
            preserve_background=preserve_background,
            progress_callback=progress_callback,
        )

    def revoice_and_replace(
        self,
        video: Video,
        text: str,
        preserve_background: bool = True,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Video:
        """Revoice a video and return a new video with the revoiced audio."""
        result = self.revoice(
            video=video,
            text=text,
            preserve_background=preserve_background,
            progress_callback=progress_callback,
        )

        speech_duration = result.speech_duration
        video_duration = video.total_seconds

        if video_duration > speech_duration:
            output_video = video[: round(speech_duration * video.fps)]
        else:
            output_video = video

        return output_video.add_audio(result.revoiced_audio, overlay=False)

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        from videopython.ai.generation.translation import OllamaTranslator

        return OllamaTranslator.get_supported_languages()

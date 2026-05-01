"""Main local video dubbing interface."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from videopython.ai.dubbing.models import DubbingResult, RevoiceResult
from videopython.ai.dubbing.pipeline import WhisperModel

if TYPE_CHECKING:
    from videopython.base.video import Video

logger = logging.getLogger(__name__)


class VideoDubber:
    """Dubs videos into different languages using the local pipeline.

    Args:
        device: Execution device (``cpu``, ``cuda``, ``mps``, or ``None`` for auto).
        low_memory: When True, each pipeline stage (Whisper, Demucs, MarianMT,
            Chatterbox TTS) is unloaded from memory after it runs, so only one
            model is resident at a time. Trades per-run latency (~10-30s of
            extra model loads) for a much lower memory ceiling. Recommended for
            GPUs with <=12GB VRAM or hosts with <32GB RAM. Default False.
        whisper_model: Whisper model size used for transcription. Larger models
            give better accuracy at the cost of VRAM and latency. One of
            ``tiny``, ``base``, ``small``, ``medium``, ``large``, ``turbo``.
            Default ``small``.
    """

    def __init__(
        self,
        device: str | None = None,
        low_memory: bool = False,
        whisper_model: WhisperModel = "small",
    ):
        self.device = device
        self.low_memory = low_memory
        self.whisper_model = whisper_model
        self._local_pipeline: Any = None
        requested = device.lower() if isinstance(device, str) else "auto"
        logger.info(
            "VideoDubber initialized with device=%s low_memory=%s whisper_model=%s",
            requested,
            low_memory,
            whisper_model,
        )

    def _init_local_pipeline(self) -> None:
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        self._local_pipeline = LocalDubbingPipeline(
            device=self.device,
            low_memory=self.low_memory,
            whisper_model=self.whisper_model,
        )

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

        Returns:
            ``DubbingResult`` with the dubbed audio, translated segments, and
            source transcription. The output video is written to ``output_path``.
        """
        from videopython.ai.dubbing.remux import replace_audio_stream
        from videopython.base.audio import Audio

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

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            dubbed_audio_path = Path(tmp.name)
        try:
            result.dubbed_audio.save(dubbed_audio_path)
            replace_audio_stream(
                video_path=input_path,
                audio_path=dubbed_audio_path,
                output_path=output_path,
            )
        finally:
            dubbed_audio_path.unlink(missing_ok=True)

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
            output_video = video.cut(0, speech_duration)
        else:
            output_video = video

        return output_video.add_audio(result.revoiced_audio, overlay=False)

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        from videopython.ai.generation.translation import TextTranslator

        return TextTranslator.get_supported_languages()

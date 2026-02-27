"""Main local video dubbing interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from videopython.ai.dubbing.models import DubbingResult, RevoiceResult

if TYPE_CHECKING:
    from videopython.base.video import Video


class VideoDubber:
    """Dubs videos into different languages using the local pipeline."""

    def __init__(self, device: str | None = None):
        self.device = device
        self._local_pipeline: Any = None

    def _init_local_pipeline(self) -> None:
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        self._local_pipeline = LocalDubbingPipeline(device=self.device)

    def dub(
        self,
        video: Video,
        target_lang: str,
        source_lang: str | None = None,
        preserve_background: bool = True,
        voice_clone: bool = True,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> DubbingResult:
        """Dub a video into a target language."""
        if self._local_pipeline is None:
            self._init_local_pipeline()

        return self._local_pipeline.process(
            video=video,
            target_lang=target_lang,
            source_lang=source_lang,
            preserve_background=preserve_background,
            voice_clone=voice_clone,
            progress_callback=progress_callback,
        )

    def dub_and_replace(
        self,
        video: Video,
        target_lang: str,
        source_lang: str | None = None,
        preserve_background: bool = True,
        voice_clone: bool = True,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Video:
        """Dub a video and return a new video with the dubbed audio."""
        result = self.dub(
            video=video,
            target_lang=target_lang,
            source_lang=source_lang,
            preserve_background=preserve_background,
            voice_clone=voice_clone,
            progress_callback=progress_callback,
        )
        return video.add_audio(result.dubbed_audio, overlay=False)

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
            video=video,
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

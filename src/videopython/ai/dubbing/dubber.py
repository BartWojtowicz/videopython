"""Main video dubbing class with multi-backend support."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from videopython.ai.backends import UnsupportedBackendError, VideoDubberBackend, get_api_key
from videopython.ai.config import get_default_backend
from videopython.ai.dubbing.models import DubbingResult, RevoiceResult

if TYPE_CHECKING:
    from videopython.base.video import Video


# Supported languages for ElevenLabs dubbing
ELEVENLABS_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "hi": "Hindi",
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "fi": "Finnish",
    "el": "Greek",
    "he": "Hebrew",
    "id": "Indonesian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nb": "Norwegian",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sv": "Swedish",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


class VideoDubber:
    """Dubs videos into different languages.

    Supports two backends:
    - elevenlabs: Cloud-based end-to-end solution via ElevenLabs Dubbing API
    - local: Local pipeline using transcription, translation, and TTS

    Example:
        >>> from videopython.ai.dubbing import VideoDubber
        >>> from videopython.base.video import Video
        >>>
        >>> # Using ElevenLabs (cloud)
        >>> dubber = VideoDubber(backend="elevenlabs")
        >>> video = Video.from_path("video.mp4")
        >>> result = dubber.dub(video, target_lang="es")
        >>>
        >>> # Using local pipeline
        >>> dubber = VideoDubber(backend="local")
        >>> result = dubber.dub(video, target_lang="es", preserve_background=True)
    """

    SUPPORTED_BACKENDS: list[str] = ["elevenlabs", "local"]

    def __init__(
        self,
        backend: VideoDubberBackend | None = None,
        api_key: str | None = None,
        translation_backend: str = "openai",
        tts_backend: str = "local",
        device: str | None = None,
    ):
        """Initialize the video dubber.

        Args:
            backend: Backend to use ('elevenlabs' or 'local').
                If None, uses config default or 'local'.
            api_key: API key for cloud backends. If None, reads from environment.
            translation_backend: Backend for text translation in local pipeline
                ('openai', 'gemini', or 'local'). Default: 'openai'.
            tts_backend: Backend for text-to-speech in local pipeline
                ('local', 'openai', 'elevenlabs'). Default: 'local'.
            device: Device for local models ('cuda', 'mps', or 'cpu').
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("video_dubber")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: VideoDubberBackend = resolved_backend  # type: ignore[assignment]
        self.api_key = api_key
        self.translation_backend = translation_backend
        self.tts_backend = tts_backend
        self.device = device

        # Lazy-loaded components
        self._local_pipeline: Any = None

    def _init_local_pipeline(self) -> None:
        """Initialize the local dubbing pipeline."""
        from videopython.ai.dubbing.pipeline import LocalDubbingPipeline

        self._local_pipeline = LocalDubbingPipeline(
            translation_backend=self.translation_backend,
            tts_backend=self.tts_backend,
            device=self.device,
        )

    def _dub_elevenlabs(
        self,
        video: Video,
        target_lang: str,
        source_lang: str | None,
        num_speakers: int,
        progress_callback: Callable[[str, float], None] | None,
    ) -> DubbingResult:
        """Dub video using ElevenLabs Dubbing API."""
        import requests

        from videopython.base.audio import Audio
        from videopython.base.text.transcription import Transcription

        api_key = get_api_key("elevenlabs", self.api_key)

        if target_lang not in ELEVENLABS_LANGUAGES:
            raise ValueError(
                f"Unsupported target language: {target_lang}. Supported: {list(ELEVENLABS_LANGUAGES.keys())}"
            )

        if source_lang is not None and source_lang not in ELEVENLABS_LANGUAGES:
            raise ValueError(
                f"Unsupported source language: {source_lang}. Supported: {list(ELEVENLABS_LANGUAGES.keys())}"
            )

        # Save video to temp file for upload
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video.save(f.name)
            video_path = Path(f.name)

        try:
            if progress_callback:
                progress_callback("Uploading video to ElevenLabs", 0.1)

            # Create dubbing project
            url = "https://api.elevenlabs.io/v1/dubbing"
            headers = {"xi-api-key": api_key}

            with open(video_path, "rb") as video_file:
                files = {"file": ("video.mp4", video_file, "video/mp4")}
                data = {
                    "target_lang": target_lang,
                    "num_speakers": num_speakers,
                    "watermark": False,
                }
                if source_lang:
                    data["source_lang"] = source_lang

                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                dubbing_id = response.json()["dubbing_id"]

            if progress_callback:
                progress_callback("Processing dubbing", 0.3)

            # Poll for completion
            status_url = f"https://api.elevenlabs.io/v1/dubbing/{dubbing_id}"
            while True:
                response = requests.get(status_url, headers=headers)
                response.raise_for_status()
                status_data = response.json()

                status = status_data.get("status")
                if status == "dubbed":
                    break
                elif status == "failed":
                    error = status_data.get("error", "Unknown error")
                    raise RuntimeError(f"ElevenLabs dubbing failed: {error}")

                if progress_callback:
                    # Update progress based on status
                    if status == "transcribing":
                        progress_callback("Transcribing audio", 0.4)
                    elif status == "translating":
                        progress_callback("Translating content", 0.5)
                    elif status == "dubbing":
                        progress_callback("Generating dubbed audio", 0.7)

                time.sleep(5)  # Poll every 5 seconds

            if progress_callback:
                progress_callback("Downloading dubbed audio", 0.9)

            # Download dubbed audio
            audio_url = f"https://api.elevenlabs.io/v1/dubbing/{dubbing_id}/audio/{target_lang}"
            response = requests.get(audio_url, headers=headers)
            response.raise_for_status()

            # Save audio to temp file and load
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(response.content)
                dubbed_audio_path = Path(f.name)

            dubbed_audio = Audio.from_path(dubbed_audio_path)
            dubbed_audio_path.unlink()

            if progress_callback:
                progress_callback("Complete", 1.0)

            # Create minimal result (ElevenLabs doesn't return detailed data)
            return DubbingResult(
                dubbed_audio=dubbed_audio,
                translated_segments=[],  # Not available from API
                source_transcription=Transcription(segments=[]),
                source_lang=source_lang or "auto",
                target_lang=target_lang,
                separated_audio=None,
                voice_samples={},
            )

        finally:
            video_path.unlink()

    def _dub_local(
        self,
        video: Video,
        target_lang: str,
        source_lang: str | None,
        preserve_background: bool,
        voice_clone: bool,
        progress_callback: Callable[[str, float], None] | None,
    ) -> DubbingResult:
        """Dub video using local pipeline."""
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

    def dub(
        self,
        video: Video,
        target_lang: str,
        source_lang: str | None = None,
        preserve_background: bool = True,
        voice_clone: bool = True,
        num_speakers: int = 1,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> DubbingResult:
        """Dub a video into a target language.

        Args:
            video: The video to dub.
            target_lang: Target language code (e.g., "es" for Spanish).
            source_lang: Source language code. If None, auto-detected.
            preserve_background: Keep background audio (music, sound effects).
                Only applies to local backend.
            voice_clone: Clone original speaker voices. Only applies to local backend.
            num_speakers: Expected number of speakers. Used by ElevenLabs for
                better speaker separation.
            progress_callback: Optional callback for progress updates.
                Called with (stage_name, progress_fraction).

        Returns:
            DubbingResult containing the dubbed audio and metadata.

        Raises:
            ValueError: If language codes are invalid.
            RuntimeError: If dubbing fails.
        """
        if self.backend == "elevenlabs":
            return self._dub_elevenlabs(
                video=video,
                target_lang=target_lang,
                source_lang=source_lang,
                num_speakers=num_speakers,
                progress_callback=progress_callback,
            )
        elif self.backend == "local":
            return self._dub_local(
                video=video,
                target_lang=target_lang,
                source_lang=source_lang,
                preserve_background=preserve_background,
                voice_clone=voice_clone,
                progress_callback=progress_callback,
            )
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)

    def dub_and_replace(
        self,
        video: Video,
        target_lang: str,
        source_lang: str | None = None,
        preserve_background: bool = True,
        voice_clone: bool = True,
        num_speakers: int = 1,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> Video:
        """Dub a video and return a new video with the dubbed audio.

        Convenience method that combines dubbing with audio replacement.

        Args:
            video: The video to dub.
            target_lang: Target language code.
            source_lang: Source language code (optional).
            preserve_background: Keep background audio (local backend only).
            voice_clone: Clone speaker voices (local backend only).
            num_speakers: Expected number of speakers (ElevenLabs only).
            progress_callback: Optional progress callback.

        Returns:
            New Video with dubbed audio track.
        """
        result = self.dub(
            video=video,
            target_lang=target_lang,
            source_lang=source_lang,
            preserve_background=preserve_background,
            voice_clone=voice_clone,
            num_speakers=num_speakers,
            progress_callback=progress_callback,
        )

        # Replace audio in video (overlay=False to replace, not overlay)
        return video.add_audio(result.dubbed_audio, overlay=False)

    def revoice(
        self,
        video: Video,
        text: str,
        preserve_background: bool = True,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> RevoiceResult:
        """Replace speech in a video with new text using voice cloning.

        Uses the original speaker's voice to generate new speech with the
        provided text. Background audio (music, effects) can be preserved.

        Only available with the local backend.

        Args:
            video: The video to revoice.
            text: New text for the speaker to say.
            preserve_background: Keep background audio (music, sound effects).
            progress_callback: Optional callback for progress updates.
                Called with (stage_name, progress_fraction).

        Returns:
            RevoiceResult containing the revoiced audio and metadata.

        Raises:
            UnsupportedBackendError: If using a backend other than 'local'.

        Example:
            >>> dubber = VideoDubber(backend="local")
            >>> result = dubber.revoice(video, "Hello, this is my new message!")
            >>> new_video = video.add_audio(result.revoiced_audio, overlay=False)
        """
        if self.backend != "local":
            raise UnsupportedBackendError(self.backend, ["local"])

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
        """Revoice a video and return a new video with the revoiced audio.

        Convenience method that combines revoicing with audio replacement.
        The output video duration matches the generated speech duration.

        Only available with the local backend.

        Args:
            video: The video to revoice.
            text: New text for the speaker to say.
            preserve_background: Keep background audio (music, sound effects).
            progress_callback: Optional progress callback.

        Returns:
            New Video with revoiced audio track. Duration matches speech length.
        """
        result = self.revoice(
            video=video,
            text=text,
            preserve_background=preserve_background,
            progress_callback=progress_callback,
        )

        # Trim or extend video to match audio duration
        speech_duration = result.speech_duration
        video_duration = video.total_seconds

        if video_duration > speech_duration:
            # Trim video to match speech
            output_video = video.cut(0, speech_duration)
        else:
            # Use full video (speech will extend beyond video if longer)
            output_video = video

        return output_video.add_audio(result.revoiced_audio, overlay=False)

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        """Get dictionary of supported language codes and names.

        Returns:
            Dictionary mapping language codes to language names.
        """
        return ELEVENLABS_LANGUAGES.copy()

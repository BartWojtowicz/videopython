"""Local dubbing pipeline that combines transcription, translation, and TTS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from videopython.ai.dubbing.models import DubbingResult, RevoiceResult, SeparatedAudio
from videopython.ai.dubbing.timing import TimingSynchronizer

if TYPE_CHECKING:
    from videopython.base.video import Video


class LocalDubbingPipeline:
    """Local pipeline for video dubbing.

    Combines multiple AI components:
    1. AudioToText for transcription
    2. AudioSeparator for separating speech from background (optional)
    3. TextTranslator for translation
    4. TextToSpeech with voice cloning for generating dubbed speech
    5. TimingSynchronizer for matching timing

    Example:
        >>> from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
        >>> from videopython.base.video import Video
        >>>
        >>> pipeline = LocalDubbingPipeline()
        >>> video = Video.from_path("video.mp4")
        >>> result = pipeline.process(video, target_lang="es")
    """

    def __init__(
        self,
        translation_backend: str = "openai",
        tts_backend: str = "local",
        transcription_backend: str = "local",
        device: str | None = None,
    ):
        """Initialize the local dubbing pipeline.

        Args:
            translation_backend: Backend for text translation ('openai', 'gemini', 'local').
            tts_backend: Backend for text-to-speech ('local', 'openai', 'elevenlabs').
            transcription_backend: Backend for transcription ('local', 'openai', 'gemini').
            device: Device for local models ('cuda', 'mps', 'cpu').
        """
        self.translation_backend = translation_backend
        self.tts_backend = tts_backend
        self.transcription_backend = transcription_backend
        self.device = device

        # Lazy-loaded components
        self._transcriber: Any = None
        self._translator: Any = None
        self._tts: Any = None
        self._separator: Any = None
        self._synchronizer: TimingSynchronizer | None = None

    def _init_transcriber(self) -> None:
        """Initialize the transcription model."""
        from videopython.ai.understanding.audio import AudioToText

        self._transcriber = AudioToText(
            backend=self.transcription_backend,  # type: ignore
            device=self.device or "cpu",
        )

    def _init_translator(self) -> None:
        """Initialize the translation model."""
        from videopython.ai.generation.translation import TextTranslator

        self._translator = TextTranslator(
            backend=self.translation_backend,  # type: ignore
        )

    def _init_tts(self, voice_clone: bool = False) -> None:
        """Initialize the text-to-speech model."""
        from videopython.ai.generation.audio import TextToSpeech

        # For voice cloning, we need local XTTS backend
        if voice_clone and self.tts_backend == "local":
            self._tts = TextToSpeech(
                backend="local",
                model_size="xtts",  # Use XTTS for voice cloning
                device=self.device,
            )
        else:
            self._tts = TextToSpeech(
                backend=self.tts_backend,  # type: ignore
                device=self.device,
            )

    def _init_separator(self) -> None:
        """Initialize the audio separator."""
        from videopython.ai.understanding.separation import AudioSeparator

        self._separator = AudioSeparator(device=self.device)

    def _init_synchronizer(self) -> None:
        """Initialize the timing synchronizer."""
        self._synchronizer = TimingSynchronizer()

    def _extract_voice_samples(
        self,
        audio: Any,
        transcription: Any,
        min_duration: float = 3.0,
        max_duration: float = 10.0,
    ) -> dict[str, Any]:
        """Extract voice samples for each speaker from the audio.

        Args:
            audio: Source audio.
            transcription: Transcription with speaker information.
            min_duration: Minimum sample duration in seconds.
            max_duration: Maximum sample duration in seconds.

        Returns:
            Dictionary mapping speaker IDs to Audio samples.
        """
        from videopython.base.audio import Audio

        voice_samples: dict[str, Audio] = {}

        # Group segments by speaker
        segments_by_speaker: dict[str, list[Any]] = {}
        for segment in transcription.segments:
            speaker = segment.speaker or "speaker_0"
            if speaker not in segments_by_speaker:
                segments_by_speaker[speaker] = []
            segments_by_speaker[speaker].append(segment)

        # Extract best sample for each speaker
        for speaker, segments in segments_by_speaker.items():
            # Find segment with duration closest to target
            target_duration = 6.0  # XTTS works best with ~6 second samples
            best_segment = None
            best_diff = float("inf")

            for segment in segments:
                duration = segment.end - segment.start
                if duration >= min_duration:
                    diff = abs(duration - target_duration)
                    if diff < best_diff:
                        best_diff = diff
                        best_segment = segment

            if best_segment is not None:
                start = best_segment.start
                end = min(best_segment.end, start + max_duration)
                voice_samples[speaker] = audio.slice(start, end)

        return voice_samples

    def process(
        self,
        video: Video,
        target_lang: str,
        source_lang: str | None = None,
        preserve_background: bool = True,
        voice_clone: bool = True,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> DubbingResult:
        """Process a video through the dubbing pipeline.

        Args:
            video: Video to dub.
            target_lang: Target language code.
            source_lang: Source language code (optional, auto-detected if None).
            preserve_background: Preserve background audio (music, effects).
            voice_clone: Clone original speaker voices using XTTS.
            progress_callback: Optional progress callback (stage_name, progress).

        Returns:
            DubbingResult with dubbed audio and metadata.
        """
        from videopython.base.audio import Audio

        def report_progress(stage: str, progress: float) -> None:
            if progress_callback:
                progress_callback(stage, progress)

        # Step 1: Transcribe audio
        report_progress("Transcribing audio", 0.05)
        if self._transcriber is None:
            self._init_transcriber()

        source_audio = video.audio
        transcription = self._transcriber.transcribe(source_audio)

        if not transcription.segments:
            # No speech found, return original audio
            return DubbingResult(
                dubbed_audio=source_audio,
                translated_segments=[],
                source_transcription=transcription,
                source_lang=source_lang or "unknown",
                target_lang=target_lang,
            )

        # Detect source language if not provided
        detected_lang = source_lang or "en"  # Default to English

        # Step 2: Separate audio (if preserving background)
        separated_audio: SeparatedAudio | None = None
        vocal_audio = source_audio

        if preserve_background:
            report_progress("Separating audio", 0.15)
            if self._separator is None:
                self._init_separator()

            separated_audio = self._separator.separate(source_audio)
            vocal_audio = separated_audio.vocals

        # Step 3: Extract voice samples (if voice cloning)
        voice_samples: dict[str, Audio] = {}
        if voice_clone:
            report_progress("Extracting voice samples", 0.25)
            voice_samples = self._extract_voice_samples(vocal_audio, transcription)

        # Step 4: Translate segments
        report_progress("Translating text", 0.35)
        if self._translator is None:
            self._init_translator()

        translated_segments = self._translator.translate_segments(
            segments=transcription.segments,
            target_lang=target_lang,
            source_lang=detected_lang,
        )

        # Step 5: Generate dubbed speech
        report_progress("Generating dubbed speech", 0.50)
        if self._tts is None:
            self._init_tts(voice_clone=voice_clone)

        dubbed_segments: list[Audio] = []
        target_durations: list[float] = []
        start_times: list[float] = []

        for i, segment in enumerate(translated_segments):
            progress = 0.50 + (0.30 * (i / len(translated_segments)))
            report_progress(f"Generating speech ({i + 1}/{len(translated_segments)})", progress)

            # Get voice sample for this speaker
            speaker = segment.speaker or "speaker_0"
            voice_sample = voice_samples.get(speaker)

            # Generate speech
            if voice_clone and voice_sample is not None:
                dubbed_audio = self._tts.generate_audio(
                    segment.translated_text,
                    voice_sample=voice_sample,
                )
            else:
                dubbed_audio = self._tts.generate_audio(segment.translated_text)

            dubbed_segments.append(dubbed_audio)
            target_durations.append(segment.duration)
            start_times.append(segment.start)

        # Step 6: Synchronize timing
        report_progress("Synchronizing timing", 0.85)
        if self._synchronizer is None:
            self._init_synchronizer()
        assert self._synchronizer is not None

        synchronized_segments, _ = self._synchronizer.synchronize_segments(dubbed_segments, target_durations)

        # Step 7: Assemble final audio
        report_progress("Assembling final audio", 0.90)
        total_duration = source_audio.metadata.duration_seconds

        dubbed_speech = self._synchronizer.assemble_with_timing(synchronized_segments, start_times, total_duration)

        # Mix with background if available
        if separated_audio is not None:
            # Resample dubbed speech to match background sample rate if needed
            background_sr = separated_audio.background.metadata.sample_rate
            if dubbed_speech.metadata.sample_rate != background_sr:
                dubbed_speech = dubbed_speech.resample(background_sr)

            # Overlay dubbed speech on background
            final_audio = separated_audio.background.overlay(dubbed_speech, position=0.0)
        else:
            final_audio = dubbed_speech

        report_progress("Complete", 1.0)

        return DubbingResult(
            dubbed_audio=final_audio,
            translated_segments=translated_segments,
            source_transcription=transcription,
            source_lang=detected_lang,
            target_lang=target_lang,
            separated_audio=separated_audio,
            voice_samples=voice_samples,
        )

    def revoice(
        self,
        video: Video,
        text: str,
        preserve_background: bool = True,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> RevoiceResult:
        """Replace speech in a video with new text using voice cloning.

        Extracts the original speaker's voice and generates new speech
        with the provided text while preserving background audio.

        Args:
            video: Video to revoice.
            text: New text for the speaker to say.
            preserve_background: Preserve background audio (music, effects).
            progress_callback: Optional progress callback (stage_name, progress).

        Returns:
            RevoiceResult with revoiced audio and metadata.
        """
        from videopython.base.audio import Audio

        def report_progress(stage: str, progress: float) -> None:
            if progress_callback:
                progress_callback(stage, progress)

        source_audio = video.audio
        original_duration = source_audio.metadata.duration_seconds

        # Step 1: Transcribe to find speech segments for voice extraction
        report_progress("Analyzing audio", 0.05)
        if self._transcriber is None:
            self._init_transcriber()

        transcription = self._transcriber.transcribe(source_audio)

        # Step 2: Separate audio (if preserving background)
        separated_audio: SeparatedAudio | None = None
        vocal_audio = source_audio

        if preserve_background:
            report_progress("Separating audio", 0.20)
            if self._separator is None:
                self._init_separator()

            separated_audio = self._separator.separate(source_audio)
            vocal_audio = separated_audio.vocals

        # Step 3: Extract voice sample
        report_progress("Extracting voice sample", 0.40)
        voice_sample: Audio | None = None

        if transcription.segments:
            voice_samples = self._extract_voice_samples(vocal_audio, transcription)
            # Use first speaker's voice sample
            if voice_samples:
                voice_sample = next(iter(voice_samples.values()))

        if voice_sample is None:
            # Fallback: use first few seconds of audio as voice sample
            sample_duration = min(6.0, original_duration)
            voice_sample = vocal_audio.slice(0, sample_duration)

        # Step 4: Generate new speech with cloned voice
        report_progress("Generating speech", 0.60)
        if self._tts is None:
            self._init_tts(voice_clone=True)

        generated_speech = self._tts.generate_audio(text, voice_sample=voice_sample)
        speech_duration = generated_speech.metadata.duration_seconds

        # Step 5: Mix with background if available
        report_progress("Assembling audio", 0.85)

        if separated_audio is not None:
            # Resample speech to match background sample rate if needed
            background_sr = separated_audio.background.metadata.sample_rate
            if generated_speech.metadata.sample_rate != background_sr:
                generated_speech = generated_speech.resample(background_sr)

            # Trim or extend background to match speech duration
            background = separated_audio.background
            if background.metadata.duration_seconds > speech_duration:
                background = background.slice(0, speech_duration)
            elif background.metadata.duration_seconds < speech_duration:
                # Pad background with silence
                silence_duration = speech_duration - background.metadata.duration_seconds
                silence = Audio.silence(
                    duration=silence_duration,
                    sample_rate=background_sr,
                    channels=background.metadata.channels,
                )
                background = background.concat(silence)

            # Overlay speech on background
            final_audio = background.overlay(generated_speech, position=0.0)
        else:
            final_audio = generated_speech

        report_progress("Complete", 1.0)

        return RevoiceResult(
            revoiced_audio=final_audio,
            text=text,
            separated_audio=separated_audio,
            voice_sample=voice_sample,
            original_duration=original_duration,
            speech_duration=speech_duration,
        )

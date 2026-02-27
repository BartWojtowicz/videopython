"""Local dubbing pipeline that combines transcription, translation, and TTS."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from videopython.ai.dubbing.models import DubbingResult, RevoiceResult, SeparatedAudio
from videopython.ai.dubbing.timing import TimingSynchronizer

if TYPE_CHECKING:
    from videopython.base.video import Video

logger = logging.getLogger(__name__)


class LocalDubbingPipeline:
    """Local pipeline for video dubbing."""

    def __init__(self, device: str | None = None):
        self.device = device
        requested = device.lower() if isinstance(device, str) else "auto"
        logger.info("LocalDubbingPipeline initialized with device=%s", requested)

        self._transcriber: Any = None
        self._translator: Any = None
        self._tts: Any = None
        self._separator: Any = None
        self._synchronizer: TimingSynchronizer | None = None

    def _init_transcriber(self) -> None:
        """Initialize the transcription model."""
        from videopython.ai.understanding.audio import AudioToText

        self._transcriber = AudioToText(device=self.device)

    def _init_translator(self) -> None:
        """Initialize the translation model."""
        from videopython.ai.generation.translation import TextTranslator

        self._translator = TextTranslator(device=self.device)

    def _init_tts(self, voice_clone: bool = False) -> None:
        """Initialize the text-to-speech model."""
        from videopython.ai.generation.audio import TextToSpeech

        if voice_clone:
            self._tts = TextToSpeech(
                model_size="xtts",
                device=self.device,
            )
        else:
            self._tts = TextToSpeech(device=self.device)

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
        """Extract voice samples for each speaker from the audio."""
        from videopython.base.audio import Audio

        voice_samples: dict[str, Audio] = {}

        segments_by_speaker: dict[str, list[Any]] = {}
        for segment in transcription.segments:
            speaker = segment.speaker or "speaker_0"
            if speaker not in segments_by_speaker:
                segments_by_speaker[speaker] = []
            segments_by_speaker[speaker].append(segment)

        for speaker, segments in segments_by_speaker.items():
            target_duration = 6.0
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
        """Process a video through the local dubbing pipeline."""
        from videopython.base.audio import Audio

        def report_progress(stage: str, progress: float) -> None:
            if progress_callback:
                progress_callback(stage, progress)

        report_progress("Transcribing audio", 0.05)
        if self._transcriber is None:
            self._init_transcriber()

        source_audio = video.audio
        transcription = self._transcriber.transcribe(source_audio)

        if not transcription.segments:
            return DubbingResult(
                dubbed_audio=source_audio,
                translated_segments=[],
                source_transcription=transcription,
                source_lang=source_lang or "unknown",
                target_lang=target_lang,
            )

        detected_lang = source_lang or "en"

        separated_audio: SeparatedAudio | None = None
        vocal_audio = source_audio

        if preserve_background:
            report_progress("Separating audio", 0.15)
            if self._separator is None:
                self._init_separator()

            separated_audio = self._separator.separate(source_audio)
            vocal_audio = separated_audio.vocals

        voice_samples: dict[str, Audio] = {}
        if voice_clone:
            report_progress("Extracting voice samples", 0.25)
            voice_samples = self._extract_voice_samples(vocal_audio, transcription)

        report_progress("Translating text", 0.35)
        if self._translator is None:
            self._init_translator()

        translated_segments = self._translator.translate_segments(
            segments=transcription.segments,
            target_lang=target_lang,
            source_lang=detected_lang,
        )

        report_progress("Generating dubbed speech", 0.50)
        if self._tts is None:
            self._init_tts(voice_clone=voice_clone)

        dubbed_segments: list[Audio] = []
        target_durations: list[float] = []
        start_times: list[float] = []

        for i, segment in enumerate(translated_segments):
            progress = 0.50 + (0.30 * (i / len(translated_segments)))
            report_progress(f"Generating speech ({i + 1}/{len(translated_segments)})", progress)

            speaker = segment.speaker or "speaker_0"
            voice_sample = voice_samples.get(speaker)

            if voice_clone and voice_sample is not None:
                dubbed_audio = self._tts.generate_audio(segment.translated_text, voice_sample=voice_sample)
            else:
                dubbed_audio = self._tts.generate_audio(segment.translated_text)

            dubbed_segments.append(dubbed_audio)
            target_durations.append(segment.duration)
            start_times.append(segment.start)

        report_progress("Synchronizing timing", 0.85)
        if self._synchronizer is None:
            self._init_synchronizer()
        assert self._synchronizer is not None

        synchronized_segments, _ = self._synchronizer.synchronize_segments(dubbed_segments, target_durations)

        report_progress("Assembling final audio", 0.90)
        total_duration = source_audio.metadata.duration_seconds
        dubbed_speech = self._synchronizer.assemble_with_timing(synchronized_segments, start_times, total_duration)

        if separated_audio is not None:
            background_sr = separated_audio.background.metadata.sample_rate
            if dubbed_speech.metadata.sample_rate != background_sr:
                dubbed_speech = dubbed_speech.resample(background_sr)

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
        """Replace speech in a video with new text using voice cloning."""
        from videopython.base.audio import Audio

        def report_progress(stage: str, progress: float) -> None:
            if progress_callback:
                progress_callback(stage, progress)

        source_audio = video.audio
        original_duration = source_audio.metadata.duration_seconds

        report_progress("Analyzing audio", 0.05)
        if self._transcriber is None:
            self._init_transcriber()

        transcription = self._transcriber.transcribe(source_audio)

        separated_audio: SeparatedAudio | None = None
        vocal_audio = source_audio

        if preserve_background:
            report_progress("Separating audio", 0.20)
            if self._separator is None:
                self._init_separator()

            separated_audio = self._separator.separate(source_audio)
            vocal_audio = separated_audio.vocals

        report_progress("Extracting voice sample", 0.40)
        voice_sample: Audio | None = None

        if transcription.segments:
            voice_samples = self._extract_voice_samples(vocal_audio, transcription)
            if voice_samples:
                voice_sample = next(iter(voice_samples.values()))

        if voice_sample is None:
            sample_duration = min(6.0, original_duration)
            voice_sample = vocal_audio.slice(0, sample_duration)

        report_progress("Generating speech", 0.60)
        if self._tts is None:
            self._init_tts(voice_clone=True)

        generated_speech = self._tts.generate_audio(text, voice_sample=voice_sample)
        speech_duration = generated_speech.metadata.duration_seconds

        report_progress("Assembling audio", 0.85)

        if separated_audio is not None:
            background_sr = separated_audio.background.metadata.sample_rate
            if generated_speech.metadata.sample_rate != background_sr:
                generated_speech = generated_speech.resample(background_sr)

            background = separated_audio.background
            if background.metadata.duration_seconds > speech_duration:
                background = background.slice(0, speech_duration)
            elif background.metadata.duration_seconds < speech_duration:
                silence_duration = speech_duration - background.metadata.duration_seconds
                silence = Audio.silence(
                    duration=silence_duration,
                    sample_rate=background_sr,
                    channels=background.metadata.channels,
                )
                background = background.concat(silence)

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

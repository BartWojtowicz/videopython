"""Local dubbing pipeline that combines transcription, translation, and TTS."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from videopython.ai._device import select_device
from videopython.ai.dubbing import expressiveness, loudness, voice_sample
from videopython.ai.dubbing.models import DubbingResult, Expressiveness, RevoiceResult, SeparatedAudio, TimingSummary
from videopython.ai.dubbing.quality import GarbageTranscriptError, assess_transcript
from videopython.ai.dubbing.timing import TimingSynchronizer
from videopython.ai.generation.qwen3 import Qwen3Translator
from videopython.ai.generation.translation import (
    MarianTranslator,
    TranslationBackend,
    UnsupportedLanguageError,
)

if TYPE_CHECKING:
    from videopython.ai.dubbing.models import TranslatedSegment
    from videopython.audio import Audio
    from videopython.base.transcription import Transcription


TranslatorChoice = Literal["auto", "marian", "qwen3"]
WhisperModel = Literal["tiny", "base", "small", "medium", "large", "turbo"]

logger = logging.getLogger(__name__)


class LocalDubbingPipeline:
    """Local pipeline for video dubbing.

    When ``low_memory=True``, each stage's model is unloaded after it runs, so
    only one model is resident at a time. This trades per-run latency (models
    re-load from disk between stages) for peak memory. Recommended for GPUs
    with <=12GB VRAM or hosts with <32GB RAM.
    """

    def __init__(
        self,
        device: str | None = None,
        low_memory: bool = False,
        whisper_model: WhisperModel = "turbo",
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = 0.6,
        logprob_threshold: float | None = -1.0,
        vocabulary: list[str] | None = None,
        strict_quality: bool = False,
        translator: TranslatorChoice = "auto",
    ):
        self.device = device
        self.low_memory = low_memory
        self.whisper_model = whisper_model
        self.condition_on_previous_text = condition_on_previous_text
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.vocabulary = vocabulary
        self.strict_quality = strict_quality
        self.translator = translator
        requested = device.lower() if isinstance(device, str) else "auto"
        logger.info(
            "LocalDubbingPipeline initialized with device=%s low_memory=%s whisper_model=%s translator=%s",
            requested,
            low_memory,
            whisper_model,
            translator,
        )

        self._transcriber: Any = None
        self._transcriber_diarization: bool | None = None
        self._translator: Any = None
        self._tts: Any = None
        self._tts_language: str | None = None
        self._separator: Any = None
        self._synchronizer: TimingSynchronizer | None = None

    def _maybe_unload(self, component_name: str) -> None:
        """Unload a stage's model when low_memory mode is enabled.

        No-op when low_memory=False or the component was never initialized
        (e.g. caller supplied a pre-computed transcription so the transcriber
        was skipped).
        """
        if not self.low_memory:
            return
        component = getattr(self, component_name, None)
        if component is None:
            return
        unload = getattr(component, "unload", None)
        if callable(unload):
            logger.info("low_memory: unloading %s", component_name.lstrip("_"))
            unload()

    def _transcribe(
        self,
        source_audio: Audio,
        enable_diarization: bool,
    ) -> Transcription:
        """Lazy-init the transcriber and run it on ``source_audio``."""
        if self._transcriber is None or self._transcriber_diarization != enable_diarization:
            self._init_transcriber(enable_diarization=enable_diarization)
            self._transcriber_diarization = enable_diarization

        transcription = self._transcriber.transcribe(source_audio)
        self._maybe_unload("_transcriber")
        return transcription

    def _tts_segment_audio(
        self,
        segment: TranslatedSegment,
        speaker: str,
        target_lang: str,
        voice_clone: bool,
        voice_samples: dict[str, Audio],
        speaker_wav_paths: dict[str, Path],
        expressiveness: Expressiveness = Expressiveness(),
    ) -> Audio | None:
        """Produce the TTS audio for a single segment.

        Returns the synthesized :class:`Audio`, or ``None`` if Chatterbox
        crashed on the segment (the caller skips it). The TTS model is
        lazy-initialized and per-speaker temp WAVs are materialized once
        across the loop.

        ``expressiveness`` carries the M4 Chatterbox knobs derived from
        the source segment's prosody. Default is the no-knobs profile —
        lets Chatterbox use its own defaults — so callers that don't yet
        derive prosody (e.g. ``revoice``) keep pre-M4 behaviour.
        """
        if self._tts is None or self._tts_language != target_lang:
            self._init_tts(language=target_lang)
            self._tts_language = target_lang
        if voice_clone and speaker not in speaker_wav_paths and speaker in voice_samples:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                voice_samples[speaker].save(f.name)
                speaker_wav_paths[speaker] = Path(f.name)

        wav_path = speaker_wav_paths.get(speaker) if voice_clone else None
        try:
            return self._tts.generate_audio(
                segment.translated_text,
                voice_sample_path=wav_path,
                **expressiveness.as_kwargs(),
            )
        except Exception as exc:
            # Chatterbox occasionally crashes on short translated text
            # (alignment_stream_analyzer indexing on tensors with <=5
            # speech tokens). One bad segment shouldn't lose a long
            # multi-hour run — log and let the caller skip.
            logger.warning(
                "TTS failed for segment (speaker=%s, text=%r): %s — skipping",
                speaker,
                segment.translated_text,
                exc,
            )
            return None

    def _translate(
        self,
        transcription: Transcription,
        source_lang: str,
        target_lang: str,
        report_progress: Callable[[str, float], None],
    ) -> tuple[list[TranslatedSegment], list[int]]:
        """Translate the transcription's segments into ``target_lang``.

        Returns ``(translated_segments, translation_failures)``. The
        progress callback maps the backend's [0, 1] fraction onto the
        pipeline's translation window (0.35 → 0.50).
        """
        if self._translator is None:
            self._init_translator(source_lang=source_lang, target_lang=target_lang)

        # Translation stage spans 0.35 → 0.50 of overall pipeline progress.
        # MarianMT runs sequentially over 8-segment batches; on a 15-min
        # source that's minutes of silent dwell on 0.35 without per-batch
        # ticks. Map the [0,1] translation fraction onto that 15% window.
        def _on_translation_progress(fraction: float) -> None:
            clamped = max(0.0, min(1.0, fraction))
            report_progress(f"Translating text ({int(clamped * 100)}%)", 0.35 + 0.15 * clamped)

        translated_segments = self._translator.translate_segments(
            segments=transcription.segments,
            target_lang=target_lang,
            source_lang=source_lang,
            progress_callback=_on_translation_progress,
        )
        # Capture per-segment failures (always empty for Marian) before
        # _maybe_unload nukes the backend in low_memory mode.
        translation_failures = list(self._translator.translation_failures)
        self._maybe_unload("_translator")

        return translated_segments, translation_failures

    def _init_transcriber(self, enable_diarization: bool = False) -> None:
        """Initialize the transcription model."""
        from videopython.ai.understanding.audio import AudioToText

        self._transcriber = AudioToText(
            model_name=self.whisper_model,
            device=self.device,
            enable_diarization=enable_diarization,
            condition_on_previous_text=self.condition_on_previous_text,
            no_speech_threshold=self.no_speech_threshold,
            logprob_threshold=self.logprob_threshold,
            vocabulary=self.vocabulary,
        )

    def _init_translator(self, source_lang: str, target_lang: str) -> None:
        """Initialize the translation backend.

        Resolves the configured ``self.translator`` choice into a concrete
        backend. ``"auto"`` uses :meth:`_resolve_translator_auto`; explicit
        choices instantiate the named backend directly. Re-initialization
        is a no-op when ``self._translator`` is already a matching instance
        for the same language pair (handled at call sites via the existing
        ``self._translator is None`` gate).
        """
        if self.translator == "marian":
            self._translator = MarianTranslator(device=self.device)
        elif self.translator == "qwen3":
            self._translator = Qwen3Translator(device=self.device)
        else:  # "auto"
            self._translator = self._resolve_translator_auto(source_lang, target_lang)

    def _resolve_translator_auto(self, source_lang: str, target_lang: str) -> TranslationBackend:
        """Pick a backend based on language coverage AND device.

        Qwen3-4B Q4_K_M on CPU is roughly 10-15x slower than MarianMT (M2.1
        spike on dreams_15min.mp4). The resolver picks Marian on CPU
        whenever it covers the language pair and only escalates to Qwen
        when a GPU is available or Marian doesn't cover the pair.
        """
        device = select_device(self.device, mps_allowed=True)
        has_gpu = device in ("cuda", "mps")

        # 1. GPU + Qwen covers the pair → Qwen wins (best quality).
        if has_gpu and Qwen3Translator.supports(source_lang, target_lang):
            logger.info(
                "translator: auto-selected qwen3 (device=%s, supports %s->%s)",
                device,
                source_lang,
                target_lang,
            )
            return Qwen3Translator(device=self.device)

        # 2. Marian covers the pair → Marian (fast).
        if MarianTranslator.has_model_for(source_lang, target_lang):
            if has_gpu:
                reason = f"Qwen does not cover {source_lang}->{target_lang}"
            else:
                reason = f"device={device} (Qwen would be ~10-15x slower; pass translator='qwen3' to override)"
            logger.info("translator: auto-selected marian (%s)", reason)
            return MarianTranslator(device=self.device)

        # 3. CPU + only Qwen covers it: warn loudly and use Qwen anyway.
        if Qwen3Translator.supports(source_lang, target_lang):
            logger.warning(
                "translator: auto-selected qwen3 on CPU (%s->%s not in Marian); "
                "translation will be slow (~10-15x MarianMT). Consider GPU.",
                source_lang,
                target_lang,
            )
            return Qwen3Translator(device=self.device)

        raise UnsupportedLanguageError(source_lang, target_lang)

    def _init_tts(self, language: str = "en") -> None:
        """Initialize the text-to-speech model."""
        from videopython.ai.generation.audio import TextToSpeech

        self._tts = TextToSpeech(device=self.device, language=language)

    def _init_separator(self) -> None:
        """Initialize the audio separator."""
        from videopython.ai.understanding.separation import AudioSeparator

        self._separator = AudioSeparator(device=self.device)

    def _init_synchronizer(self) -> None:
        """Initialize the timing synchronizer."""
        self._synchronizer = TimingSynchronizer()

    def _finalise_audio(
        self,
        dubbed_speech: Audio,
        source_audio: Audio,
        background_audio: Audio | None,
    ) -> Audio:
        """Overlay onto background (if any) and loudness-match against source.

        Shared finalisation tail between :meth:`process` and :meth:`revoice`.
        Resamples the dubbed/generated speech to the background's sample
        rate when they differ, overlays at position 0, then loudness-matches
        the result against the original ``source_audio`` so the dub doesn't
        land perceptually thinner than the input.
        """
        if background_audio is not None:
            background_sr = background_audio.metadata.sample_rate
            if dubbed_speech.metadata.sample_rate != background_sr:
                dubbed_speech = dubbed_speech.resample(background_sr)
            final_audio = background_audio.overlay(dubbed_speech, position=0.0)
        else:
            final_audio = dubbed_speech
        return loudness.loudness_match(final_audio, source_audio)

    def process(
        self,
        source_audio: Audio,
        target_lang: str,
        source_lang: str | None = None,
        preserve_background: bool = True,
        voice_clone: bool = True,
        enable_diarization: bool = False,
        progress_callback: Callable[[str, float], None] | None = None,
        transcription: Any | None = None,
    ) -> DubbingResult:
        """Run the dubbing pipeline against the given source audio.

        Args:
            source_audio: Source audio track to dub. Callers with a ``Video``
                object should pass ``video.audio``; callers with only a file path
                can use ``Audio.from_path(path)`` to avoid loading video frames.
            transcription: Optional pre-computed Transcription object. When provided,
                the internal Whisper transcription step is skipped (saving time and VRAM).
                Must be a ``videopython.base.transcription.Transcription`` instance
                with populated ``segments``. Speaker labels on the supplied transcription
                drive per-speaker voice cloning. If the supplied transcription has no
                speakers and ``enable_diarization=True``, pyannote is run standalone on
                ``source_audio`` and speakers are attached to the supplied words
                (requires word-level timings).
            enable_diarization: When True, run speaker diarization to enable per-speaker
                voice cloning. With ``transcription=None``, runs alongside Whisper. With
                a supplied ``transcription`` that has no speakers, runs pyannote
                standalone and overlays speakers onto the supplied words. Ignored when
                the supplied transcription already has speaker labels.
        """

        def report_progress(stage: str, progress: float) -> None:
            if progress_callback:
                progress_callback(stage, progress)

        if transcription is not None:
            report_progress("Using provided transcription", 0.05)
            if transcription.speakers:
                logger.info(
                    "Using provided transcription: %d segment(s), %d speaker(s)",
                    len(transcription.segments),
                    len(transcription.speakers),
                )
                if enable_diarization:
                    logger.info("enable_diarization=True ignored: supplied transcription already has speaker labels.")
            elif enable_diarization:
                report_progress("Diarizing supplied transcription", 0.10)
                if self._transcriber is None or self._transcriber_diarization is not True:
                    self._init_transcriber(enable_diarization=True)
                    self._transcriber_diarization = True
                transcription = self._transcriber.diarize_transcription(source_audio, transcription)
                self._maybe_unload("_transcriber")
                logger.info(
                    "Diarized supplied transcription: %d segment(s), %d speaker(s)",
                    len(transcription.segments),
                    len(transcription.speakers),
                )
            else:
                logger.info(
                    "Using provided transcription: %d segment(s), no speaker labels. "
                    "All segments will share a single voice clone. Pass "
                    "enable_diarization=True to add per-speaker labels, or "
                    "voice_clone=False to use the default TTS voice.",
                    len(transcription.segments),
                )
        else:
            report_progress("Transcribing audio", 0.05)
            transcription = self._transcribe(source_audio, enable_diarization)

        if not transcription.segments:
            return DubbingResult(
                dubbed_audio=source_audio,
                translated_segments=[],
                source_transcription=transcription,
                source_lang=source_lang or "unknown",
                target_lang=target_lang,
            )

        # Cheap heuristic gate before the expensive Demucs/translation/TTS
        # stages. Lets strict_quality callers refuse-and-refund without
        # running the rest of the pipeline; non-strict runs continue but
        # surface the assessment on DubbingResult.
        transcript_quality = assess_transcript(transcription, source_audio.metadata.duration_seconds)
        if transcript_quality.recommendation == "reject" and self.strict_quality:
            raise GarbageTranscriptError(
                f"Refusing to dub: {', '.join(transcript_quality.flags)}",
                transcript_quality,
            )
        if transcript_quality.recommendation in ("warn", "reject"):
            logger.warning(
                "Transcript quality flags raised: %s (recommendation=%s)",
                ", ".join(transcript_quality.flags),
                transcript_quality.recommendation,
            )

        detected_lang = source_lang or transcription.language or "en"

        separated_audio: SeparatedAudio | None = None
        vocal_audio = source_audio
        background_audio: Audio | None = None

        if preserve_background:
            report_progress("Separating audio", 0.15)
            if self._separator is None:
                self._init_separator()

            # Limit Demucs to the speech-bearing portion of the audio. The
            # transcription has already located every speech region; running
            # source separation outside those is pure overhead (no vocals to
            # isolate). On talk-heavy sources with silence/music gaps this
            # roughly halves separation time. When speech covers most of the
            # track separate_regions falls back to a full-track separate().
            from videopython.ai.understanding.separation import _merge_regions

            speech_regions = _merge_regions(
                [(s.start, s.end) for s in transcription.segments],
                audio_duration=source_audio.metadata.duration_seconds,
            )
            separated_audio = self._separator.separate_regions(source_audio, speech_regions)
            self._maybe_unload("_separator")
            vocal_audio = separated_audio.vocals
            background_audio = separated_audio.background
            # In low_memory mode, drop the SeparatedAudio container so vocals
            # and background can be released as soon as their last local
            # reference goes (after voice-sample extraction and final overlay
            # respectively). The result will report separated_audio=None.
            if self.low_memory:
                separated_audio = None

        voice_samples: dict[str, Audio] = {}
        if voice_clone:
            report_progress("Extracting voice samples", 0.25)
            voice_samples = voice_sample.extract(vocal_audio, background_audio, transcription)

        report_progress("Translating text", 0.35)
        translated_segments, translation_failures = self._translate(
            transcription, detected_lang, target_lang, report_progress
        )

        # Per-segment expressiveness derived from source vocals RMS.
        # Computed before vocal_audio is released so the TTS loop doesn't
        # hold the buffer. Segment ends are clamped to the vocals duration
        # — transcription timestamps can drift past the buffer tail
        # (especially on synthetic test audio) and Audio.slice rejects
        # out-of-range ends past a 0.1s tolerance.
        baseline_rms = expressiveness.rms(vocal_audio.data)
        vocal_duration = vocal_audio.metadata.duration_seconds
        expressiveness_per_segment = [
            expressiveness.expressiveness_for(
                vocal_audio.slice(min(s.start, vocal_duration), min(s.end, vocal_duration)),
                baseline_rms,
            )
            for s in translated_segments
        ]

        # vocals is no longer needed; voice_samples are independent copies.
        # In low_memory mode this is the only ref keeping the buffer alive
        # (separated_audio was dropped above), so dropping the local frees it.
        del vocal_audio

        report_progress("Generating dubbed speech", 0.50)

        dubbed_segments: list[Audio] = []
        target_durations: list[float] = []
        start_times: list[float] = []

        # Per-speaker temp WAVs are materialized lazily by _tts_segment_audio.
        # The dict is loop-scoped state so the finally block can clean up.
        speaker_wav_paths: dict[str, Path] = {}
        try:
            for i, segment in enumerate(translated_segments):
                if segment.duration < 0.1:
                    continue
                # Translation filter (translation.py:_is_translatable_text)
                # leaves translated_text="" for punctuation-only or empty
                # segments. Don't TTS those — saves a model call and avoids
                # injecting hallucinated speech into the dubbed track.
                if not segment.translated_text.strip():
                    continue

                progress = 0.50 + (0.30 * (i / len(translated_segments)))
                report_progress(f"Generating speech ({i + 1}/{len(translated_segments)})", progress)

                speaker = segment.speaker or "speaker_0"
                dubbed_audio = self._tts_segment_audio(
                    segment=segment,
                    speaker=speaker,
                    target_lang=target_lang,
                    voice_clone=voice_clone,
                    voice_samples=voice_samples,
                    speaker_wav_paths=speaker_wav_paths,
                    expressiveness=expressiveness_per_segment[i],
                )
                if dubbed_audio is None:
                    continue

                dubbed_segments.append(dubbed_audio)
                target_durations.append(segment.duration)
                start_times.append(segment.start)
        finally:
            for path in speaker_wav_paths.values():
                path.unlink(missing_ok=True)

        self._maybe_unload("_tts")

        report_progress("Synchronizing timing", 0.85)
        if self._synchronizer is None:
            self._init_synchronizer()
        assert self._synchronizer is not None

        synchronized_segments, adjustments = self._synchronizer.synchronize_segments(dubbed_segments, target_durations)
        timing_summary = TimingSummary.from_adjustments(adjustments)
        del dubbed_segments

        report_progress("Assembling final audio", 0.90)
        total_duration = source_audio.metadata.duration_seconds
        dubbed_speech = self._synchronizer.assemble_with_timing(synchronized_segments, start_times, total_duration)
        del synchronized_segments

        final_audio = self._finalise_audio(dubbed_speech, source_audio, background_audio)
        # Drop the local; in low_memory this releases the background
        # buffer (~1.3 GB for 2h sources). In non-low_memory the same
        # array is still held by separated_audio.background.
        del background_audio

        report_progress("Complete", 1.0)

        return DubbingResult(
            dubbed_audio=final_audio,
            translated_segments=translated_segments,
            source_transcription=transcription,
            source_lang=detected_lang,
            target_lang=target_lang,
            separated_audio=separated_audio,
            voice_samples=voice_samples,
            timing_summary=timing_summary,
            transcript_quality=transcript_quality,
            translation_failures=translation_failures,
        )

    def revoice(
        self,
        source_audio: Audio,
        text: str,
        preserve_background: bool = True,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> RevoiceResult:
        """Replace speech in audio with new text using voice cloning.

        Args:
            source_audio: Source audio track to revoice. Callers with a ``Video``
                object should pass ``video.audio``.
        """
        from videopython.audio import Audio

        def report_progress(stage: str, progress: float) -> None:
            if progress_callback:
                progress_callback(stage, progress)

        original_duration = source_audio.metadata.duration_seconds

        report_progress("Analyzing audio", 0.05)
        if self._transcriber is None or self._transcriber_diarization is not False:
            self._init_transcriber(enable_diarization=False)
            self._transcriber_diarization = False

        transcription = self._transcriber.transcribe(source_audio)
        self._maybe_unload("_transcriber")

        separated_audio: SeparatedAudio | None = None
        vocal_audio = source_audio
        background_audio: Audio | None = None

        if preserve_background:
            report_progress("Separating audio", 0.20)
            if self._separator is None:
                self._init_separator()

            from videopython.ai.understanding.separation import _merge_regions

            speech_regions = _merge_regions(
                [(s.start, s.end) for s in transcription.segments],
                audio_duration=source_audio.metadata.duration_seconds,
            )
            separated_audio = self._separator.separate_regions(source_audio, speech_regions)
            self._maybe_unload("_separator")
            vocal_audio = separated_audio.vocals
            background_audio = separated_audio.background
            if self.low_memory:
                separated_audio = None

        report_progress("Extracting voice sample", 0.40)
        chosen_sample: Audio | None = None

        if transcription.segments:
            # revoice doesn't track the background after the low_memory drop,
            # so quality gating degrades to "no RMS check" here. Clipping is
            # still rejected.
            voice_samples = voice_sample.extract(vocal_audio, None, transcription)
            if voice_samples:
                chosen_sample = next(iter(voice_samples.values()))

        if chosen_sample is None:
            sample_duration = min(6.0, original_duration)
            sliced = vocal_audio.slice(0, sample_duration)
            # Copy so the short sample doesn't pin the full vocals array.
            chosen_sample = Audio(sliced.data.copy(), sliced.metadata)

        del vocal_audio

        report_progress("Generating speech", 0.60)
        if self._tts is None or self._tts_language != "en":
            self._init_tts(language="en")
            self._tts_language = "en"

        generated_speech = self._tts.generate_audio(text, voice_sample=chosen_sample)
        speech_duration = generated_speech.metadata.duration_seconds
        self._maybe_unload("_tts")

        report_progress("Assembling audio", 0.85)

        if background_audio is not None:
            background_sr = background_audio.metadata.sample_rate
            if background_audio.metadata.duration_seconds > speech_duration:
                background_audio = background_audio.slice(0, speech_duration)
            elif background_audio.metadata.duration_seconds < speech_duration:
                silence_duration = speech_duration - background_audio.metadata.duration_seconds
                silence = Audio.silence(
                    duration=silence_duration,
                    sample_rate=background_sr,
                    channels=background_audio.metadata.channels,
                )
                background_audio = background_audio.concat(silence)

        final_audio = self._finalise_audio(generated_speech, source_audio, background_audio)
        del background_audio

        report_progress("Complete", 1.0)

        return RevoiceResult(
            revoiced_audio=final_audio,
            text=text,
            separated_audio=separated_audio,
            voice_sample=chosen_sample,
            original_duration=original_duration,
            speech_duration=speech_duration,
        )

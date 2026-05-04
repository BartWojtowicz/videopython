"""Local dubbing pipeline that combines transcription, translation, and TTS."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np

from videopython.ai.dubbing.models import DubbingResult, RevoiceResult, SeparatedAudio, TimingSummary
from videopython.ai.dubbing.timing import TimingSynchronizer

if TYPE_CHECKING:
    from videopython.base.audio import Audio


def _peak_match(target: Audio, reference: Audio) -> Audio:
    """Scale ``target`` so its peak amplitude matches ``reference``.

    Demucs background normalization and the timing-assembler peak guard
    each clamp at 1.0 instead of restoring headroom, so a dubbed mix
    typically lands quieter than the source — perceptually "thinner."
    A single peak match recovers most of that drift without LUFS deps.

    No-op when either side has zero peak (silent input or all-silent dub).
    The new ``Audio`` shares no buffer with ``target``.
    """
    from videopython.base.audio import Audio as _Audio

    target_peak = float(np.max(np.abs(target.data))) if target.data.size else 0.0
    reference_peak = float(np.max(np.abs(reference.data))) if reference.data.size else 0.0

    if target_peak <= 0.0 or reference_peak <= 0.0:
        return target

    scale = reference_peak / target_peak
    if abs(scale - 1.0) < 1e-3:
        return target

    return _Audio(target.data * scale, target.metadata)


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
        strict_quality: bool = False,
    ):
        self.device = device
        self.low_memory = low_memory
        self.whisper_model = whisper_model
        self.condition_on_previous_text = condition_on_previous_text
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.strict_quality = strict_quality
        requested = device.lower() if isinstance(device, str) else "auto"
        logger.info(
            "LocalDubbingPipeline initialized with device=%s low_memory=%s whisper_model=%s",
            requested,
            low_memory,
            whisper_model,
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
        )

    def _init_translator(self) -> None:
        """Initialize the translation model."""
        from videopython.ai.generation.translation import TextTranslator

        self._translator = TextTranslator(device=self.device)

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

    # Voice-sample quality gating thresholds. Tuned conservatively to favor
    # accepting real-world dialogue over rejecting it; failures fall back to
    # the longest segment with a WARNING log so we can re-tune from production
    # data instead of guessing.
    _PEAK_CLIP_THRESHOLD = 0.99
    _MIN_VOCAL_BG_RMS_RATIO = 1.5
    _VOICE_SAMPLE_TARGET_DURATION = 6.0

    def _extract_voice_samples(
        self,
        vocal_audio: Any,
        background_audio: Any | None,
        transcription: Any,
        min_duration: float = 3.0,
        max_duration: float = 10.0,
    ) -> dict[str, Any]:
        """Extract a per-speaker voice sample with quality gating.

        Picks the highest-scored segment per speaker after rejecting clipped
        slices (peak >= ``_PEAK_CLIP_THRESHOLD``) and slices where Demucs left
        the background louder than the vocals
        (``vocal_rms / bg_rms < _MIN_VOCAL_BG_RMS_RATIO``). When the
        background track isn't available (e.g. ``revoice`` after
        ``low_memory`` dropped it), the RMS check is skipped silently.

        Falls back to the longest available segment with a WARNING log when
        every candidate is rejected, so the dub continues with the best
        sample we have rather than silently dropping the speaker.
        """
        from videopython.base.audio import Audio

        voice_samples: dict[str, Audio] = {}

        segments_by_speaker: dict[str, list[Any]] = {}
        for segment in transcription.segments:
            speaker = segment.speaker or "speaker_0"
            if speaker not in segments_by_speaker:
                segments_by_speaker[speaker] = []
            segments_by_speaker[speaker].append(segment)

        for speaker, segments in segments_by_speaker.items():
            chosen, fallback_reason = self._pick_voice_segment(
                speaker, segments, vocal_audio, background_audio, min_duration
            )

            if chosen is None:
                logger.warning("No usable voice-sample segment for speaker %r (no candidates)", speaker)
                continue

            if fallback_reason is not None:
                logger.warning(
                    "Voice-sample quality fallback for speaker %r (%d candidates): %s — using longest segment",
                    speaker,
                    len(segments),
                    fallback_reason,
                )

            start = chosen.start
            end = min(chosen.end, start + max_duration)
            sliced = vocal_audio.slice(start, end)
            # Audio.slice returns a numpy view into the source. Copy so the
            # short voice sample doesn't keep the full vocals array (~1.3 GB
            # for 2h sources) alive across translate + TTS.
            voice_samples[speaker] = Audio(sliced.data.copy(), sliced.metadata)

        return voice_samples

    def _pick_voice_segment(
        self,
        speaker: str,
        segments: list[Any],
        vocal_audio: Any,
        background_audio: Any | None,
        min_duration: float,
    ) -> tuple[Any | None, str | None]:
        """Score eligible segments and pick the best one for ``speaker``.

        Returns ``(segment, fallback_reason)``. ``fallback_reason`` is None
        when scoring picked a segment cleanly; non-None when every candidate
        was rejected and the longest segment was used instead.
        """
        if not segments:
            return None, None

        eligible = [s for s in segments if (s.end - s.start) >= min_duration]

        rejection_reasons: list[str] = []
        scored: list[tuple[float, Any]] = []
        for segment in eligible:
            score, reason = self._score_voice_segment(segment, vocal_audio, background_audio)
            if score is None:
                rejection_reasons.append(reason or "rejected")
            else:
                scored.append((score, segment))

        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[0][1], None

        # All eligible segments rejected (or none met the min duration).
        # Fall back to the longest segment overall so the speaker still
        # gets a clone reference.
        longest = max(segments, key=lambda s: s.end - s.start)
        if eligible:
            reason = ", ".join(sorted(set(rejection_reasons)))
        else:
            reason = f"no segment >= {min_duration:.1f}s"
        return longest, reason

    def _score_voice_segment(
        self,
        segment: Any,
        vocal_audio: Any,
        background_audio: Any | None,
    ) -> tuple[float | None, str | None]:
        """Return ``(score, reason)`` for a candidate segment.

        ``score`` is ``None`` when the segment is rejected; ``reason`` carries
        the rejection cause so the fallback logger can summarize.
        """
        vocal_slice = vocal_audio.slice(segment.start, segment.end)
        if vocal_slice.data.size == 0:
            return None, "empty slice"

        peak = float(np.max(np.abs(vocal_slice.data)))
        if peak >= self._PEAK_CLIP_THRESHOLD:
            return None, "clipped"

        vocal_rms = float(np.sqrt(np.mean(vocal_slice.data**2)))

        if background_audio is not None:
            bg_slice = background_audio.slice(segment.start, segment.end)
            if bg_slice.data.size > 0:
                bg_rms = float(np.sqrt(np.mean(bg_slice.data**2)))
                if bg_rms > 0 and (vocal_rms / bg_rms) < self._MIN_VOCAL_BG_RMS_RATIO:
                    return None, "background-dominated"

        duration = segment.end - segment.start
        duration_penalty = abs(duration - self._VOICE_SAMPLE_TARGET_DURATION)
        return vocal_rms - 0.05 * duration_penalty, None

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
                Must be a ``videopython.base.text.transcription.Transcription`` instance
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
            if self._transcriber is None or self._transcriber_diarization != enable_diarization:
                self._init_transcriber(enable_diarization=enable_diarization)
                self._transcriber_diarization = enable_diarization

            transcription = self._transcriber.transcribe(source_audio)
            self._maybe_unload("_transcriber")

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
        from videopython.ai.dubbing.quality import GarbageTranscriptError, assess_transcript

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
            voice_samples = self._extract_voice_samples(vocal_audio, background_audio, transcription)

        # vocals is no longer needed; voice_samples are independent copies.
        # In low_memory mode this is the only ref keeping the buffer alive
        # (separated_audio was dropped above), so dropping the local frees it.
        del vocal_audio

        report_progress("Translating text", 0.35)
        if self._translator is None:
            self._init_translator()

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
            source_lang=detected_lang,
            progress_callback=_on_translation_progress,
        )
        self._maybe_unload("_translator")

        report_progress("Generating dubbed speech", 0.50)
        if self._tts is None or self._tts_language != target_lang:
            self._init_tts(language=target_lang)
            self._tts_language = target_lang

        dubbed_segments: list[Audio] = []
        target_durations: list[float] = []
        start_times: list[float] = []

        # Encode each speaker's voice sample to a temp WAV exactly once and
        # reuse the path across every segment for that speaker. Without this
        # cache, TextToSpeech.generate_audio re-encodes the same voice sample
        # on every call (one temp WAV write + delete per segment), which is
        # pure overhead for long dubs with many segments per speaker.
        speaker_wav_paths: dict[str, Path] = {}
        try:
            if voice_clone:
                for speaker, sample in voice_samples.items():
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        sample.save(f.name)
                        speaker_wav_paths[speaker] = Path(f.name)

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
                cached_path = speaker_wav_paths.get(speaker) if voice_clone else None

                try:
                    if cached_path is not None:
                        dubbed_audio = self._tts.generate_audio(segment.translated_text, voice_sample_path=cached_path)
                    else:
                        dubbed_audio = self._tts.generate_audio(segment.translated_text)
                except Exception as e:
                    # Chatterbox occasionally crashes on short translated text
                    # (alignment_stream_analyzer indexing on tensors with <=5
                    # speech tokens). One bad segment shouldn't lose a long
                    # multi-hour run — log and skip so the rest proceeds.
                    logger.warning(
                        "TTS failed for segment %d/%d (speaker=%s, text=%r): %s — skipping",
                        i + 1,
                        len(translated_segments),
                        speaker,
                        segment.translated_text,
                        e,
                    )
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

        if background_audio is not None:
            background_sr = background_audio.metadata.sample_rate
            if dubbed_speech.metadata.sample_rate != background_sr:
                dubbed_speech = dubbed_speech.resample(background_sr)

            final_audio = background_audio.overlay(dubbed_speech, position=0.0)
            # Drop the local; in low_memory this releases the background
            # buffer (~1.3 GB for 2h sources). In non-low_memory the same
            # array is still held by separated_audio.background.
            del background_audio
        else:
            final_audio = dubbed_speech

        # Peak-match against the source so the dub doesn't land quieter
        # than the original. Done last so it captures both vocals+background
        # mixes and speech-only outputs uniformly.
        final_audio = _peak_match(final_audio, source_audio)

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
        from videopython.base.audio import Audio

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
        voice_sample: Audio | None = None

        if transcription.segments:
            # revoice doesn't track the background after the low_memory drop,
            # so quality gating degrades to "no RMS check" here. Clipping is
            # still rejected.
            voice_samples = self._extract_voice_samples(vocal_audio, None, transcription)
            if voice_samples:
                voice_sample = next(iter(voice_samples.values()))

        if voice_sample is None:
            sample_duration = min(6.0, original_duration)
            sliced = vocal_audio.slice(0, sample_duration)
            # Copy so the short sample doesn't pin the full vocals array.
            voice_sample = Audio(sliced.data.copy(), sliced.metadata)

        del vocal_audio

        report_progress("Generating speech", 0.60)
        if self._tts is None or self._tts_language != "en":
            self._init_tts(language="en")
            self._tts_language = "en"

        generated_speech = self._tts.generate_audio(text, voice_sample=voice_sample)
        speech_duration = generated_speech.metadata.duration_seconds
        self._maybe_unload("_tts")

        report_progress("Assembling audio", 0.85)

        if background_audio is not None:
            background_sr = background_audio.metadata.sample_rate
            if generated_speech.metadata.sample_rate != background_sr:
                generated_speech = generated_speech.resample(background_sr)

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

            final_audio = background_audio.overlay(generated_speech, position=0.0)
            del background_audio
        else:
            final_audio = generated_speech

        final_audio = _peak_match(final_audio, source_audio)

        report_progress("Complete", 1.0)

        return RevoiceResult(
            revoiced_audio=final_audio,
            text=text,
            separated_audio=separated_audio,
            voice_sample=voice_sample,
            original_duration=original_duration,
            speech_duration=speech_duration,
        )

"""Audio understanding using local models."""

from __future__ import annotations

import logging
from bisect import bisect_left
from typing import Any, Literal

from videopython.ai._device import log_device_initialization, select_device
from videopython.ai._predictor import ManagedPredictor
from videopython.ai._revisions import pinned
from videopython.audio import Audio
from videopython.base.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video

logger = logging.getLogger(__name__)

# Whisper's initial_prompt budget; longer prompts are silently truncated by the decoder.
_INITIAL_PROMPT_TOKEN_BUDGET = 224
_INITIAL_PROMPT_TEMPLATE = "Transcript may include the following names: {terms}."
_WHISPER_SAMPLE_RATE = 16_000
_WHISPER_LANGUAGE_SAMPLES = 30 * _WHISPER_SAMPLE_RATE
_WHISPER_MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large": "Systran/faster-whisper-large-v3",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}


def _normalize_vocabulary(vocabulary: list[str] | None) -> list[str]:
    """Strip, drop empties, and order-preserving dedup (case-insensitive key).

    Original casing is kept — Whisper biases toward what it sees in the
    prompt, so ``"InPost"`` is a stronger anchor than ``"inpost"`` for
    a stylized brand name. Rejects a non-list ``vocabulary`` early so a
    bare string isn't silently iterated as one-char terms.
    """
    if vocabulary is None:
        return []
    if not isinstance(vocabulary, list):
        raise TypeError(f"vocabulary must be a list[str] or None, got {type(vocabulary).__name__}")

    seen: set[str] = set()
    result: list[str] = []
    for term in vocabulary:
        stripped = term.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(stripped)
    return result


def _render_initial_prompt(terms: list[str]) -> str:
    return _INITIAL_PROMPT_TEMPLATE.format(terms=", ".join(terms))


def _build_initial_prompt(vocabulary: list[str], tokenizer: Any) -> str | None:
    """Render the prompt and trim tail terms until it fits Whisper's
    224-token ``initial_prompt`` budget; ``None`` for empty input."""
    if not vocabulary:
        return None

    kept = list(vocabulary)
    while (
        kept
        and len(tokenizer.encode(_render_initial_prompt(kept), add_special_tokens=False).ids)
        > _INITIAL_PROMPT_TOKEN_BUDGET
    ):
        kept.pop()

    if not kept:
        return None
    if len(kept) < len(vocabulary):
        logger.warning(
            "vocabulary truncated to fit Whisper's %d-token initial_prompt budget: dropped %d trailing term(s)",
            _INITIAL_PROMPT_TOKEN_BUDGET,
            len(vocabulary) - len(kept),
        )
    return _render_initial_prompt(kept)


def _attach_confidence_by_overlap(
    target_segments: list[TranscriptionSegment],
    source_segments: list[TranscriptionSegment],
) -> None:
    """Copy confidence from the greatest-overlap source into chronological targets.

    Both lists must be chronological and non-overlapping within themselves. Ties use
    the earlier source. Targets without an overlap stay unchanged.
    """
    source_index = 0
    for tgt in target_segments:
        while source_index < len(source_segments) and source_segments[source_index].end <= tgt.start:
            source_index += 1

        best_overlap = 0.0
        best_src: TranscriptionSegment | None = None
        candidate_index = source_index
        while candidate_index < len(source_segments) and source_segments[candidate_index].start < tgt.end:
            src = source_segments[candidate_index]
            overlap = min(tgt.end, src.end) - max(tgt.start, src.start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_src = src
            candidate_index += 1

        if best_src is not None:
            tgt.avg_logprob = best_src.avg_logprob
            tgt.no_speech_prob = best_src.no_speech_prob
            tgt.compression_ratio = best_src.compression_ratio


class AudioToText(ManagedPredictor):
    """Transcription service for audio and video using local Whisper models.

    Uses faster-whisper in float32 for transcription (with word-level timestamps) and
    pyannote-audio for optional speaker diarization. By default, Silero VAD
    runs before Whisper to gate language detection on a 30s window built from
    voiced regions only — fixes Whisper's tendency to lock onto the wrong
    language when the file opens with silence, music, or non-vocal credits.
    Set ``enable_vad=False`` to detect language from the leading audio without
    voice-activity gating.

    Three Whisper decoder kwargs are surfaced for anti-hallucination tuning:

    - ``condition_on_previous_text`` defaults to ``False`` (Whisper's own
      default is ``True``). With conditioning on, a single hallucinated filler
      phrase cascades through the rest of the file because each window's
      decoder is primed by the previous window's decoded text. Turning it off
      is the most commonly recommended fix for that failure mode; the cost on
      clean audio is small (slightly less context for ambiguous homophones
      across sentence boundaries).
    - ``no_speech_threshold`` and ``logprob_threshold`` are forwarded with
      Whisper's documented defaults (``0.6`` and ``-1.0``); raising
      ``no_speech_threshold`` biases toward dropping low-confidence windows
      instead of emitting filler.

    ``vocabulary`` biases Whisper's first-window decoder toward a caller-
    supplied list of brand names, product names, or proper nouns via the
    native ``initial_prompt`` channel. Recovers near-mishears (e.g. Klarna
    → "carna") without new model deps; will not catch zero-prior names.
    Per-call override is available on :meth:`transcribe`.
    """

    PYANNOTE_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
    _model_attrs = ("_model", "_diarization_pipeline", "_vad_model")

    def __init__(
        self,
        model_name: Literal["tiny", "base", "small", "medium", "large", "turbo"] = "turbo",
        enable_diarization: bool = False,
        enable_vad: bool = True,
        condition_on_previous_text: bool = False,
        no_speech_threshold: float = 0.6,
        logprob_threshold: float | None = -1.0,
        vocabulary: list[str] | None = None,
        device: str | None = None,
    ):
        if model_name not in _WHISPER_MODELS:
            choices = ", ".join(_WHISPER_MODELS)
            raise ValueError(f"Unsupported Whisper model {model_name!r}. Choose one of: {choices}.")
        self.model_name = model_name
        self.enable_diarization = enable_diarization
        self.enable_vad = enable_vad
        self.condition_on_previous_text = condition_on_previous_text
        self.no_speech_threshold = no_speech_threshold
        self.logprob_threshold = logprob_threshold
        self.vocabulary = _normalize_vocabulary(vocabulary)
        self.device = select_device(device, mps_allowed=False)
        log_device_initialization(
            "AudioToText",
            requested_device=device,
            resolved_device=self.device,
        )
        self._model: Any = None
        self._diarization_pipeline: Any = None
        self._vad_model: Any = None

    def _transcribe_kwargs(self, language: str | None, vocabulary: list[str]) -> dict[str, Any]:
        """Kwargs threaded into faster-whisper from both call sites.

        ``initial_prompt`` is omitted entirely on the no-vocab path."""
        kwargs: dict[str, Any] = {
            "word_timestamps": True,
            "language": language,
            "beam_size": 1,
            "best_of": 5,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "compression_ratio_threshold": 2.4,
            "condition_on_previous_text": self.condition_on_previous_text,
            "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.logprob_threshold,
            "vad_filter": False,
        }
        prompt = _build_initial_prompt(vocabulary, self._model.hf_tokenizer)
        if prompt is not None:
            kwargs["initial_prompt"] = prompt
        return kwargs

    def _init_local(self) -> None:
        """Initialize local Whisper model."""
        from videopython.ai._optional import require

        faster_whisper = require("faster_whisper", feature="AudioToText")
        model_id = _WHISPER_MODELS[self.model_name]
        self._model = faster_whisper.WhisperModel(
            model_id,
            revision=pinned(model_id),
            device=self.device,
            compute_type="float32",
        )

    def _run_whisper(self, audio: Any, language: str | None, vocabulary: list[str]) -> dict[str, Any]:
        segments_source, info = self._model.transcribe(
            audio=audio,
            **self._transcribe_kwargs(language, vocabulary),
        )
        segments = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": [{"start": word.start, "end": word.end, "word": word.word} for word in segment.words or []],
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "compression_ratio": segment.compression_ratio,
            }
            for segment in segments_source
        ]
        return {"segments": segments, "language": info.language}

    def _init_diarization(self) -> None:
        """Initialize pyannote speaker diarization pipeline."""
        import torch

        from videopython.ai._optional import require
        from videopython.ai.understanding import _pyannote_patches

        Pipeline = require("pyannote.audio", feature="AudioToText diarization").Pipeline

        self._diarization_pipeline = Pipeline.from_pretrained(
            self.PYANNOTE_DIARIZATION_MODEL, revision=pinned(self.PYANNOTE_DIARIZATION_MODEL)
        )
        _pyannote_patches.install(self._diarization_pipeline)
        self._diarization_pipeline.to(torch.device(self.device))

    def _init_vad(self) -> None:
        """Initialize Silero VAD model.

        The model is ~2 MB and CPU-fast (~5-15s for a 90 min movie); we keep
        it on CPU regardless of ``self.device`` since dispatch overhead would
        outweigh inference cost.
        """
        from videopython.ai._optional import require

        load_silero_vad = require("silero_vad", feature="AudioToText VAD").load_silero_vad

        self._vad_model = load_silero_vad()

    def _process_transcription_result(self, transcription_result: dict[str, Any]) -> Transcription:
        """Process raw transcription result into a Transcription object."""
        transcription_segments = []
        for segment in transcription_result["segments"]:
            transcription_words = [
                TranscriptionWord(word=word["word"], start=float(word["start"]), end=float(word["end"]))
                for word in segment.get("words", [])
            ]
            transcription_segment = TranscriptionSegment(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
                words=transcription_words,
                avg_logprob=segment.get("avg_logprob"),
                no_speech_prob=segment.get("no_speech_prob"),
                compression_ratio=segment.get("compression_ratio"),
            )
            transcription_segments.append(transcription_segment)

        return Transcription(segments=transcription_segments, language=transcription_result.get("language"))

    @staticmethod
    def _assign_speakers_to_words(
        words: list[TranscriptionWord],
        diarization_result: Any,
    ) -> list[TranscriptionWord]:
        """Assign speakers to chronological words from exclusive diarization tracks."""
        speaker_segments: list[tuple[float, float, str]] = []
        annotation = diarization_result.exclusive_speaker_diarization
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append((turn.start, turn.end, speaker))

        if not speaker_segments:
            return words

        speaker_segments.sort(key=lambda segment: (segment[0], segment[1]))
        speaker_midpoints = [(start + end) / 2.0 for start, end, _ in speaker_segments]
        result = []
        segment_index = 0
        for word in words:
            while segment_index < len(speaker_segments) and speaker_segments[segment_index][1] <= word.start:
                segment_index += 1

            best_speaker: str | None = None
            best_overlap = 0.0
            candidate_index = segment_index
            while candidate_index < len(speaker_segments) and speaker_segments[candidate_index][0] < word.end:
                seg_start, seg_end, speaker = speaker_segments[candidate_index]
                overlap = min(word.end, seg_end) - max(word.start, seg_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker
                candidate_index += 1

            if best_speaker is None:
                word_mid = (word.start + word.end) / 2.0
                nearest_index = bisect_left(speaker_midpoints, word_mid)
                if nearest_index == len(speaker_segments):
                    nearest_index -= 1
                elif nearest_index > 0:
                    previous_distance = word_mid - speaker_midpoints[nearest_index - 1]
                    next_distance = speaker_midpoints[nearest_index] - word_mid
                    if previous_distance <= next_distance:
                        nearest_index -= 1
                best_speaker = speaker_segments[nearest_index][2]

            result.append(
                TranscriptionWord(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    speaker=best_speaker,
                )
            )
        return result

    def diarize_transcription(self, audio: Audio, transcription: Transcription) -> Transcription:
        """Attach speaker labels to a pre-computed transcription using pyannote.

        Useful when callers have a transcription (e.g. pre-computed and edited)
        but no speakers, and want per-speaker voice cloning in dubbing without
        re-running Whisper. Runs pyannote standalone on ``audio`` and overlays
        speakers onto the supplied transcription's words.

        Requires word-level timings: at least one segment must contain more
        than one word. Transcriptions loaded from SRT (one synthetic word per
        segment) will not produce useful speakers and are rejected.
        """
        import numpy as np
        import torch

        all_words = sorted(transcription.words, key=lambda word: (word.start, word.end))
        if not all_words:
            raise ValueError("Cannot diarize a transcription with no words.")

        if not any(len(seg.words) > 1 for seg in transcription.segments):
            raise ValueError(
                "Cannot diarize a transcription without word-level timings. "
                "Supplied transcription has at most one word per segment "
                "(e.g. loaded from SRT). Provide a transcription with "
                "word-level timings, or omit `transcription` to let the "
                "pipeline transcribe and diarize from scratch."
            )

        if self._diarization_pipeline is None:
            self._init_diarization()

        audio_mono = audio.to_mono().resample(_WHISPER_SAMPLE_RATE)
        waveform = torch.from_numpy(audio_mono.data.astype(np.float32)).unsqueeze(0)
        diarization_result = self._diarization_pipeline(
            {"waveform": waveform, "sample_rate": audio_mono.metadata.sample_rate}
        )

        all_words = self._assign_speakers_to_words(all_words, diarization_result)

        # Rebuilding from words regroups by speaker and drops the per-segment
        # confidence the supplied transcription carried, exactly as it does on the
        # combined path -- so re-attach it the same way. Without this, splitting
        # transcription and diarization into two calls silently loses confidence
        # that running them as one keeps.
        source_segments = sorted(transcription.segments, key=lambda segment: (segment.start, segment.end))
        rebuilt = Transcription(words=all_words, language=transcription.language)
        _attach_confidence_by_overlap(rebuilt.segments, source_segments)
        return rebuilt

    def _run_vad(self, audio_mono: Audio) -> list[tuple[float, float]]:
        """Return voiced spans in seconds using Silero VAD.

        Audio must already be mono at 16 kHz,
        which is one of Silero's two supported rates.
        """
        import numpy as np
        import torch

        if self._vad_model is None:
            self._init_vad()

        from silero_vad import get_speech_timestamps

        waveform = torch.from_numpy(audio_mono.data.astype(np.float32))
        timestamps = get_speech_timestamps(
            waveform,
            self._vad_model,
            sampling_rate=audio_mono.metadata.sample_rate,
            return_seconds=True,
        )
        return [(float(ts["start"]), float(ts["end"])) for ts in timestamps]

    def _detect_language(self, audio_mono: Audio, voiced_spans: list[tuple[float, float]]) -> str:
        """Run Whisper language detection on a 30s window of voiced audio.

        Whisper's auto-detection only inspects the first 30s of input. When
        the file opens with silence/music/credits, that window contains no
        speech and detection picks the closest-looking thing (typically
        English). Concatenating up to 30 seconds of voiced audio fixes this.
        """
        import numpy as np

        sample_rate = audio_mono.metadata.sample_rate
        chunks: list[np.ndarray] = []
        remaining = _WHISPER_LANGUAGE_SAMPLES
        for start, end in voiced_spans:
            if remaining <= 0:
                break
            chunk = audio_mono.data[int(start * sample_rate) : int(end * sample_rate)][:remaining]
            chunks.append(chunk)
            remaining -= len(chunk)

        voiced_audio = np.concatenate(chunks).astype(np.float32)
        language, _, _ = self._model.detect_language(audio=voiced_audio)
        return language

    def _transcribe_with_diarization(
        self, audio_mono: Audio, language: str | None, vocabulary: list[str]
    ) -> Transcription:
        """Transcribe with word timestamps and assign speakers via pyannote."""
        import numpy as np
        import torch

        if self._diarization_pipeline is None:
            self._init_diarization()

        audio_data = audio_mono.data
        transcription_result = self._run_whisper(audio_data, language, vocabulary)

        waveform = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0)
        diarization_result = self._diarization_pipeline(
            {"waveform": waveform, "sample_rate": audio_mono.metadata.sample_rate}
        )

        transcription = self._process_transcription_result(transcription_result)

        # Capture original Whisper segments before flattening to words. The
        # diarization rebuild via Transcription(words=...) regroups by speaker,
        # which loses the per-segment confidence M1.3 plumbed through. We
        # re-attach by max-overlap match below so M2's confidence-aware
        # translation prompts have signal on the diarized path too.
        whisper_segments = transcription.segments

        all_words: list[TranscriptionWord] = []
        for seg in transcription.segments:
            all_words.extend(seg.words)

        if all_words:
            all_words = self._assign_speakers_to_words(all_words, diarization_result)

        rebuilt = Transcription(words=all_words, language=transcription.language)
        _attach_confidence_by_overlap(rebuilt.segments, whisper_segments)
        return rebuilt

    def _transcribe_local(self, audio: Audio, vocabulary: list[str]) -> Transcription:
        """Transcribe using local Whisper model.

        When ``enable_vad`` is True (default), Silero VAD locates voiced
        regions and a 30s voiced window is used for Whisper language
        detection -- avoiding the well-known failure where Whisper locks
        onto the wrong language because the first 30s of input is silence
        or music. The detected language is then passed into
        ``transcribe()`` so chunked decoding stays consistent. If VAD
        finds no speech, an empty Transcription is returned without
        invoking Whisper.
        """
        if self._model is None:
            self._init_local()

        audio_mono = audio.to_mono().resample(_WHISPER_SAMPLE_RATE)

        language: str | None = None
        if self.enable_vad:
            voiced_spans = self._run_vad(audio_mono)
            if not voiced_spans:
                return Transcription(segments=[])
            language = self._detect_language(audio_mono, voiced_spans)

        if self.enable_diarization:
            return self._transcribe_with_diarization(audio_mono, language, vocabulary)

        transcription_result = self._run_whisper(audio_mono.data, language, vocabulary)
        return self._process_transcription_result(transcription_result)

    def transcribe(self, media: Audio | Video, vocabulary: list[str] | None = None) -> Transcription:
        """Transcribe audio or video to text.

        ``vocabulary`` overrides the constructor default for this call only;
        a per-call list wins over the instance's vocabulary so one
        :class:`AudioToText` instance can serve multiple tenants. Pass
        ``None`` (the default) to use the constructor's list.
        """
        if isinstance(media, Video):
            if media.audio.is_silent:
                return Transcription(segments=[])
            audio = media.audio
        elif isinstance(media, Audio):
            if media.is_silent:
                return Transcription(segments=[])
            audio = media
        else:
            raise TypeError(f"Unsupported media type: {type(media)}. Expected Audio or Video.")

        effective_vocab = self.vocabulary if vocabulary is None else _normalize_vocabulary(vocabulary)
        return self._transcribe_local(audio, effective_vocab)

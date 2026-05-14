"""Audio understanding using local models."""

from __future__ import annotations

import logging
from typing import Any, Literal

from videopython.ai._device import log_device_initialization, release_device_memory, select_device
from videopython.audio import Audio
from videopython.base.description import AudioClassification, AudioEvent
from videopython.base.text.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import Video

logger = logging.getLogger(__name__)

# Whisper's initial_prompt budget; longer prompts are silently truncated by the decoder.
_INITIAL_PROMPT_TOKEN_BUDGET = 224
_INITIAL_PROMPT_TEMPLATE = "Transcript may include the following names: {terms}."


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


def _build_initial_prompt(vocabulary: list[str]) -> str | None:
    """Render the prompt and trim tail terms until it fits Whisper's
    224-token ``initial_prompt`` budget; ``None`` for empty input."""
    if not vocabulary:
        return None

    import whisper.tokenizer

    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=True, task="transcribe")
    kept = list(vocabulary)
    while kept and len(tokenizer.encode(_render_initial_prompt(kept))) > _INITIAL_PROMPT_TOKEN_BUDGET:
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
    """Stamp Whisper confidence (avg_logprob, no_speech_prob, compression_ratio)
    onto ``target_segments`` from the ``source_segments`` they overlap most with.

    Used to re-attach per-segment confidence after diarization rebuilds segments
    from words and drops the original Whisper-segment metadata. Whisper's
    confidence is window-level, not phoneme-level, so overlap-by-time is the
    right granularity — re-deriving per-word and re-aggregating wouldn't be
    more accurate.

    Mutates ``target_segments`` in place. Segments with no overlap to any
    source segment are left untouched (their confidence stays None).
    """
    for tgt in target_segments:
        best_overlap = 0.0
        best_src: TranscriptionSegment | None = None
        for src in source_segments:
            overlap = max(0.0, min(tgt.end, src.end) - max(tgt.start, src.start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_src = src
        if best_src is not None:
            tgt.avg_logprob = best_src.avg_logprob
            tgt.no_speech_prob = best_src.no_speech_prob
            tgt.compression_ratio = best_src.compression_ratio


class AudioToText:
    """Transcription service for audio and video using local Whisper models.

    Uses openai-whisper for transcription (with word-level timestamps) and
    pyannote-audio for optional speaker diarization. By default, Silero VAD
    runs before Whisper to gate language detection on a 30s window built from
    voiced regions only — fixes Whisper's tendency to lock onto the wrong
    language when the file opens with silence, music, or non-vocal credits.
    Disable with ``enable_vad=False`` to reproduce pre-0.27 behaviour.

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
        """Kwargs threaded into ``whisper.Whisper.transcribe`` from both call sites.
        ``initial_prompt`` is omitted entirely on the no-vocab path."""
        kwargs: dict[str, Any] = {
            "word_timestamps": True,
            "language": language,
            "condition_on_previous_text": self.condition_on_previous_text,
            "no_speech_threshold": self.no_speech_threshold,
            "logprob_threshold": self.logprob_threshold,
        }
        prompt = _build_initial_prompt(vocabulary)
        if prompt is not None:
            kwargs["initial_prompt"] = prompt
        return kwargs

    def _init_local(self) -> None:
        """Initialize local Whisper model."""
        import whisper

        self._model = whisper.load_model(name=self.model_name, device=self.device)

    def _init_diarization(self) -> None:
        """Initialize pyannote speaker diarization pipeline."""
        import torch
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]

        self._diarization_pipeline = Pipeline.from_pretrained(self.PYANNOTE_DIARIZATION_MODEL)
        self._diarization_pipeline.to(torch.device(self.device))

    def _init_vad(self) -> None:
        """Initialize Silero VAD model.

        The model is ~2 MB and CPU-fast (~5-15s for a 90 min movie); we keep
        it on CPU regardless of ``self.device`` since dispatch overhead would
        outweigh inference cost.
        """
        from silero_vad import load_silero_vad

        self._vad_model = load_silero_vad()

    def unload(self) -> None:
        """Release the Whisper, diarization, and VAD models so the next call re-initializes.

        Used by low-memory dubbing to free VRAM between pipeline stages.
        """
        self._model = None
        self._diarization_pipeline = None
        self._vad_model = None
        release_device_memory(self.device)

    def _process_transcription_result(self, transcription_result: dict) -> Transcription:
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
        """Assign speaker labels to words based on diarization segment overlap.

        For each word, finds the diarization segment with the greatest time overlap
        and assigns that speaker. Words with no overlapping diarization segment get
        the nearest speaker by midpoint distance.
        """
        speaker_segments: list[tuple[float, float, str]] = []
        # pyannote-audio 4.x returns DiarizeOutput; use exclusive_speaker_diarization
        # (no overlapping turns) for cleaner word assignment.
        annotation = getattr(diarization_result, "exclusive_speaker_diarization", diarization_result)
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append((turn.start, turn.end, speaker))

        if not speaker_segments:
            return words

        result = []
        for word in words:
            best_speaker: str | None = None
            best_overlap = 0.0

            for seg_start, seg_end, speaker in speaker_segments:
                overlap = max(0.0, min(word.end, seg_end) - max(word.start, seg_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker

            if best_speaker is None:
                word_mid = (word.start + word.end) / 2.0
                best_dist = float("inf")
                for seg_start, seg_end, speaker in speaker_segments:
                    seg_mid = (seg_start + seg_end) / 2.0
                    dist = abs(word_mid - seg_mid)
                    if dist < best_dist:
                        best_dist = dist
                        best_speaker = speaker

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

        all_words: list[TranscriptionWord] = list(transcription.words)
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

        import whisper

        audio_mono = audio.to_mono().resample(whisper.audio.SAMPLE_RATE)
        waveform = torch.from_numpy(audio_mono.data.astype(np.float32)).unsqueeze(0)
        diarization_result = self._diarization_pipeline(
            {"waveform": waveform, "sample_rate": audio_mono.metadata.sample_rate}
        )

        all_words = self._assign_speakers_to_words(all_words, diarization_result)
        return Transcription(words=all_words, language=transcription.language)

    def _run_vad(self, audio_mono: Audio) -> list[tuple[float, float]]:
        """Return voiced spans in seconds using Silero VAD.

        Audio must already be mono at ``whisper.audio.SAMPLE_RATE`` (16 kHz),
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
        English). Concatenating voiced spans up to 30s and running
        ``model.detect_language()`` on the resulting mel fixes this.
        """
        import numpy as np
        import torch
        import whisper

        sample_rate = audio_mono.metadata.sample_rate
        chunks: list[np.ndarray] = []
        remaining = whisper.audio.N_SAMPLES
        for start, end in voiced_spans:
            if remaining <= 0:
                break
            chunk = audio_mono.data[int(start * sample_rate) : int(end * sample_rate)][:remaining]
            chunks.append(chunk)
            remaining -= len(chunk)

        voiced_audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
        padded = whisper.audio.pad_or_trim(torch.from_numpy(voiced_audio))
        mel = whisper.audio.log_mel_spectrogram(padded, n_mels=self._model.dims.n_mels).to(self._model.device)

        _, probs = self._model.detect_language(mel)
        return max(probs, key=probs.get)

    def _transcribe_with_diarization(
        self, audio_mono: Audio, language: str | None, vocabulary: list[str]
    ) -> Transcription:
        """Transcribe with word timestamps and assign speakers via pyannote."""
        import numpy as np
        import torch

        if self._diarization_pipeline is None:
            self._init_diarization()

        audio_data = audio_mono.data
        transcription_result = self._model.transcribe(audio=audio_data, **self._transcribe_kwargs(language, vocabulary))

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
        import whisper

        if self._model is None:
            self._init_local()

        audio_mono = audio.to_mono().resample(whisper.audio.SAMPLE_RATE)

        language: str | None = None
        if self.enable_vad:
            voiced_spans = self._run_vad(audio_mono)
            if not voiced_spans:
                return Transcription(segments=[])
            language = self._detect_language(audio_mono, voiced_spans)

        if self.enable_diarization:
            return self._transcribe_with_diarization(audio_mono, language, vocabulary)

        transcription_result = self._model.transcribe(
            audio=audio_mono.data, **self._transcribe_kwargs(language, vocabulary)
        )
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


class AudioClassifier:
    """Audio event and sound classification using AST."""

    SUPPORTED_MODELS: list[str] = ["MIT/ast-finetuned-audioset-10-10-0.4593"]
    AST_SAMPLE_RATE: int = 16000
    AST_CHUNK_SECONDS: float = 10.0
    AST_HOP_SECONDS: float = 5.0

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        confidence_threshold: float = 0.3,
        top_k: int = 10,
        device: str | None = None,
    ):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Supported: {self.SUPPORTED_MODELS}")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.device = select_device(device, mps_allowed=True)
        log_device_initialization(
            "AudioClassifier",
            requested_device=device,
            resolved_device=self.device,
        )

        self._model: Any = None
        self._processor: Any = None
        self._labels: list[str] = []

    def _init_local(self) -> None:
        """Initialize local AST model from HuggingFace."""
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self._processor = ASTFeatureExtractor.from_pretrained(self.model_name)
        self._model = ASTForAudioClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        self._labels = [self._model.config.id2label[i] for i in range(len(self._model.config.id2label))]

    def _merge_events(self, events: list[AudioEvent], gap_threshold: float = 0.5) -> list[AudioEvent]:
        """Merge consecutive events of the same class."""
        if not events:
            return []

        events_by_label: dict[str, list[AudioEvent]] = {}
        for event in events:
            if event.label not in events_by_label:
                events_by_label[event.label] = []
            events_by_label[event.label].append(event)

        merged = []
        for label, label_events in events_by_label.items():
            sorted_events = sorted(label_events, key=lambda e: e.start)
            current = sorted_events[0]

            for next_event in sorted_events[1:]:
                if next_event.start - current.end <= gap_threshold:
                    current = AudioEvent(
                        start=current.start,
                        end=next_event.end,
                        label=label,
                        confidence=max(current.confidence, next_event.confidence),
                    )
                else:
                    merged.append(current)
                    current = next_event

            merged.append(current)

        return sorted(merged, key=lambda e: e.start)

    def _classify_local(self, audio: Audio) -> AudioClassification:
        """Classify audio using local AST model with sliding window."""
        import numpy as np
        import torch

        if self._model is None:
            self._init_local()

        audio_processed = audio.to_mono().resample(self.AST_SAMPLE_RATE)
        audio_data = audio_processed.data.astype(np.float32)

        chunk_samples = int(self.AST_CHUNK_SECONDS * self.AST_SAMPLE_RATE)
        hop_samples = int(self.AST_HOP_SECONDS * self.AST_SAMPLE_RATE)
        total_samples = len(audio_data)

        all_chunk_probs = []
        chunk_times = []

        if total_samples <= chunk_samples:
            chunks = [(0, audio_data)]
        else:
            chunks = []
            start = 0
            while start < total_samples:
                end = min(start + chunk_samples, total_samples)
                chunk = audio_data[start:end]
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
                chunks.append((start, chunk))
                start += hop_samples

        for start_sample, chunk in chunks:
            start_time = start_sample / self.AST_SAMPLE_RATE

            inputs = self._processor(
                chunk,
                sampling_rate=self.AST_SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits[0]
                probs = torch.sigmoid(logits).cpu().numpy()

            all_chunk_probs.append(probs)
            chunk_times.append(start_time)

        chunk_probs_array = np.array(all_chunk_probs)

        events = []
        for start_time, probs in zip(chunk_times, chunk_probs_array):
            end_time = start_time + self.AST_CHUNK_SECONDS
            top_indices = np.argsort(probs)[-self.top_k :][::-1]

            for class_idx in top_indices:
                confidence = float(probs[class_idx])
                if confidence >= self.confidence_threshold:
                    label = self._labels[class_idx]
                    events.append(
                        AudioEvent(
                            start=start_time,
                            end=min(end_time, total_samples / self.AST_SAMPLE_RATE),
                            label=label,
                            confidence=confidence,
                        )
                    )

        merged_events = self._merge_events(events)

        clip_preds = np.mean(chunk_probs_array, axis=0)
        top_clip_indices = np.argsort(clip_preds)[-self.top_k :][::-1]
        clip_predictions = {
            self._labels[idx]: float(clip_preds[idx])
            for idx in top_clip_indices
            if clip_preds[idx] >= self.confidence_threshold
        }

        return AudioClassification(events=merged_events, clip_predictions=clip_predictions)

    def classify(self, media: Audio | Video) -> AudioClassification:
        """Classify audio events in audio or video."""
        if isinstance(media, Video):
            if media.audio.is_silent:
                return AudioClassification(events=[], clip_predictions={})
            audio = media.audio
        elif isinstance(media, Audio):
            if media.is_silent:
                return AudioClassification(events=[], clip_predictions={})
            audio = media
        else:
            raise TypeError(f"Unsupported media type: {type(media)}. Expected Audio or Video.")

        return self._classify_local(audio)

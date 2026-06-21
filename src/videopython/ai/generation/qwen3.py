"""Qwen3-Instruct translation backend (M2).

GGUF inference via ``llama-cpp-python``. One model for now —
``Qwen3-4B-Instruct-2507`` (Apache-2.0, ~2.4 GB Q4_K_M). The original M2
plan called for low/medium/high tiers (4B / 8B / 30B-A3B); we deferred
that complexity until M2.4 eval data shows the larger models actually
deliver a quality lift worth the VRAM cost.

Latency note: on CPU the 4B model is roughly 10-15× slower than
:class:`MarianTranslator` per the M2.1 spike. On GPU it lands within ~2×
of Marian. Translation quality is decisively higher than Marian on
context-dependent and idiomatic content. The pipeline's
:class:`LocalDubbingPipeline` chooses based on ``device`` + the
``translator`` kwarg.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from videopython.ai._device import release_device_memory, select_device
from videopython.ai._predictor import ManagedPredictor
from videopython.ai._revisions import pinned
from videopython.ai.generation.translation import (
    LANGUAGE_NAMES,
    MarianTranslator,
    _is_translatable_text,
)
from videopython.base.transcription import TranscriptionSegment

# Imported under TYPE_CHECKING only — qwen3 sits below videopython.ai.dubbing
# in the import order (pipeline.py imports Qwen3Translator), so a top-level
# import would create a cycle. The runtime constructor reaches for it via a
# lazy local import inside translate_segments.
if TYPE_CHECKING:
    from videopython.ai.dubbing.models import TranslatedSegment

logger = logging.getLogger(__name__)


# Default model. Constants are module-level so an eval harness or future
# tier pick can override at the call site without forking the class.
DEFAULT_REPO_ID = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
DEFAULT_FILENAME = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"


# Average characters per second of natural speech, used to derive the
# per-segment ``target_chars`` budget. Rough field measurements; the prompt
# tells Qwen this is a target ±15%, not a hard cap.
_SPEECH_CHARS_PER_SEC: dict[str, float] = {
    "en": 14.0,
    "es": 14.0,
    "pt": 13.5,
    "it": 13.5,
    "fr": 13.0,
    "de": 12.0,
    "pl": 12.5,
    "nl": 12.5,
    "ru": 12.0,
    "uk": 12.0,
    "cs": 12.0,
    "sk": 12.0,
    "ro": 13.0,
    "hu": 12.0,
    "fi": 11.0,
    "sv": 12.5,
    "da": 13.0,
    "nb": 13.0,
    "no": 13.0,
    "ja": 8.0,
    "ko": 9.0,
    "zh": 7.0,
    "zh-CN": 7.0,
    "zh-TW": 7.0,
    "th": 9.0,
    "vi": 11.0,
    "ar": 10.0,
    "he": 10.0,
    "hi": 11.0,
    "ta": 10.0,
    "id": 12.0,
    "ms": 12.0,
    "tr": 12.0,
    "el": 12.0,
}
_SPEECH_CHARS_DEFAULT = 12.0


# Qwen's avg_logprob is in [-inf, 0]. Values below this threshold mark a
# transcription window we don't trust — Qwen gets a hint not to over-anchor.
_LOW_LOGPROB_HINT_THRESHOLD = -1.0


# Conservative chars-per-token used to size chunks without invoking the
# tokenizer. Morphologically rich languages land around 1.5-2.0
# chars/token; ASCII is ~3-4. We use the low end so chunks stay safe for
# any source language.
_CHARS_PER_TOKEN = 2.0
# Token reserve for the system prompt + user-prompt envelope ("Input
# segments:" / "Translations (...)" wrappers). Empirical upper bound.
_PROMPT_OVERHEAD_TOKENS = 300
# Per-segment JSON wrapper cost (keys, braces, commas, index). Added on
# top of len(seg.text) when sizing chunks.
_SEGMENT_ENVELOPE_CHARS = 40


def _chunk_segment_indices(
    segments: list[TranscriptionSegment],
    n_ctx: int,
    max_tokens: int,
) -> list[list[int]]:
    """Group positions in ``segments`` into batches that fit one Qwen call.

    Each batch must satisfy ``prompt_tokens + max_tokens <= n_ctx``, which
    llama.cpp enforces. We approximate prompt token count from character
    length using ``_CHARS_PER_TOKEN``; the conservative ratio means a chunk
    estimated at the budget will tokenize to comfortably less.

    A segment whose own serialized form exceeds the per-call budget goes in
    its own chunk anyway — better to let llama.cpp report a clean overflow
    on one giant segment than to silently swallow it.
    """
    prompt_token_budget = n_ctx - max_tokens - _PROMPT_OVERHEAD_TOKENS
    if prompt_token_budget <= 0:
        return [[i] for i in range(len(segments))]
    char_budget = int(prompt_token_budget * _CHARS_PER_TOKEN)

    chunks: list[list[int]] = []
    current: list[int] = []
    current_chars = 0
    for i, seg in enumerate(segments):
        seg_chars = len(seg.text) + _SEGMENT_ENVELOPE_CHARS
        if current and current_chars + seg_chars > char_budget:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(i)
        current_chars += seg_chars
    if current:
        chunks.append(current)
    return chunks


def _target_chars_for(duration_seconds: float, target_lang: str) -> int:
    """Character-count budget for a segment of ``duration_seconds`` in ``target_lang``."""
    rate = _SPEECH_CHARS_PER_SEC.get(target_lang, _SPEECH_CHARS_DEFAULT)
    return max(1, int(duration_seconds * rate * 1.15))


def _build_system_prompt(source_lang: str, target_lang: str) -> str:
    """Stable system + format spec. The few-shot example uses generic
    phrases (no fixture-specific content) so the prompt generalizes.
    """
    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    return (
        f"You are a professional dub translator. Translate from {src_name} to {tgt_name}.\n"
        "Preserve register and proper nouns. Match each segment's syllable count so the\n"
        "dub fits the original timing — translation is for spoken audio, not subtitles.\n"
        "Aim for ``target_chars`` characters per segment (±15%).\n"
        "If a segment is non-speech filler (grunts, laughter, music cues) keep it as filler in\n"
        "the target language; do not invent content.\n"
        "If a segment carries ``low_confidence``, the source transcription may be wrong;\n"
        "translate conservatively rather than committing to a specific phrase.\n"
        "\n"
        "Output one JSON object per line, no preamble, no commentary, no markdown:\n"
        '{"i": <segment_index>, "translated": "<text>"}\n'
    )


def _build_user_prompt(segments: list[TranscriptionSegment], target_lang: str) -> str:
    """Per-call body — the segments to translate."""
    lines: list[str] = []
    for idx, seg in enumerate(segments):
        budget = _target_chars_for(seg.end - seg.start, target_lang)
        entry: dict[str, Any] = {
            "i": idx,
            "text": seg.text,
            "target_chars": budget,
        }
        if seg.avg_logprob is not None and seg.avg_logprob < _LOW_LOGPROB_HINT_THRESHOLD:
            entry["low_confidence"] = True
        lines.append(json.dumps(entry, ensure_ascii=False))
    request_block = "\n".join(lines)
    return (
        f"Input segments:\n{request_block}\nTranslations (one JSON object per line, exactly {len(segments)} lines):\n"
    )


def _parse_jsonl_response(raw: str) -> dict[int, str]:
    """Extract ``{i: translated_text}`` from Qwen output. Permissive — tolerates
    markdown fences and preamble lines that the model occasionally adds."""
    parsed: dict[int, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "i" in obj and "translated" in obj:
            try:
                parsed[int(obj["i"])] = str(obj["translated"])
            except (TypeError, ValueError):
                continue
    return parsed


class Qwen3Translator(ManagedPredictor):
    """Qwen3-Instruct translation via llama-cpp-python (GGUF).

    Args:
        device: ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` for auto.
        marian_fallback: If True (default), fall back to Marian for any
            segment that fails Qwen's parse retry. Set False to disable
            (failures land in ``translation_failures`` instead).
        repo_id: HuggingFace repo for the GGUF weights. Defaults to
            ``DEFAULT_REPO_ID``; override for eval harnesses.
        filename: GGUF filename within ``repo_id``. Defaults to
            ``DEFAULT_FILENAME``.
        n_ctx: llama.cpp context window. ``translate_segments`` splits the
            input across multiple calls when it doesn't fit, so 8192 stays
            safe even for very long sources; raise to reduce the number of
            calls (and gain cross-segment context per call) at the cost of
            VRAM. Hard cap is the model's training context (262K for
            Qwen3-4B-Instruct-2507).
        max_tokens: Generation cap per call. 4× the input character count
            is a safe upper bound for translation output.
        temperature: Decoding temperature. 0.1 keeps output structurally
            consistent (high JSON parse rate) without being deterministic.
    """

    def __init__(
        self,
        device: str | None = None,
        marian_fallback: bool = True,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_FILENAME,
        n_ctx: int = 8192,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ):
        self.device = device
        self.marian_fallback = marian_fallback
        self.repo_id = repo_id
        self.filename = filename
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Lazily initialized.
        self._llm: Any = None
        self._marian: MarianTranslator | None = None
        # Tracks which segment indices both Qwen and Marian failed on. The
        # pipeline reads this to populate DubbingResult.translation_failures.
        self._failures_last_call: list[int] = []

    def _init_local(self) -> None:
        """Download (if needed) and load the GGUF weights."""
        from huggingface_hub import hf_hub_download

        from videopython.ai._optional import require

        Llama = require("llama_cpp", "ai", feature="Qwen3Translator").Llama

        # Warn about CPU latency at load time (not __init__) — the warning is
        # about runtime cost, which only applies once the model is actually
        # loaded. Construction is cheap; tests instantiate Qwen3Translator
        # without intending to run inference, so __init__ shouldn't shout.
        resolved = select_device(self.device, mps_allowed=True)
        if resolved == "cpu":
            logger.warning(
                "Qwen3Translator on CPU is ~10-15x slower than MarianTranslator. "
                "Consider translator='marian' for development or pass device='cuda'/'mps'.",
            )

        logger.info("Qwen3Translator: loading %s", self.filename)
        model_path = Path(hf_hub_download(repo_id=self.repo_id, filename=self.filename, revision=pinned(self.repo_id)))

        # n_gpu_layers=-1 offloads everything to GPU when one is available;
        # 0 forces CPU. llama-cpp-python's Metal/CUDA support detects and
        # uses whatever the build was compiled against.
        n_gpu_layers = 0 if resolved == "cpu" else -1
        # n_threads omitted on purpose — llama-cpp-python defaults to a
        # sensible per-host value (min(physical cores, 4)). Hardcoding 8
        # under-utilizes a 16-core box and over-subscribes a 4-core CI.
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self.n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def _qwen_translate(
        self, segments: list[TranscriptionSegment], target_lang: str, source_lang: str
    ) -> dict[int, str]:
        """One Qwen call to translate all segments. Empty result on parse failure."""
        if self._llm is None:
            self._init_local()

        system = _build_system_prompt(source_lang, target_lang)
        user = _build_user_prompt(segments, target_lang)
        prompt = system + user

        response = self._llm(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=None,
        )
        raw = response["choices"][0]["text"]
        return _parse_jsonl_response(raw)

    def _qwen_translate_chunked(
        self,
        segments: list[TranscriptionSegment],
        target_lang: str,
        source_lang: str,
        progress_callback: Callable[[float], None] | None = None,
        progress_start: float = 0.0,
        progress_end: float = 1.0,
    ) -> dict[int, str]:
        """Translate ``segments`` across one or more Qwen calls.

        Returns a dict keyed by position in ``segments``. Splitting into
        chunks keeps each call under llama.cpp's ``n_ctx`` cap — without
        chunking, a long source with hundreds of dense segments easily
        blows past the default 8192 token window.

        Progress is reported as a linear ramp from ``progress_start`` to
        ``progress_end``, one tick per chunk completed.
        """
        results: dict[int, str] = {}
        if not segments:
            if progress_callback is not None:
                progress_callback(progress_end)
            return results

        chunks = _chunk_segment_indices(segments, self.n_ctx, self.max_tokens)
        if len(chunks) > 1:
            logger.info(
                "Qwen3Translator: splitting %d segments into %d chunks (n_ctx=%d)",
                len(segments),
                len(chunks),
                self.n_ctx,
            )
        for chunk_num, chunk_positions in enumerate(chunks):
            chunk_segments = [segments[p] for p in chunk_positions]
            chunk_result = self._qwen_translate(chunk_segments, target_lang, source_lang)
            # chunk_result keys are 0..len(chunk_positions)-1; map back to
            # positions in the caller-provided ``segments`` list.
            for local_idx, text in chunk_result.items():
                if 0 <= local_idx < len(chunk_positions):
                    results[chunk_positions[local_idx]] = text
            if progress_callback is not None:
                fraction = (chunk_num + 1) / len(chunks)
                progress_callback(progress_start + (progress_end - progress_start) * fraction)
        return results

    def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        target_lang: str,
        source_lang: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[TranslatedSegment]:
        """Translate segments via Qwen with parse-retry + optional Marian fallback.

        The progress_callback ramps from 0 to 0.5 across the first-pass
        Qwen chunks, hits 0.9 after the optional retry/fallback, and 1.0
        at the end. Input larger than the model's context window is split
        across multiple Qwen calls (see ``_qwen_translate_chunked``).
        """
        effective_source = source_lang or "en"
        self._failures_last_call = []

        translatable_indices = [i for i, seg in enumerate(segments) if _is_translatable_text(seg.text)]
        translatable_segments = [segments[i] for i in translatable_indices]

        # First attempt — chunked to fit n_ctx.
        qwen_results = self._qwen_translate_chunked(
            translatable_segments,
            target_lang,
            effective_source,
            progress_callback=progress_callback,
            progress_start=0.0,
            progress_end=0.5,
        )

        # Identify segments Qwen failed (unparseable or missing index).
        # Indices in qwen_results / translatable_segments are 0-based positions
        # within translatable_segments, NOT positions in the full ``segments``
        # list. Map back at the end.
        missing_local_indices = [li for li in range(len(translatable_segments)) if li not in qwen_results]

        # Retry once on the missing subset with stricter instructions.
        if missing_local_indices:
            retry_segments = [translatable_segments[li] for li in missing_local_indices]
            logger.info(
                "Qwen3Translator: retrying %d/%d segments after first parse",
                len(retry_segments),
                len(translatable_segments),
            )
            retry_results = self._qwen_translate_chunked(
                retry_segments,
                target_lang,
                effective_source,
            )
            # retry_results keys are positions in retry_segments; map back to
            # translatable_segments.
            for retry_local, translation in retry_results.items():
                qwen_results[missing_local_indices[retry_local]] = translation
        if progress_callback is not None:
            progress_callback(0.9)

        # Anything still missing → Marian fallback (or surface as failure).
        still_missing_local = [li for li in range(len(translatable_segments)) if li not in qwen_results]
        if still_missing_local and self.marian_fallback:
            fallback_segments = [translatable_segments[li] for li in still_missing_local]
            logger.warning(
                "Qwen3Translator: falling back to Marian for %d segments after retry",
                len(fallback_segments),
            )
            if self._marian is None:
                self._marian = MarianTranslator(device=self.device)
            try:
                fallback_translated = self._marian.translate_segments(
                    fallback_segments, target_lang=target_lang, source_lang=effective_source
                )
                for li, ts in zip(still_missing_local, fallback_translated):
                    qwen_results[li] = ts.translated_text
            except Exception as exc:
                logger.warning("Qwen3Translator: Marian fallback failed (%s)", exc)
                # Leave them missing; they'll be recorded as failures below.

        # Whatever's still missing is a hard failure. Record original-segment
        # indices (positions in the full ``segments`` list) so the caller
        # can reconcile against translated_segments.
        for li in range(len(translatable_segments)):
            if li not in qwen_results:
                self._failures_last_call.append(translatable_indices[li])

        # Lazy import to avoid a circular dep through videopython.ai.dubbing
        # (see TYPE_CHECKING import at the top of the module).
        from videopython.ai.dubbing.models import TranslatedSegment

        # Materialize TranslatedSegments parallel to the input list.
        translated_segments: list[TranslatedSegment] = []
        local_translation_for_orig: dict[int, str] = {}
        for li, original_idx in enumerate(translatable_indices):
            if li in qwen_results:
                local_translation_for_orig[original_idx] = qwen_results[li]

        for i, segment in enumerate(segments):
            translated_text = local_translation_for_orig.get(i, "")
            translated_segments.append(
                TranslatedSegment(
                    original_segment=segment,
                    translated_text=translated_text,
                    source_lang=effective_source,
                    target_lang=target_lang,
                    speaker=segment.speaker,
                    start=segment.start,
                    end=segment.end,
                )
            )

        if progress_callback is not None:
            progress_callback(1.0)
        return translated_segments

    @property
    def translation_failures(self) -> list[int]:
        """Indices (in the most recent ``segments`` input) where translation
        failed entirely. Empty if all segments translated.
        """
        return list(self._failures_last_call)

    def unload(self) -> None:
        """Release the model so the next call re-initializes. Used by
        :class:`LocalDubbingPipeline` in ``low_memory`` mode."""
        self._llm = None
        if self._marian is not None:
            self._marian.unload()
            self._marian = None
        release_device_memory(self.device)

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        """Qwen handles all of Marian's language set plus more; we expose the
        Marian set for now and let M2.4 eval add anything Qwen-only.
        """
        return LANGUAGE_NAMES.copy()

    @classmethod
    def supports(cls, source_lang: str, target_lang: str) -> bool:
        """Coverage hint for the M2.3 ``auto`` resolver."""
        if source_lang == target_lang:
            return True
        return source_lang in LANGUAGE_NAMES and target_lang in LANGUAGE_NAMES

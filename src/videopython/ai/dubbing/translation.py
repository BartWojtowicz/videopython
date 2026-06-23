"""Context-aware dub translation via a local Ollama text model.

``OllamaTranslator`` is the single translation backend: it sends the
transcription segments to a local Ollama model under a structured-output schema
and reads back length-budgeted, context-aware translations. The pipeline always
uses it (the old Marian / llama-cpp backends were removed in the Ollama
consolidation).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable

from videopython.ai._ollama import OllamaError, OllamaStructuredClient
from videopython.ai._predictor import ManagedPredictor
from videopython.base.transcription import TranscriptionSegment

if TYPE_CHECKING:
    from videopython.ai.dubbing.models import TranslatedSegment

logger = logging.getLogger(__name__)

# Default Ollama text model for translation; override via the `model` arg (and
# `ollama pull` it first). Any instruct model that supports structured output works.
DEFAULT_TRANSLATION_MODEL = "qwen3.6:27b"


def _is_translatable_text(text: str) -> bool:
    """Return True if text has enough content to be worth translating.

    Whisper routinely emits punctuation-only or single-character segments
    (" .", "...", "?", "♪"). Require at least 2 alphanumeric characters.
    """
    return sum(1 for c in text if c.isalnum()) >= 2


LANGUAGE_NAMES = {
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
    "no": "Norwegian",
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
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
}


# Average characters per second of natural speech, for the per-segment
# ``target_chars`` budget. The prompt treats it as a ±15% target, not a cap.
_SPEECH_CHARS_PER_SEC: dict[str, float] = {
    "en": 14.0, "es": 14.0, "pt": 13.5, "it": 13.5, "fr": 13.0, "de": 12.0,
    "pl": 12.5, "nl": 12.5, "ru": 12.0, "uk": 12.0, "cs": 12.0, "sk": 12.0,
    "ro": 13.0, "hu": 12.0, "fi": 11.0, "sv": 12.5, "da": 13.0, "nb": 13.0,
    "no": 13.0, "ja": 8.0, "ko": 9.0, "zh": 7.0, "zh-CN": 7.0, "zh-TW": 7.0,
    "th": 9.0, "vi": 11.0, "ar": 10.0, "he": 10.0, "hi": 11.0, "ta": 10.0,
    "id": 12.0, "ms": 12.0, "tr": 12.0, "el": 12.0,
}  # fmt: skip
_SPEECH_CHARS_DEFAULT = 12.0

# avg_logprob below this marks a transcription window we don't trust.
_LOW_LOGPROB_HINT_THRESHOLD = -1.0

# Conservative chars/token for sizing chunks without a tokenizer (low end so any
# source language stays safe), plus prompt-envelope and per-segment overheads.
_CHARS_PER_TOKEN = 2.0
_PROMPT_OVERHEAD_TOKENS = 300
_SEGMENT_ENVELOPE_CHARS = 40

_TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"i": {"type": "integer"}, "translated": {"type": "string"}},
                "required": ["i", "translated"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


def _chunk_segment_indices(segments: list[TranscriptionSegment], n_ctx: int, max_tokens: int) -> list[list[int]]:
    """Group positions in ``segments`` into batches that fit one model call.

    Each batch keeps ``prompt_tokens + max_tokens <= n_ctx``, approximated from
    character length via ``_CHARS_PER_TOKEN``. A segment whose own serialized
    form exceeds the budget gets its own chunk.
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
    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    return (
        f"You are a professional dub translator. Translate from {src_name} to {tgt_name}.\n"
        "Preserve register and proper nouns. Match each segment's syllable count so the\n"
        "dub fits the original timing -- translation is for spoken audio, not subtitles.\n"
        "Aim for `target_chars` characters per segment (+/-15%).\n"
        "If a segment is non-speech filler keep it as filler; do not invent content.\n"
        "If a segment carries `low_confidence`, translate conservatively.\n"
        "\n"
        'Return a JSON object {"translations": [{"i": <segment_index>, "translated": "<text>"}, ...]} '
        "with exactly one entry per input segment."
    )


def _build_user_prompt(segments: list[TranscriptionSegment], target_lang: str) -> str:
    lines: list[str] = []
    for idx, seg in enumerate(segments):
        entry: dict[str, Any] = {
            "i": idx,
            "text": seg.text,
            "target_chars": _target_chars_for(seg.end - seg.start, target_lang),
        }
        if seg.avg_logprob is not None and seg.avg_logprob < _LOW_LOGPROB_HINT_THRESHOLD:
            entry["low_confidence"] = True
        lines.append(json.dumps(entry, ensure_ascii=False))
    return "Input segments:\n" + "\n".join(lines) + f"\n\nTranslate all {len(segments)} segments."


def _parse_translations(data: dict[str, Any]) -> dict[int, str]:
    """Extract ``{i: translated_text}`` from the model's ``{"translations": [...]}``."""
    out: dict[int, str] = {}
    for obj in data.get("translations", []):
        if isinstance(obj, dict) and "i" in obj and "translated" in obj:
            try:
                out[int(obj["i"])] = str(obj["translated"])
            except (TypeError, ValueError):
                continue
    return out


class OllamaTranslator(ManagedPredictor):
    """Dub translation via a local Ollama text model.

    The model must support Ollama's structured-output ``format``; ``ollama pull
    <model>`` first. ``n_ctx`` sizes the per-call chunking (long sources are
    split across calls); ``options`` are extra Ollama generation options.
    """

    def __init__(
        self,
        model: str = DEFAULT_TRANSLATION_MODEL,
        *,
        host: str | None = None,
        n_ctx: int = 8192,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        client_options = {"temperature": temperature, "num_ctx": n_ctx, "num_predict": max_tokens, **(options or {})}
        self._client = OllamaStructuredClient(model=model, host=host, options=client_options)
        self._failures_last_call: list[int] = []

    def _translate_chunk(
        self, segments: list[TranscriptionSegment], target_lang: str, source_lang: str
    ) -> dict[int, str]:
        """One model call. Empty dict on unusable output (caller retries / records failure)."""
        try:
            data = self._client.generate_json(
                system=_build_system_prompt(source_lang, target_lang),
                text=_build_user_prompt(segments, target_lang),
                schema=_TRANSLATION_SCHEMA,
            )
        except OllamaError:
            return {}
        return _parse_translations(data)

    def _translate_chunked(
        self,
        segments: list[TranscriptionSegment],
        target_lang: str,
        source_lang: str,
        progress_callback: Callable[[float], None] | None = None,
        progress_start: float = 0.0,
        progress_end: float = 1.0,
    ) -> dict[int, str]:
        """Translate across one or more calls, each kept under ``n_ctx``."""
        results: dict[int, str] = {}
        if not segments:
            if progress_callback is not None:
                progress_callback(progress_end)
            return results

        chunks = _chunk_segment_indices(segments, self.n_ctx, self.max_tokens)
        if len(chunks) > 1:
            logger.info("OllamaTranslator: splitting %d segments into %d chunks", len(segments), len(chunks))
        for chunk_num, chunk_positions in enumerate(chunks):
            chunk_result = self._translate_chunk([segments[p] for p in chunk_positions], target_lang, source_lang)
            for local_idx, text in chunk_result.items():
                # Drop out-of-range model indices; those segments stay "missing" and get retried.
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
        """Translate segments with a parse-retry pass; unrecovered ones land in
        ``translation_failures`` with empty text. Progress ramps 0 -> 0.5 (first
        pass), 0.9 (after retry), 1.0 (done)."""
        from videopython.ai.dubbing.models import TranslatedSegment

        effective_source = source_lang or "en"
        self._failures_last_call = []

        translatable_indices = [i for i, seg in enumerate(segments) if _is_translatable_text(seg.text)]
        translatable_segments = [segments[i] for i in translatable_indices]

        results = self._translate_chunked(
            translatable_segments, target_lang, effective_source, progress_callback, 0.0, 0.5
        )

        missing_local = [li for li in range(len(translatable_segments)) if li not in results]
        if missing_local:
            logger.info("OllamaTranslator: retrying %d/%d segments", len(missing_local), len(translatable_segments))
            retry = self._translate_chunked(
                [translatable_segments[li] for li in missing_local],
                target_lang,
                effective_source,
                progress_callback,
                0.5,
                0.9,
            )
            for retry_local, text in retry.items():
                results[missing_local[retry_local]] = text
        if progress_callback is not None:
            progress_callback(0.9)

        for li in range(len(translatable_segments)):
            if li not in results:
                self._failures_last_call.append(translatable_indices[li])

        translation_for_orig = {translatable_indices[li]: text for li, text in results.items()}
        translated_segments = [
            TranslatedSegment(
                original_segment=seg,
                translated_text=translation_for_orig.get(i, ""),
                source_lang=effective_source,
                target_lang=target_lang,
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
            )
            for i, seg in enumerate(segments)
        ]
        if progress_callback is not None:
            progress_callback(1.0)
        return translated_segments

    @property
    def translation_failures(self) -> list[int]:
        """Indices (in the most recent ``segments`` input) where translation failed entirely."""
        return list(self._failures_last_call)

    def unload(self) -> None:
        self._client.unload()

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        return LANGUAGE_NAMES.copy()

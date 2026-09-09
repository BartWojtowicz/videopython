from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

from videopython.ai._ollama import OllamaError, OllamaStructuredClient
from videopython.ai._predictor import ManagedPredictor
from videopython.ai._text_chunks import split_text
from videopython.base.transcription import TranscriptionSegment

if TYPE_CHECKING:
    from videopython.ai.dubbing.models import TranslatedSegment

# Default Ollama text model for translation; override via the `model` arg (and
# `ollama pull` it first). Any instruct model that supports structured output works.
DEFAULT_TRANSLATION_MODEL = "qwen3.6:27b"


def _is_translatable_text(text: str) -> bool:
    """Ignore punctuation/music markers, but retain single-letter spoken words."""
    return any(c.isalnum() for c in text)


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


# Conservative character/token estimate without a language-specific tokenizer.
_CHARS_PER_TOKEN = 2.0

# Soft spoken-length hints, never a reason to discard source meaning.
_SPEECH_CHARS_PER_SEC: dict[str, float] = {
    "en": 14.0, "es": 14.0, "pt": 13.5, "it": 13.5, "fr": 13.0, "de": 12.0,
    "pl": 12.5, "nl": 12.5, "ru": 12.0, "uk": 12.0, "cs": 12.0, "sk": 12.0,
    "ro": 13.0, "hu": 12.0, "fi": 11.0, "sv": 12.5, "da": 13.0, "nb": 13.0,
    "no": 13.0, "ja": 8.0, "ko": 9.0, "zh": 7.0, "zh-CN": 7.0, "zh-TW": 7.0,
    "th": 9.0, "vi": 11.0, "ar": 10.0, "he": 10.0, "hi": 11.0, "ta": 10.0,
    "id": 12.0, "ms": 12.0, "tr": 12.0, "el": 12.0,
}  # fmt: skip


_TRANSLATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
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


def _build_system_prompt(source_lang: str, target_lang: str) -> str:
    src_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    tgt_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    return (
        f"You are a professional dub translator. Translate from {src_name} to {tgt_name}.\n"
        "Translate ONLY the target text into natural spoken language. Context is for understanding only: "
        "never translate or borrow content from context. Preserve every claim, negation, number, unit, "
        "proper name, and speaker perspective. Interpret idioms by their meaning in context, not literally. "
        "If low_confidence is set, translate conservatively without inventing missing words. "
        "Use correct financial and technical terminology. Do not summarize, explain, embellish or "
        "complete unfinished fragments. Semantic fidelity takes priority over timing or length. "
        "Prefer concise phrasing when equally faithful, but never omit meaning to fit timing.\n"
        "Aim for target_chars characters (+/-15%) using concise spoken phrasing. This is a soft "
        "timing target, not a cap: retain every claim even when it requires more characters.\n"
        "\n"
        'Return a JSON object {"translations": [{"i": <segment_index>, "translated": "<text>"}, ...]} '
        "with exactly one entry per input segment."
    )


def _parse_translations(data: dict[str, Any]) -> dict[int, str]:
    """Extract ``{i: translated_text}`` from the model's ``{"translations": [...]}``."""
    out: dict[int, str] = {}
    invalid: set[int] = set()
    entries = data.get("translations")
    if not isinstance(entries, list):
        return {}
    for obj in entries:
        if not isinstance(obj, dict) or type(obj.get("i")) is not int:
            return {}
        index = obj["i"]
        value = obj.get("translated")
        if index in out or index in invalid:
            out.pop(index, None)
            invalid.add(index)
        elif not isinstance(value, str) or not value.strip() or index < 0:
            invalid.add(index)
        else:
            out[index] = " ".join(value.split())
    return {} if invalid else out


class OllamaTranslator(ManagedPredictor):
    """Dub translation via a local Ollama text model.

    The model must support Ollama's structured-output ``format``; ``ollama pull
    <model>`` first. Long text is split into bounded requests. ``n_ctx`` reserves
    room for the prompt, source text and ``max_tokens`` output budget. ``options``
    can override these as ``num_ctx`` and ``num_predict``; effective budgets are
    validated at construction.
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
        keep_alive: str | int | None = "5m",
    ) -> None:
        client_options = {"temperature": temperature, "num_ctx": n_ctx, "num_predict": max_tokens, **(options or {})}
        self.n_ctx = int(client_options["num_ctx"])
        self.max_tokens = int(client_options["num_predict"])
        # Reserve space for target-language expansion and the JSON envelope.
        self._part_chars = min(
            800,
            int((self.n_ctx - self.max_tokens - 1000) * _CHARS_PER_TOKEN),
            int((self.max_tokens - 100) * _CHARS_PER_TOKEN / 2),
        )
        if self._part_chars < 40:
            raise ValueError(
                f"Translation requires max_tokens (num_predict) >= 140 and n_ctx (num_ctx) "
                f">= max_tokens + 1020; got n_ctx={self.n_ctx}, max_tokens={self.max_tokens}"
            )
        # Keep the model resident between bounded requests; low-memory pipelines
        # explicitly unload it at the end of translation before loading TTS.
        self._client = OllamaStructuredClient(model=model, host=host, options=client_options, keep_alive=keep_alive)
        self._failures_last_call: list[int] = []

    def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        target_lang: str,
        source_lang: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[TranslatedSegment]:
        """Translate bounded source parts independently, retrying invalid replies.

        A failed part leaves its entire parent empty in ``translation_failures``.
        """
        from videopython.ai.dubbing.models import TranslatedSegment

        effective_source = source_lang or "en"
        self._failures_last_call = []

        units: list[tuple[int, int, str]] = []
        for parent, segment in enumerate(segments):
            if _is_translatable_text(segment.text):
                for part, text in enumerate(split_text(segment.text, self._part_chars)):
                    units.append((parent, part, text))
        translated_parts: dict[int, list[str]] = {}
        failed: set[int] = set()
        for identity, (parent, part, text) in enumerate(units):
            before = units[identity - 1][2][-240:] if identity else ""
            after = units[identity + 1][2][:240] if identity + 1 < len(units) else ""
            entry: dict[str, Any] = {"i": identity, "parent": parent, "part": part, "text": text}
            source_chars = len(" ".join(segments[parent].text.split()))
            part_duration = max(0.0, segments[parent].end - segments[parent].start) * len(text) / source_chars
            entry["target_chars"] = max(1, round(part_duration * _SPEECH_CHARS_PER_SEC.get(target_lang, 12.0)))
            logprob = segments[parent].avg_logprob
            if logprob is not None and logprob < -1.0:
                entry["low_confidence"] = True
            prompt = (
                "Context only (do not translate): "
                + json.dumps({"before": before, "after": after}, ensure_ascii=False)
                + "\nTarget:\n"
                + json.dumps(entry, ensure_ascii=False)
            )
            translated = None
            schema = deepcopy(_TRANSLATION_SCHEMA)
            schema["properties"]["translations"]["items"]["properties"]["i"]["const"] = identity
            for _attempt in range(2):
                try:
                    data = self._client.generate_json(
                        system=_build_system_prompt(effective_source, target_lang),
                        text=prompt
                        + ("\nReturn exactly the requested identity and all target text." if _attempt else ""),
                        schema=schema,
                    )
                    parsed = _parse_translations(data)
                    if set(parsed) == {identity} and len(parsed[identity]) <= max(80, 6 * len(text)):
                        translated = parsed[identity]
                        break
                except OllamaError:
                    pass
            if translated is None:
                failed.add(parent)
            else:
                translated_parts.setdefault(parent, []).append(translated)
            if progress_callback is not None:
                progress_callback(0.95 * (identity + 1) / len(units))
        # Never publish an incomplete parent when just one of its parts failed.
        self._failures_last_call = sorted(failed)
        translation_for_orig = {
            parent: " ".join(parts) for parent, parts in translated_parts.items() if parent not in failed
        }
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

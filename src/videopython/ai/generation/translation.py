"""Text translation backends.

Two backends share the :class:`TranslationBackend` protocol:

- :class:`MarianTranslator` (HuggingFace Helsinki-NLP MarianMT) — fast,
  segment-isolated, available for ~30 language pairs. Default on CPU.
- :class:`Qwen3Translator` (Qwen3-4B/8B/14B-Instruct via llama-cpp-python) —
  slower but produces context-aware, length-budgeted translations. Default
  on GPU.

The pipeline picks via :class:`videopython.ai.dubbing.pipeline` based on a
``translator`` kwarg (``"auto"`` resolves at runtime).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from videopython.ai._device import log_device_initialization, release_device_memory, select_device
from videopython.ai._predictor import ManagedPredictor
from videopython.ai._revisions import pinned
from videopython.base.transcription import TranscriptionSegment

# Imported under TYPE_CHECKING to avoid a circular dep through
# videopython.ai.dubbing (the dubbing pipeline imports both
# MarianTranslator and Qwen3Translator, which both import
# TranslatedSegment from dubbing.models). Runtime users do a lazy
# local import inside translate_segments.
if TYPE_CHECKING:
    from videopython.ai.dubbing.models import TranslatedSegment


class UnsupportedLanguageError(ValueError):
    """Raised when no available translation backend supports a given
    ``(source, target)`` language pair.

    Carries the requested pair so callers can introspect:

        try:
            dubber.dub(video, target_lang="xh")
        except UnsupportedLanguageError as e:
            print(f"No backend covers {e.source_lang}->{e.target_lang}")
    """

    def __init__(self, source_lang: str, target_lang: str, message: str | None = None):
        self.source_lang = source_lang
        self.target_lang = target_lang
        super().__init__(message or f"No translation backend supports {source_lang}->{target_lang}")


def _is_translatable_text(text: str) -> bool:
    """Return True if text has enough content to be worth translating.

    Whisper routinely emits punctuation-only or single-character segments
    (" .", "...", "?", "♪") that MarianMT can hallucinate full sentences
    from. Require at least 2 alphanumeric characters to filter these out.
    """
    return sum(1 for c in text if c.isalnum()) >= 2


@runtime_checkable
class TranslationBackend(Protocol):
    """Pipeline-facing translation interface.

    Both :class:`MarianTranslator` and :class:`Qwen3Translator` satisfy
    this. The pipeline only depends on these methods.
    """

    def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        target_lang: str,
        source_lang: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[TranslatedSegment]: ...

    def unload(self) -> None: ...

    @property
    def translation_failures(self) -> list[int]:
        """Indices into the most recent ``segments`` input where the backend
        could not produce a translation. Empty for backends that never fail
        per-segment (e.g. MarianTranslator). The dubbing pipeline copies
        this onto :class:`DubbingResult.translation_failures`."""
        ...

    @staticmethod
    def get_supported_languages() -> dict[str, str]: ...


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


class MarianTranslator(ManagedPredictor):
    """Translates text between languages using local Helsinki-NLP MarianMT models."""

    # Languages without a direct opus-mt-{src}-{tgt} model. Maps (source, target)
    # to an alternative HuggingFace model identifier.
    _MODEL_OVERRIDES: dict[tuple[str, str], str] = {
        ("en", "pt"): "Helsinki-NLP/opus-mt-tc-big-en-pt",
        ("en", "ko"): "Helsinki-NLP/opus-mt-tc-big-en-ko",
        ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
        ("en", "pl"): "Helsinki-NLP/opus-mt-en-zlw",
    }

    @classmethod
    def has_model_for(cls, source_lang: str, target_lang: str) -> bool:
        """Return True if Marian has (or is likely to have) a model for ``(source, target)``.

        Same-language pairs return True (translation is the identity).
        Otherwise: True if either an entry in ``_MODEL_OVERRIDES`` exists or
        both languages are in :data:`LANGUAGE_NAMES`. The latter is a
        permissive proxy — Marian publishes ``opus-mt-{src}-{tgt}`` for
        most ISO-639-1 pairs we expose, but not all (e.g. some Asian-to-
        Asian pairs route through English). Used by the M2.3 ``auto``
        resolver as a *coverage hint*; the actual existence check happens
        at first-use download time.
        """
        if source_lang == target_lang:
            return True
        if (source_lang, target_lang) in cls._MODEL_OVERRIDES:
            return True
        return source_lang in LANGUAGE_NAMES and target_lang in LANGUAGE_NAMES

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._current_lang_pair: tuple[str, str] | None = None

    def _get_local_model_name(self, source_lang: str, target_lang: str) -> str:
        if self.model_name:
            return self.model_name
        override = self._MODEL_OVERRIDES.get((source_lang, target_lang))
        if override:
            return override
        return f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    def _init_local(self, source_lang: str, target_lang: str) -> None:
        from videopython.ai._optional import require

        _transformers = require("transformers", "translation", feature="MarianTranslator")
        MarianMTModel = _transformers.MarianMTModel
        MarianTokenizer = _transformers.MarianTokenizer

        model_name = self._get_local_model_name(source_lang, target_lang)

        requested_device = self.device
        device = select_device(self.device, mps_allowed=True)

        # Marian model names are dynamic per language pair, so pinned() returns
        # None for them by design (revision=None tracks main, the safe default).
        self._tokenizer = MarianTokenizer.from_pretrained(model_name, revision=pinned(model_name))
        self._model = MarianMTModel.from_pretrained(model_name, revision=pinned(model_name)).to(device)
        self.device = device
        log_device_initialization(
            "MarianTranslator",
            requested_device=requested_device,
            resolved_device=device,
        )
        self._current_lang_pair = (source_lang, target_lang)

    def _translate_local(self, text: str, target_lang: str, source_lang: str) -> str:
        import torch

        if self._model is None or self._current_lang_pair != (source_lang, target_lang):
            self._init_local(source_lang, target_lang)

        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_length=512)

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str | None = None,
    ) -> str:
        """Translate text to target language."""
        if not text.strip():
            return text

        effective_source = source_lang or "en"
        if effective_source == target_lang:
            return text
        return self._translate_local(text, target_lang, effective_source)

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[str]:
        """Translate multiple texts to target language.

        ``progress_callback`` is called once per batch with a fraction in
        ``[0, 1]`` representing translation-stage progress. It does not fire
        on the empty-input or same-language shortcuts (those are O(0) work
        and the caller frames its own progress events around the call).
        """
        import torch

        if not texts:
            return []

        effective_source = source_lang or "en"
        if effective_source == target_lang:
            return list(texts)
        if self._model is None or self._current_lang_pair != (effective_source, target_lang):
            self._init_local(effective_source, target_lang)

        translated: list[str] = []
        batch_size = 8
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_length=512)

            for output in outputs:
                translated.append(self._tokenizer.decode(output, skip_special_tokens=True))

            if progress_callback is not None:
                progress_callback(min(1.0, (i + len(batch)) / total))

        return translated

    def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        target_lang: str,
        source_lang: str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[TranslatedSegment]:
        """Translate transcription segments while preserving timing/speaker info.

        Segments whose text is empty or contains fewer than 2 alphanumeric
        characters are not sent to the model — they receive
        ``translated_text=""`` instead. This avoids MarianMT hallucinating
        full sentences from " .", "...", or single-token Whisper segments,
        which would otherwise be TTS'd into the dubbed track.

        ``progress_callback`` is forwarded to :meth:`translate_batch` so
        callers can render translation-stage progress without knowing the
        batch size.
        """
        # Lazy import to avoid a circular dep through videopython.ai.dubbing
        # (see TYPE_CHECKING import at the top of the module).
        from videopython.ai.dubbing.models import TranslatedSegment

        effective_source = source_lang or "en"

        translatable_indices = [i for i, segment in enumerate(segments) if _is_translatable_text(segment.text)]
        translatable_texts = [segments[i].text for i in translatable_indices]
        translated_texts = self.translate_batch(
            translatable_texts, target_lang, source_lang, progress_callback=progress_callback
        )

        translation_map: dict[int, str] = dict(zip(translatable_indices, translated_texts))

        translated_segments = []
        for i, segment in enumerate(segments):
            translated_segments.append(
                TranslatedSegment(
                    original_segment=segment,
                    translated_text=translation_map.get(i, ""),
                    source_lang=effective_source,
                    target_lang=target_lang,
                    speaker=segment.speaker,
                    start=segment.start,
                    end=segment.end,
                )
            )

        return translated_segments

    def unload(self) -> None:
        """Release the translation model so the next translate() re-initializes.

        Used by low-memory dubbing to free VRAM between pipeline stages.
        """
        self._model = None
        self._tokenizer = None
        self._current_lang_pair = None
        release_device_memory(self.device)

    @property
    def translation_failures(self) -> list[int]:
        """Marian never fails per-segment (worst case it produces poor
        output, not no output). Always empty; satisfies the
        :class:`TranslationBackend` protocol."""
        return []

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        return LANGUAGE_NAMES.copy()


# Back-compat alias. ``TextTranslator`` was the class name through 0.28.x;
# 0.29.0 renames to ``MarianTranslator`` to make room for ``Qwen3Translator``
# behind a shared :class:`TranslationBackend` protocol. The alias will be
# removed in 0.30.0.
TextTranslator = MarianTranslator

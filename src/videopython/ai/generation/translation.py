"""Text translation using local Helsinki-NLP models."""

from __future__ import annotations

from typing import Any

from videopython.ai._device import log_device_initialization, release_device_memory, select_device
from videopython.ai.dubbing.models import TranslatedSegment
from videopython.base.text.transcription import TranscriptionSegment


def _is_translatable_text(text: str) -> bool:
    """Return True if text has enough content to be worth translating.

    Whisper routinely emits punctuation-only or single-character segments
    (" .", "...", "?", "♪") that MarianMT can hallucinate full sentences
    from. Require at least 2 alphanumeric characters to filter these out.
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


class TextTranslator:
    """Translates text between languages using local seq2seq models."""

    # Languages without a direct opus-mt-{src}-{tgt} model. Maps (source, target)
    # to an alternative HuggingFace model identifier.
    _MODEL_OVERRIDES: dict[tuple[str, str], str] = {
        ("en", "pt"): "Helsinki-NLP/opus-mt-tc-big-en-pt",
        ("en", "ko"): "Helsinki-NLP/opus-mt-tc-big-en-ko",
        ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
        ("en", "pl"): "Helsinki-NLP/opus-mt-en-zlw",
    }

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
        from transformers import MarianMTModel, MarianTokenizer  # type: ignore[attr-defined]

        model_name = self._get_local_model_name(source_lang, target_lang)

        requested_device = self.device
        device = select_device(self.device, mps_allowed=True)

        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name).to(device)
        self.device = device
        log_device_initialization(
            "TextTranslator",
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
    ) -> list[str]:
        """Translate multiple texts to target language."""
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

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(**inputs, max_length=512)

            for output in outputs:
                translated.append(self._tokenizer.decode(output, skip_special_tokens=True))

        return translated

    def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslatedSegment]:
        """Translate transcription segments while preserving timing/speaker info.

        Segments whose text is empty or contains fewer than 2 alphanumeric
        characters are not sent to the model — they receive
        ``translated_text=""`` instead. This avoids MarianMT hallucinating
        full sentences from " .", "...", or single-token Whisper segments,
        which would otherwise be TTS'd into the dubbed track.
        """
        effective_source = source_lang or "en"

        translatable_indices = [i for i, segment in enumerate(segments) if _is_translatable_text(segment.text)]
        translatable_texts = [segments[i].text for i in translatable_indices]
        translated_texts = self.translate_batch(translatable_texts, target_lang, source_lang)

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

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        return LANGUAGE_NAMES.copy()

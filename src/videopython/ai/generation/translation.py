"""Text translation using local Helsinki-NLP models."""

from __future__ import annotations

from typing import Any

from videopython.ai._device import log_device_initialization, select_device
from videopython.ai.dubbing.models import TranslatedSegment
from videopython.base.text.transcription import TranscriptionSegment

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

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._current_lang_pair: tuple[str, str] | None = None

    def _get_local_model_name(self, source_lang: str, target_lang: str) -> str:
        if self.model_name:
            return self.model_name
        return f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    def _init_local(self, source_lang: str, target_lang: str) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[attr-defined]

        model_name = self._get_local_model_name(source_lang, target_lang)

        requested_device = self.device
        device = select_device(self.device, mps_allowed=True)

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
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
        """Translate transcription segments while preserving timing/speaker info."""
        effective_source = source_lang or "en"
        texts = [segment.text for segment in segments]
        translated_texts = self.translate_batch(texts, target_lang, source_lang)

        translated_segments = []
        for segment, translated_text in zip(segments, translated_texts):
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

        return translated_segments

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        return LANGUAGE_NAMES.copy()

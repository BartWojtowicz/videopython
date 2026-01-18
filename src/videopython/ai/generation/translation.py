"""Text translation with multi-backend support."""

from __future__ import annotations

from typing import Any

from videopython.ai.backends import TextTranslatorBackend, UnsupportedBackendError, get_api_key
from videopython.ai.config import get_default_backend
from videopython.ai.dubbing.models import TranslatedSegment
from videopython.base.text.transcription import TranscriptionSegment

# Language code to full name mapping
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
    """Translates text between languages.

    Supports multiple backends:
    - openai: GPT-4 for high-quality translation
    - gemini: Google Gemini for translation
    - local: Helsinki-NLP models for offline translation

    Example:
        >>> from videopython.ai.generation.translation import TextTranslator
        >>>
        >>> translator = TextTranslator(backend="openai")
        >>> spanish = translator.translate("Hello, world!", target_lang="es")
        >>> print(spanish)  # "Hola, mundo!"
    """

    SUPPORTED_BACKENDS: list[str] = ["openai", "gemini", "local"]

    def __init__(
        self,
        backend: TextTranslatorBackend | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        device: str | None = None,
    ):
        """Initialize the text translator.

        Args:
            backend: Backend to use. If None, uses config default or 'openai'.
            model_name: Model name (backend-specific).
                - openai: 'gpt-4o', 'gpt-4o-mini', etc.
                - gemini: 'gemini-2.0-flash', etc.
                - local: Helsinki-NLP model name (auto-selected based on languages)
            api_key: API key for cloud backends. If None, reads from environment.
            device: Device for local backend ('cuda', 'mps', 'cpu').
        """
        resolved_backend: str = backend if backend is not None else get_default_backend("text_translator")
        if resolved_backend not in self.SUPPORTED_BACKENDS:
            raise UnsupportedBackendError(resolved_backend, self.SUPPORTED_BACKENDS)

        self.backend: TextTranslatorBackend = resolved_backend  # type: ignore[assignment]
        self.model_name = model_name
        self.api_key = api_key
        self.device = device

        self._model: Any = None
        self._tokenizer: Any = None

    def _get_language_name(self, code: str) -> str:
        """Get full language name from code."""
        return LANGUAGE_NAMES.get(code, code)

    def _translate_openai(self, text: str, target_lang: str, source_lang: str | None) -> str:
        """Translate using OpenAI GPT."""
        from openai import OpenAI

        api_key = get_api_key("openai", self.api_key)
        client = OpenAI(api_key=api_key)

        model = self.model_name or "gpt-4o"
        target_name = self._get_language_name(target_lang)

        if source_lang:
            source_name = self._get_language_name(source_lang)
            system_prompt = (
                f"You are a professional translator. Translate the following text from "
                f"{source_name} to {target_name}. Preserve the tone, style, and meaning. "
                f"Only return the translated text, nothing else."
            )
        else:
            system_prompt = (
                f"You are a professional translator. Translate the following text to "
                f"{target_name}. Preserve the tone, style, and meaning. "
                f"Only return the translated text, nothing else."
            )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.3,  # Lower temperature for more consistent translations
        )

        return response.choices[0].message.content or text

    def _translate_gemini(self, text: str, target_lang: str, source_lang: str | None) -> str:
        """Translate using Google Gemini."""
        import google.generativeai as genai

        api_key = get_api_key("gemini", self.api_key)
        genai.configure(api_key=api_key)

        model_name = self.model_name or "gemini-2.0-flash"
        model = genai.GenerativeModel(model_name)

        target_name = self._get_language_name(target_lang)

        if source_lang:
            source_name = self._get_language_name(source_lang)
            prompt = (
                f"Translate the following text from {source_name} to {target_name}. "
                f"Preserve the tone, style, and meaning. Only return the translated text:\n\n{text}"
            )
        else:
            prompt = (
                f"Translate the following text to {target_name}. "
                f"Preserve the tone, style, and meaning. Only return the translated text:\n\n{text}"
            )

        response = model.generate_content(prompt)
        return response.text.strip()

    def _get_local_model_name(self, source_lang: str, target_lang: str) -> str:
        """Get the appropriate Helsinki-NLP model for a language pair."""
        # Helsinki-NLP model naming convention: opus-mt-{src}-{tgt}
        # Some common models have specific naming
        return f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"

    def _init_local(self, source_lang: str, target_lang: str) -> None:
        """Initialize local Helsinki-NLP model."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore[attr-defined]

        model_name = self._get_local_model_name(source_lang, target_lang)

        # Determine device
        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device
        self._current_lang_pair = (source_lang, target_lang)

    def _translate_local(self, text: str, target_lang: str, source_lang: str) -> str:
        """Translate using local Helsinki-NLP model."""
        import torch

        # Check if we need to reload model for different language pair
        if self._model is None or getattr(self, "_current_lang_pair", None) != (source_lang, target_lang):
            self._init_local(source_lang, target_lang)

        # Tokenize and translate
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_length=512)

        translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str | None = None,
    ) -> str:
        """Translate text to target language.

        Args:
            text: Text to translate.
            target_lang: Target language code (e.g., 'es', 'fr', 'de').
            source_lang: Source language code. If None, auto-detected
                (cloud backends) or defaults to 'en' (local backend).

        Returns:
            Translated text.

        Raises:
            ValueError: If language codes are invalid.
            UnsupportedBackendError: If backend is not supported.
        """
        if not text.strip():
            return text

        if self.backend == "openai":
            return self._translate_openai(text, target_lang, source_lang)
        elif self.backend == "gemini":
            return self._translate_gemini(text, target_lang, source_lang)
        elif self.backend == "local":
            # Local backend requires source language
            effective_source = source_lang or "en"
            return self._translate_local(text, target_lang, effective_source)
        else:
            raise UnsupportedBackendError(self.backend, self.SUPPORTED_BACKENDS)

    def translate_batch(
        self,
        texts: list[str],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[str]:
        """Translate multiple texts to target language.

        Args:
            texts: List of texts to translate.
            target_lang: Target language code.
            source_lang: Source language code (optional).

        Returns:
            List of translated texts.
        """
        if self.backend == "local" and self._model is not None:
            # Batch translate for efficiency
            import torch

            effective_source = source_lang or "en"

            if self._model is None or getattr(self, "_current_lang_pair", None) != (effective_source, target_lang):
                self._init_local(effective_source, target_lang)

            # Process in batches
            translated = []
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
        else:
            # Fall back to individual translations for cloud backends
            return [self.translate(text, target_lang, source_lang) for text in texts]

    def translate_segments(
        self,
        segments: list[TranscriptionSegment],
        target_lang: str,
        source_lang: str | None = None,
    ) -> list[TranslatedSegment]:
        """Translate transcription segments to target language.

        Preserves timing and speaker information while translating text.

        Args:
            segments: List of transcription segments to translate.
            target_lang: Target language code.
            source_lang: Source language code (optional).

        Returns:
            List of TranslatedSegment objects with translated text.
        """
        effective_source = source_lang or "en"

        # Extract texts
        texts = [segment.text for segment in segments]

        # Translate all at once
        translated_texts = self.translate_batch(texts, target_lang, source_lang)

        # Create TranslatedSegment objects
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
        """Get dictionary of supported language codes and names.

        Returns:
            Dictionary mapping language codes to language names.
        """
        return LANGUAGE_NAMES.copy()

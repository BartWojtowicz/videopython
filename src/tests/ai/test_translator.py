"""Tests for the Ollama-backed translation helpers and OllamaTranslator (fake client)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from videopython.ai.generation.translation import (
    LANGUAGE_NAMES,
    OllamaTranslator,
    TranslationBackend,
    _build_system_prompt,
    _build_user_prompt,
    _chunk_segment_indices,
    _parse_translations,
    _target_chars_for,
)
from videopython.base.transcription import TranscriptionSegment


def _seg(text: str, start: float = 0.0, end: float = 1.0, avg_logprob: float | None = None) -> TranscriptionSegment:
    return TranscriptionSegment(start=start, end=end, text=text, words=[], avg_logprob=avg_logprob)


class _FakeOllama:
    """Returns scripted JSON contents on successive chat() calls (last repeats)."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls = 0

    def chat(self, *, model: str, messages: list[Any], format: Any, options: dict[str, Any]) -> SimpleNamespace:
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return SimpleNamespace(message=SimpleNamespace(content=content))


def _translator_with(contents: list[str]) -> tuple[OllamaTranslator, _FakeOllama]:
    translator = OllamaTranslator(model="m")
    fake = _FakeOllama(contents)
    translator._client._client = fake  # inject inside the shared OllamaStructuredClient
    return translator, fake


# --------------------------------------------------------------------------- helpers


def test_target_chars_uses_language_rate() -> None:
    assert _target_chars_for(1.0, "en") == int(14.0 * 1.15)
    assert _target_chars_for(0.0, "en") == 1  # minimum 1


def test_build_system_prompt_names_languages() -> None:
    prompt = _build_system_prompt("en", "es")
    assert "English" in prompt
    assert "Spanish" in prompt
    assert "translations" in prompt  # describes the JSON object shape


def test_build_user_prompt_marks_low_confidence() -> None:
    prompt = _build_user_prompt([_seg("hello", avg_logprob=-2.0), _seg("world", avg_logprob=0.0)], "es")
    assert '"low_confidence": true' in prompt
    assert "target_chars" in prompt


def test_chunk_segment_indices_splits_on_budget() -> None:
    segs = [_seg("a" * 100) for _ in range(10)]
    chunks = _chunk_segment_indices(segs, n_ctx=600, max_tokens=100)
    assert len(chunks) > 1
    assert sum(len(c) for c in chunks) == 10  # every segment placed exactly once


def test_parse_translations() -> None:
    data = {"translations": [{"i": 0, "translated": "hola"}, {"i": 1, "translated": "mundo"}]}
    assert _parse_translations(data) == {0: "hola", 1: "mundo"}
    assert _parse_translations({"translations": []}) == {}


# --------------------------------------------------------------------------- OllamaTranslator


def test_translate_segments_happy_path() -> None:
    content = json.dumps({"translations": [{"i": 0, "translated": "hola"}, {"i": 1, "translated": "mundo"}]})
    translator, fake = _translator_with([content])
    out = translator.translate_segments([_seg("hello"), _seg("world")], target_lang="es", source_lang="en")
    assert [s.translated_text for s in out] == ["hola", "mundo"]
    assert translator.translation_failures == []
    assert fake.calls == 1


def test_translate_segments_retries_missing() -> None:
    translator, fake = _translator_with(
        [
            json.dumps({"translations": [{"i": 0, "translated": "hola"}]}),  # i=1 missing
            json.dumps({"translations": [{"i": 0, "translated": "mundo"}]}),  # retry batch idx 0 -> orig 1
        ]
    )
    out = translator.translate_segments([_seg("hello"), _seg("world")], target_lang="es")
    assert [s.translated_text for s in out] == ["hola", "mundo"]
    assert translator.translation_failures == []
    assert fake.calls == 2


def test_translate_segments_records_failures() -> None:
    translator, _ = _translator_with([json.dumps({"translations": []})])
    out = translator.translate_segments([_seg("hello"), _seg("world")], target_lang="es")
    assert [s.translated_text for s in out] == ["", ""]
    assert translator.translation_failures == [0, 1]


def test_non_translatable_segments_skipped() -> None:
    translator, _ = _translator_with([json.dumps({"translations": [{"i": 0, "translated": "hola"}]})])
    out = translator.translate_segments([_seg("hello"), _seg(".")], target_lang="es")
    assert out[0].translated_text == "hola"
    assert out[1].translated_text == ""  # "." is not translatable, never sent
    assert translator.translation_failures == []


def test_unload_and_protocol() -> None:
    translator, _ = _translator_with(["{}"])
    assert isinstance(translator, TranslationBackend)
    translator.unload()  # idempotent
    assert OllamaTranslator.supports("en", "es")
    assert OllamaTranslator.get_supported_languages() == LANGUAGE_NAMES

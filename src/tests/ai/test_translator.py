"""Tests for the Ollama-backed translation helpers and OllamaTranslator (fake client)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

from videopython.ai.dubbing.translation import (
    LANGUAGE_NAMES,
    OllamaTranslator,
    _build_system_prompt,
    _parse_translations,
)
from videopython.base.transcription import TranscriptionSegment


def _seg(text: str, start: float = 0.0, end: float = 1.0, avg_logprob: float | None = None) -> TranscriptionSegment:
    return TranscriptionSegment(start=start, end=end, text=text, words=[], avg_logprob=avg_logprob)


class _FakeOllama:
    """Returns scripted JSON contents on successive chat() calls (last repeats)."""

    def __init__(self, contents: list[str], capabilities: list[str] | None = None) -> None:
        self.contents = list(contents)
        self.capabilities = ["completion", "thinking"] if capabilities is None else capabilities
        self.calls = 0
        self.chat_kwargs: list[dict[str, Any]] = []
        self.generate = Mock()

    def show(self, model: str) -> SimpleNamespace:
        return SimpleNamespace(capabilities=self.capabilities)

    def chat(
        self,
        *,
        model: str,
        messages: list[Any],
        format: Any,
        options: dict[str, Any],
        **kwargs: Any,
    ) -> SimpleNamespace:
        self.chat_kwargs.append(kwargs)
        content = self.contents[min(self.calls, len(self.contents) - 1)]
        self.calls += 1
        return SimpleNamespace(message=SimpleNamespace(content=content))


def _translator_with(contents: list[str]) -> tuple[OllamaTranslator, _FakeOllama]:
    translator = OllamaTranslator(model="m")
    fake = _FakeOllama(contents)
    translator._client._client = fake  # inject inside the shared OllamaStructuredClient
    return translator, fake


# --------------------------------------------------------------------------- helpers


def test_build_system_prompt_names_languages() -> None:
    prompt = _build_system_prompt("en", "es")
    assert "English" in prompt
    assert "Spanish" in prompt
    assert "translations" in prompt  # describes the JSON object shape


def test_unbroken_translation_is_bounded_with_soft_timing_and_full_progress():
    translator = OllamaTranslator(model="m", n_ctx=1200, max_tokens=140)
    entries = []

    def generate(**kwargs):
        entry = json.loads(kwargs["text"].split("\nTarget:\n")[1])
        entries.append(entry)
        return {"translations": [{"i": entry["i"], "translated": "translated"}]}

    translator._client.generate_json = Mock(side_effect=generate)
    progress = []
    text = "漢字" * 140
    result = translator.translate_segments([_seg(text, 0, 20)], "en", "zh", progress.append)
    assert "".join(e["text"] for e in entries) == text
    assert all(len(e["text"]) <= 40 for e in entries)
    assert sum(e["target_chars"] for e in entries) == 280
    assert progress == sorted(progress)
    assert progress[-2:] == [0.95, 1.0]
    assert translator.translation_failures == []
    assert result[0].translated_text == " ".join(["translated"] * len(entries))


def test_parse_translations() -> None:
    data = {"translations": [{"i": 0, "translated": "hola"}, {"i": 1, "translated": "mundo"}]}
    assert _parse_translations(data) == {0: "hola", 1: "mundo"}
    assert _parse_translations({"translations": []}) == {}


# --------------------------------------------------------------------------- OllamaTranslator


def test_translate_segments_happy_path() -> None:
    translator, fake = _translator_with(
        [
            json.dumps({"translations": [{"i": 0, "translated": "hola"}]}),
            json.dumps({"translations": [{"i": 1, "translated": "mundo"}]}),
        ]
    )
    out = translator.translate_segments([_seg("hello"), _seg("world")], target_lang="es", source_lang="en")
    assert [s.translated_text for s in out] == ["hola", "mundo"]
    assert translator.translation_failures == []
    assert fake.calls == 2
    assert all(call["keep_alive"] == "5m" for call in fake.chat_kwargs)


def test_translate_segments_retries_missing() -> None:
    translator, fake = _translator_with(
        [
            json.dumps({"translations": [{"i": 0, "translated": "hola"}]}),  # i=1 missing
            json.dumps({"translations": []}),
            json.dumps({"translations": [{"i": 1, "translated": "mundo"}]}),  # stable identity on retry
        ]
    )
    out = translator.translate_segments([_seg("hello"), _seg("world")], target_lang="es")
    assert [s.translated_text for s in out] == ["hola", "mundo"]
    assert translator.translation_failures == []
    assert fake.calls == 3


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


def test_unload_and_languages() -> None:
    translator, fake = _translator_with(["{}"])
    translator.unload()  # idempotent
    translator.unload()
    fake.generate.assert_called_once_with(model="m", keep_alive=0)
    assert OllamaTranslator.get_supported_languages() == LANGUAGE_NAMES


class _EchoOllama:
    """Translates every input segment by echoing its per-call index (works for any chunking)."""

    def __init__(self) -> None:
        self.calls = 0

    def show(self, model: str) -> SimpleNamespace:
        return SimpleNamespace(capabilities=["completion", "thinking"])

    def chat(
        self,
        *,
        model: str,
        messages: list[Any],
        format: Any,
        options: dict[str, Any],
        **kwargs: Any,
    ) -> SimpleNamespace:
        self.calls += 1
        indices: list[int] = []
        for line in messages[1]["content"].splitlines():
            line = line.strip()
            if line.startswith("{") and '"i"' in line:
                try:
                    indices.append(int(json.loads(line)["i"]))
                except (ValueError, KeyError, TypeError):
                    pass
        content = json.dumps({"translations": [{"i": i, "translated": f"t{i}"} for i in indices]})
        return SimpleNamespace(message=SimpleNamespace(content=content))


def test_translate_segments_multiple_chunks() -> None:
    translator = OllamaTranslator(model="m", n_ctx=2000, max_tokens=300)  # small ctx forces splitting
    fake = _EchoOllama()
    translator._client._client = fake
    segs = [_seg("word " * 100, start=float(i), end=float(i) + 1) for i in range(12)]

    out = translator.translate_segments(segs, target_lang="es")

    assert fake.calls > 1  # genuinely split into multiple chunks
    assert all(s.translated_text for s in out)
    assert translator.translation_failures == []


def test_ollama_error_in_both_passes_records_failure() -> None:
    translator, _ = _translator_with(["not json at all"])  # OllamaError on every call
    out = translator.translate_segments([_seg("hello")], target_lang="es")
    assert out[0].translated_text == ""
    assert translator.translation_failures == [0]


def test_translate_segments_progress_milestones() -> None:
    content = json.dumps({"translations": [{"i": 0, "translated": "hola"}]})
    translator, _ = _translator_with([content])
    ticks: list[float] = []
    translator.translate_segments([_seg("hello")], target_lang="es", progress_callback=ticks.append)
    assert ticks[-2] == 0.95  # requests span the whole translation window
    assert ticks[-1] == 1.0


def test_translation_disables_reasoning_on_thinking_model() -> None:
    """Translation calls must run with think=False on a reasoning model.

    The default qwen3.6:27b emits its reasoning before the schema-constrained
    answer. That reasoning counts against num_predict and can leave empty content.
    """
    content = json.dumps({"translations": [{"i": 0, "translated": "hola"}]})
    translator, fake = _translator_with([content])

    out = translator.translate_segments([_seg("hello")], target_lang="es")

    assert out[0].translated_text == "hola"
    assert translator.translation_failures == []
    assert fake.chat_kwargs[0]["think"] is False


def test_rejects_ambiguous_or_invalid_results() -> None:
    assert _parse_translations({"translations": [{"i": 0, "translated": "a"}, {"i": 0, "translated": "b"}]}) == {}
    invalid_values: list[Any] = [None, 42, [], "", "   "]
    for value in invalid_values:
        assert _parse_translations({"translations": [{"i": 0, "translated": value}]}) == {}
    for index in (True, "0", 0.5, -1):
        assert _parse_translations({"translations": [{"i": index, "translated": "a"}]}) == {}


def test_one_letter_speech_is_translated() -> None:
    translator, _ = _translator_with([json.dumps({"translations": [{"i": 0, "translated": "and"}]})])
    assert translator.translate_segments([_seg("I")], target_lang="en", source_lang="pl")[0].translated_text == "and"


def test_failed_part_invalidates_whole_parent() -> None:
    translator, _ = _translator_with(
        [
            json.dumps({"translations": [{"i": 0, "translated": "first"}]}),
            json.dumps({"translations": []}),
        ]
    )
    source = _seg("A sentence. " * 100)
    out = translator.translate_segments([source], target_lang="pl")
    assert out[0].translated_text == ""
    assert out[0].original_segment is source
    assert translator.translation_failures == [0]


def test_complete_json_at_output_limit_is_rejected() -> None:
    import pytest

    from videopython.ai._ollama import OllamaError

    translator, fake = _translator_with(["{}"])
    response = SimpleNamespace(message=SimpleNamespace(content="{}"), done_reason="length")
    with patch.object(fake, "chat", return_value=response), pytest.raises(OllamaError, match="exhausted"):
        translator._client.generate_json(system="test", text="test", schema={})


def test_extra_invalid_identity_invalidates_response() -> None:
    assert (
        _parse_translations(
            {
                "translations": [
                    {"i": 0, "translated": "valid"},
                    {"i": -1, "translated": "invalid"},
                ]
            }
        )
        == {}
    )


@pytest.mark.parametrize("kwargs", [{"n_ctx": 4096}, {"max_tokens": 139}, {"options": {"num_ctx": 4096}}])
def test_invalid_effective_budgets_fail_at_construction(kwargs):
    with pytest.raises(ValueError, match=r"n_ctx.*max_tokens"):
        OllamaTranslator(**kwargs)


def test_options_can_supply_valid_smaller_budgets():
    translator = OllamaTranslator(n_ctx=4096, options={"num_predict": 1024})
    assert translator.n_ctx == 4096
    assert translator.max_tokens == 1024


def test_failed_unload_preserves_translation_and_context_error(caplog):
    translator, fake = _translator_with(['{"translations": [{"i": 0, "translated": "hola"}]}'])
    fake.generate.side_effect = ConnectionError("server restarted")
    with translator:
        result = translator.translate_segments([_seg("hello")], "es", "en")
    assert result[0].translated_text == "hola"
    assert translator._client._client is None
    assert "server memory may remain allocated" in caplog.text
    translator.unload()
    fake.generate.assert_called_once_with(model="m", keep_alive=0)

    translator._client._client = fake
    with pytest.raises(ValueError, match="original failure"), translator:
        raise ValueError("original failure")

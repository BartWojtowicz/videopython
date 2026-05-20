"""Unit tests for Qwen3Translator. No real model weights load — the
``llama_cpp.Llama`` constructor is patched to return a fake LLM that
returns canned JSON-line output."""

from __future__ import annotations

from typing import Any

import pytest

from videopython.ai.generation.qwen3 import (
    Qwen3Translator,
    _build_system_prompt,
    _build_user_prompt,
    _chunk_segment_indices,
    _parse_jsonl_response,
    _target_chars_for,
)
from videopython.base.transcription import TranscriptionSegment, TranscriptionWord


def _seg(
    start: float,
    end: float,
    text: str,
    *,
    speaker: str | None = None,
    avg_logprob: float | None = None,
) -> TranscriptionSegment:
    return TranscriptionSegment(
        start=start,
        end=end,
        text=text,
        words=[TranscriptionWord(start=start, end=end, word=text)],
        speaker=speaker,
        avg_logprob=avg_logprob,
    )


class _FakeLlama:
    """Stand-in for ``llama_cpp.Llama`` that yields scripted responses.

    Constructed with a list of raw text outputs; each ``__call__`` pops
    the next one. Records the prompt argument for inspection.
    """

    def __init__(self, scripted_outputs: list[str]) -> None:
        self._outputs = list(scripted_outputs)
        self.prompts_seen: list[str] = []

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        self.prompts_seen.append(prompt)
        if not self._outputs:
            raise AssertionError("FakeLlama got an unexpected extra call")
        text = self._outputs.pop(0)
        return {"choices": [{"text": text}]}


def _patch_qwen_with_fake(monkeypatch: pytest.MonkeyPatch, fake: _FakeLlama) -> None:
    """Replace _init_local so it installs the fake LLM without downloads."""

    def fake_init_local(self: Qwen3Translator) -> None:
        self._llm = fake

    monkeypatch.setattr(Qwen3Translator, "_init_local", fake_init_local)


class TestPromptBuilders:
    """Pure builders, no model. Cheap to test exhaustively."""

    def test_system_prompt_names_languages(self) -> None:
        prompt = _build_system_prompt("ja", "es")
        assert "Japanese" in prompt
        assert "Spanish" in prompt
        assert "JSON" in prompt

    def test_system_prompt_unknown_language_falls_back_to_code(self) -> None:
        prompt = _build_system_prompt("xx", "yy")
        # Unknown codes show up verbatim — not ideal but doesn't crash.
        assert "xx" in prompt and "yy" in prompt

    def test_user_prompt_includes_target_chars(self) -> None:
        seg = _seg(0.0, 10.0, "hello world")
        prompt = _build_user_prompt([seg], "es")
        assert '"target_chars"' in prompt
        # 10s * 14 chars/sec * 1.15 ≈ 161
        assert '"target_chars": 161' in prompt

    def test_user_prompt_marks_low_confidence(self) -> None:
        # Below the -1.0 hint threshold.
        seg = _seg(0.0, 10.0, "questionable", avg_logprob=-2.0)
        prompt = _build_user_prompt([seg], "es")
        assert '"low_confidence": true' in prompt

    def test_user_prompt_omits_low_confidence_when_healthy(self) -> None:
        seg = _seg(0.0, 10.0, "clean", avg_logprob=-0.3)
        prompt = _build_user_prompt([seg], "es")
        assert "low_confidence" not in prompt

    def test_user_prompt_omits_low_confidence_when_none(self) -> None:
        seg = _seg(0.0, 10.0, "no logprob")
        prompt = _build_user_prompt([seg], "es")
        assert "low_confidence" not in prompt

    def test_user_prompt_announces_segment_count(self) -> None:
        segs = [_seg(0, 1, "a"), _seg(1, 2, "b"), _seg(2, 3, "c")]
        prompt = _build_user_prompt(segs, "es")
        assert "exactly 3 lines" in prompt

    def test_target_chars_uses_language_rate(self) -> None:
        # Spanish ~14 c/s, Japanese ~8 c/s, default ~12 c/s.
        assert _target_chars_for(10.0, "es") == int(10.0 * 14.0 * 1.15)
        assert _target_chars_for(10.0, "ja") == int(10.0 * 8.0 * 1.15)
        assert _target_chars_for(10.0, "xx") == int(10.0 * 12.0 * 1.15)

    def test_target_chars_minimum_one(self) -> None:
        # Zero-duration shouldn't yield zero.
        assert _target_chars_for(0.0, "es") == 1


class TestJsonlParser:
    """``_parse_jsonl_response`` is permissive — must tolerate fences and
    preamble Qwen sometimes adds."""

    def test_parses_clean_lines(self) -> None:
        raw = '{"i": 0, "translated": "Hola"}\n{"i": 1, "translated": "Mundo"}'
        assert _parse_jsonl_response(raw) == {0: "Hola", 1: "Mundo"}

    def test_skips_markdown_fences(self) -> None:
        raw = '```json\n{"i": 0, "translated": "Hola"}\n```'
        assert _parse_jsonl_response(raw) == {0: "Hola"}

    def test_skips_garbage_lines(self) -> None:
        raw = 'Aquí están las traducciones:\n{"i": 0, "translated": "Hola"}\nEsa fue la última.\n'
        assert _parse_jsonl_response(raw) == {0: "Hola"}

    def test_returns_empty_on_pure_garbage(self) -> None:
        raw = "I cannot translate this content."
        assert _parse_jsonl_response(raw) == {}

    def test_skips_objects_missing_required_keys(self) -> None:
        raw = '{"i": 0}\n{"translated": "x"}\n{"i": 1, "translated": "OK"}'
        assert _parse_jsonl_response(raw) == {1: "OK"}


class TestQwenTranslateSegments:
    """Integration of prompt → fake LLM → parse → output. No real model."""

    def test_happy_path_one_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        segs = [_seg(0, 5, "alpha"), _seg(5, 10, "beta")]
        fake = _FakeLlama(['{"i": 0, "translated": "uno"}\n{"i": 1, "translated": "dos"}'])
        _patch_qwen_with_fake(monkeypatch, fake)

        translator = Qwen3Translator(device="cpu")
        result = translator.translate_segments(segs, target_lang="es", source_lang="en")

        assert [r.translated_text for r in result] == ["uno", "dos"]
        assert translator.translation_failures == []
        # Only one Qwen call (no retry needed).
        assert len(fake.prompts_seen) == 1

    def test_progress_callback_fires_three_times(self, monkeypatch: pytest.MonkeyPatch) -> None:
        segs = [_seg(0, 5, "alpha")]
        fake = _FakeLlama(['{"i": 0, "translated": "a"}'])
        _patch_qwen_with_fake(monkeypatch, fake)

        translator = Qwen3Translator(device="cpu")
        ticks: list[float] = []
        translator.translate_segments(segs, target_lang="es", source_lang="en", progress_callback=ticks.append)

        assert ticks == [0.5, 0.9, 1.0]

    def test_punctuation_only_segments_not_sent_to_qwen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        segs = [
            _seg(0, 5, "real text"),
            _seg(5, 6, "..."),
            _seg(6, 10, "more text"),
        ]
        fake = _FakeLlama(['{"i": 0, "translated": "uno"}\n{"i": 1, "translated": "dos"}'])
        _patch_qwen_with_fake(monkeypatch, fake)

        translator = Qwen3Translator(device="cpu")
        result = translator.translate_segments(segs, target_lang="es", source_lang="en")

        # The "..." segment gets translated_text="" without hitting Qwen.
        assert result[0].translated_text == "uno"
        assert result[1].translated_text == ""
        assert result[2].translated_text == "dos"
        # The prompt only included two real segments.
        prompt = fake.prompts_seen[0]
        assert "real text" in prompt
        assert "more text" in prompt
        assert "..." not in prompt


class TestRetryAndFallback:
    """Parse retry → Marian fallback → translation_failures."""

    def test_retry_recovers_missing_segments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        segs = [_seg(0, 5, "alpha"), _seg(5, 10, "beta")]
        # First call: only segment 0 came back. Second call (retry) covers
        # the missing one. Note retry uses local index 0 for the missing seg.
        fake = _FakeLlama(
            [
                '{"i": 0, "translated": "uno"}',
                '{"i": 0, "translated": "dos-retry"}',
            ]
        )
        _patch_qwen_with_fake(monkeypatch, fake)

        translator = Qwen3Translator(device="cpu")
        result = translator.translate_segments(segs, target_lang="es", source_lang="en")

        assert [r.translated_text for r in result] == ["uno", "dos-retry"]
        assert translator.translation_failures == []
        assert len(fake.prompts_seen) == 2

    def test_marian_fallback_recovers_segments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        segs = [_seg(0, 5, "alpha"), _seg(5, 10, "beta")]
        # Both Qwen calls fail to return segment 1.
        fake = _FakeLlama(
            [
                '{"i": 0, "translated": "uno"}',
                "garbage",
            ]
        )
        _patch_qwen_with_fake(monkeypatch, fake)

        # Stub Marian so it returns translated text without loading models.
        from videopython.ai.dubbing.models import TranslatedSegment
        from videopython.ai.generation.translation import MarianTranslator

        marian_calls: list[list[str]] = []

        def fake_marian_translate(
            self: MarianTranslator,
            segments: list[TranscriptionSegment],
            target_lang: str,
            source_lang: str | None = None,
            progress_callback: Any = None,
        ) -> list[TranslatedSegment]:
            marian_calls.append([s.text for s in segments])
            return [
                TranslatedSegment(
                    original_segment=s,
                    translated_text=f"marian-{s.text}",
                    source_lang=source_lang or "en",
                    target_lang=target_lang,
                    speaker=s.speaker,
                    start=s.start,
                    end=s.end,
                )
                for s in segments
            ]

        monkeypatch.setattr(MarianTranslator, "translate_segments", fake_marian_translate)

        translator = Qwen3Translator(device="cpu")
        result = translator.translate_segments(segs, target_lang="es", source_lang="en")

        assert [r.translated_text for r in result] == ["uno", "marian-beta"]
        assert translator.translation_failures == []
        # Marian was called once with just the missing segment.
        assert marian_calls == [["beta"]]

    def test_failures_recorded_when_marian_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        segs = [_seg(0, 5, "alpha"), _seg(5, 10, "beta")]
        fake = _FakeLlama(
            [
                '{"i": 0, "translated": "uno"}',
                "garbage",
            ]
        )
        _patch_qwen_with_fake(monkeypatch, fake)

        translator = Qwen3Translator(device="cpu", marian_fallback=False)
        result = translator.translate_segments(segs, target_lang="es", source_lang="en")

        # Failed segment gets empty translation; index recorded.
        assert result[0].translated_text == "uno"
        assert result[1].translated_text == ""
        assert translator.translation_failures == [1]

    def test_failures_recorded_when_marian_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        segs = [_seg(0, 5, "alpha"), _seg(5, 10, "beta")]
        fake = _FakeLlama(
            [
                '{"i": 0, "translated": "uno"}',
                "garbage",
            ]
        )
        _patch_qwen_with_fake(monkeypatch, fake)

        from videopython.ai.generation.translation import MarianTranslator

        def fake_marian_raises(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("marian model unavailable")

        monkeypatch.setattr(MarianTranslator, "translate_segments", fake_marian_raises)

        translator = Qwen3Translator(device="cpu", marian_fallback=True)
        result = translator.translate_segments(segs, target_lang="es", source_lang="en")

        assert result[1].translated_text == ""
        assert translator.translation_failures == [1]


class TestUnloadAndProtocol:
    def test_unload_clears_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        segs = [_seg(0, 5, "alpha")]
        fake = _FakeLlama(['{"i": 0, "translated": "a"}'])
        _patch_qwen_with_fake(monkeypatch, fake)

        translator = Qwen3Translator(device="cpu")
        translator.translate_segments(segs, target_lang="es", source_lang="en")
        assert translator._llm is fake

        translator.unload()
        assert translator._llm is None
        assert translator._marian is None

    def test_satisfies_translation_backend_protocol(self) -> None:
        from videopython.ai.generation.translation import TranslationBackend

        translator = Qwen3Translator(device="cpu")
        assert isinstance(translator, TranslationBackend)

    def test_supports_static_check(self) -> None:
        assert Qwen3Translator.supports("en", "es") is True
        assert Qwen3Translator.supports("en", "en") is True
        assert Qwen3Translator.supports("en", "xx") is False

    def test_get_supported_languages_returns_dict(self) -> None:
        langs = Qwen3Translator.get_supported_languages()
        assert isinstance(langs, dict)
        assert "en" in langs and "es" in langs


class TestChunkSegmentIndices:
    """Pure helper — exercises the char-budget splitter without any LLM."""

    def test_small_input_one_chunk(self) -> None:
        segs = [_seg(0, 1, "a"), _seg(1, 2, "b"), _seg(2, 3, "c")]
        chunks = _chunk_segment_indices(segs, n_ctx=8192, max_tokens=4096)
        assert chunks == [[0, 1, 2]]

    def test_large_input_splits_into_multiple_chunks(self) -> None:
        # 80 segments × ~300 chars each = ~24000 chars. With n_ctx=8192 and
        # max_tokens=4096 the prompt budget is ~3796 tokens ≈ 7592 chars, so
        # the result must be more than one chunk.
        segs = [_seg(i, i + 1, "x" * 300) for i in range(80)]
        chunks = _chunk_segment_indices(segs, n_ctx=8192, max_tokens=4096)
        assert len(chunks) >= 2
        # Every position covered exactly once, in order.
        flattened = [i for chunk in chunks for i in chunk]
        assert flattened == list(range(80))

    def test_oversized_single_segment_isolated(self) -> None:
        # A segment that on its own exceeds the budget must still appear,
        # in its own chunk — we'd rather surface llama.cpp's own error on
        # that one segment than drop it silently.
        big = _seg(0, 10, "y" * 50_000)
        small = _seg(10, 11, "ok")
        chunks = _chunk_segment_indices([big, small], n_ctx=8192, max_tokens=4096)
        assert chunks[0] == [0]
        assert [1] in chunks

    def test_pathological_n_ctx_falls_back_to_per_segment(self) -> None:
        # If max_tokens leaves no room, the helper degrades to one-per-chunk
        # instead of returning an empty list (which would silently drop work).
        segs = [_seg(0, 1, "a"), _seg(1, 2, "b")]
        chunks = _chunk_segment_indices(segs, n_ctx=1024, max_tokens=4096)
        assert chunks == [[0], [1]]

    def test_empty_input_returns_empty(self) -> None:
        assert _chunk_segment_indices([], n_ctx=8192, max_tokens=4096) == []


class TestChunkedTranslation:
    """End-to-end behaviour when input spans multiple Qwen calls."""

    def test_multi_chunk_first_pass_maps_indices_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 80 long segments → multiple chunks. Each fake call returns a
        # response that uses the chunk-local indices (0..chunk_size-1).
        segs = [_seg(i, i + 1, "x" * 300) for i in range(80)]
        chunks = _chunk_segment_indices(segs, n_ctx=8192, max_tokens=4096)
        assert len(chunks) >= 2  # sanity for the test premise

        # For each chunk, build a response that translates seg-i to
        # "tr-<original_position>" so we can verify index mapping.
        scripted: list[str] = []
        for chunk_positions in chunks:
            lines = [f'{{"i": {local}, "translated": "tr-{orig}"}}' for local, orig in enumerate(chunk_positions)]
            scripted.append("\n".join(lines))
        fake = _FakeLlama(scripted)
        _patch_qwen_with_fake(monkeypatch, fake)

        translator = Qwen3Translator(device="cpu")
        result = translator.translate_segments(segs, target_lang="es", source_lang="en")

        assert [r.translated_text for r in result] == [f"tr-{i}" for i in range(80)]
        assert translator.translation_failures == []
        assert len(fake.prompts_seen) == len(chunks)

    def test_retry_spans_chunks_when_many_segments_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # First pass: each chunk returns nothing parseable. Retry pass:
        # the missing set is the whole input and must itself be chunked.
        segs = [_seg(i, i + 1, "x" * 300) for i in range(80)]
        chunks = _chunk_segment_indices(segs, n_ctx=8192, max_tokens=4096)

        scripted: list[str] = []
        # First pass: garbage for every chunk.
        scripted.extend(["garbage"] * len(chunks))
        # Retry pass: same chunking is applied, return clean responses.
        for chunk_positions in chunks:
            lines = [f'{{"i": {local}, "translated": "rt-{orig}"}}' for local, orig in enumerate(chunk_positions)]
            scripted.append("\n".join(lines))
        fake = _FakeLlama(scripted)
        _patch_qwen_with_fake(monkeypatch, fake)

        translator = Qwen3Translator(device="cpu", marian_fallback=False)
        result = translator.translate_segments(segs, target_lang="es", source_lang="en")

        assert [r.translated_text for r in result] == [f"rt-{i}" for i in range(80)]
        assert translator.translation_failures == []
        # First-pass chunks + retry-pass chunks = 2× len(chunks).
        assert len(fake.prompts_seen) == 2 * len(chunks)

    def test_progress_callback_ramps_across_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 3 chunks in the first pass → 3 ticks ramping to 0.5, then 0.9, 1.0.
        segs = [_seg(i, i + 1, "x" * 300) for i in range(80)]
        chunks = _chunk_segment_indices(segs, n_ctx=8192, max_tokens=4096)
        scripted = []
        for chunk_positions in chunks:
            lines = [f'{{"i": {local}, "translated": "ok"}}' for local in range(len(chunk_positions))]
            scripted.append("\n".join(lines))
        fake = _FakeLlama(scripted)
        _patch_qwen_with_fake(monkeypatch, fake)

        translator = Qwen3Translator(device="cpu")
        ticks: list[float] = []
        translator.translate_segments(segs, target_lang="es", source_lang="en", progress_callback=ticks.append)

        # One ramp tick per first-pass chunk, then 0.9 and 1.0.
        assert len(ticks) == len(chunks) + 2
        assert ticks[-2:] == [0.9, 1.0]
        # First ramp ticks land between 0 and 0.5 inclusive, strictly increasing.
        ramp = ticks[: len(chunks)]
        assert ramp[-1] == pytest.approx(0.5)
        assert all(0 < t <= 0.5 for t in ramp)
        assert ramp == sorted(ramp)

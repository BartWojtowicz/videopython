import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from videopython.ai._text_chunks import split_text
from videopython.ai.generation.audio import TextToSpeech


class _Tensor(np.ndarray):
    """Only the tensor operations used at the backend boundary."""

    def cpu(self):
        return self

    def float(self):
        return self.astype(np.float32)

    def numpy(self):
        return np.asarray(self)

    def numel(self):
        return self.size


torch = SimpleNamespace(
    tensor=lambda value: np.asarray(value).view(_Tensor),
    ones=lambda *shape: np.ones(shape).view(_Tensor),
    zeros=lambda *shape: np.zeros(shape).view(_Tensor),
)


@pytest.fixture(autouse=True)
def speech_tensor_runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", torch)


@pytest.mark.parametrize(
    "text",
    [
        "First sentence. " * 90,
        "A long clause, and more words; " * 90,
        "word " * 300,
        "  Some\n whitespace.\tMore  words! ",
    ],
)
def test_split_covers_text_once(text):
    parts = split_text(text, 80)
    assert " ".join(parts) == " ".join(text.split())
    assert all(len(part) <= 80 for part in parts)


def test_punctuation_at_budget_edge_cannot_exceed_limit():
    text = "word " * 40 + ". More speech follows."
    parts = split_text(text, 200)
    assert all(len(part) <= 200 for part in parts)
    assert " ".join(parts) == text


@pytest.mark.parametrize("text", ["漢字" * 140, "ภาษาไทย" * 40, "https://example.com/" + "a" * 240])
def test_unbroken_text_is_bounded_without_dropping_characters(text):
    parts = split_text(text, 80)
    assert "".join(parts) == text
    assert all(0 < len(part) <= 80 for part in parts)


def test_unbroken_speech_reaches_backend_in_bounded_parts():
    tts = TextToSpeech(language="zh")
    tts._model = Mock()
    tts._model.generate.return_value = torch.ones(1, 240)
    text = "漢字" * 140
    tts.generate_audio(text)
    calls = tts._model.generate.call_args_list
    assert "".join(c.kwargs["text"] for c in calls) == text
    assert all(len(c.kwargs["text"]) <= 200 for c in calls)


def test_long_speech_preserves_settings_and_order():
    tts = TextToSpeech(language="pl")
    tts._model = Mock()
    tts._model.generate.side_effect = lambda **kw: torch.ones(1, 240) * len(kw["text"])
    text = "Pierwsze zdanie. Drugie zdanie, z dalszym wyjaśnieniem. " * 20
    audio = tts.generate_audio(text, voice_sample_path="speaker.wav", exaggeration=0.7, cfg_weight=0.4, temperature=0.6)
    calls = tts._model.generate.call_args_list
    assert " ".join(c.kwargs["text"] for c in calls) == text.strip()
    assert len(calls) > 1
    for i, call in enumerate(calls):
        assert call.kwargs["audio_prompt_path"] == "speaker.wav"
        assert call.kwargs["language_id"] == "pl"
        assert call.kwargs["exaggeration"] == 0.7
        assert call.kwargs["cfg_weight"] == 0.4
        assert call.kwargs["temperature"] == 0.6
        np.testing.assert_array_equal(audio.data[i * 240 : (i + 1) * 240], len(call.kwargs["text"]))


def test_cap_suspect_audio_is_replaced_with_smaller_calls():
    tts = TextToSpeech(language="en")
    tts._model = Mock()
    tts._model.generate.side_effect = [torch.zeros(1, 40 * 24000), torch.ones(1, 240), torch.ones(1, 240) * 2]
    text = "hello " * 30
    audio = tts.generate_audio(text)
    calls = tts._model.generate.call_args_list
    assert " ".join(c.kwargs["text"] for c in calls[1:]) == text.strip()
    assert len(audio.data) == 480
    assert np.all(audio.data[:240] == 1)
    assert np.all(audio.data[240:] == 2)


def test_subchunk_failure_does_not_return_partial_audio():
    tts = TextToSpeech()
    tts._model = Mock()
    tts._model.generate.side_effect = [torch.ones(1, 240), RuntimeError("failed")]
    with pytest.raises(RuntimeError, match="failed"):
        tts.generate_audio("sentence with words. " * 30)


def test_short_text_uses_one_unchanged_call():
    tts = TextToSpeech()
    tts._model = Mock()
    tts._model.generate.return_value = torch.ones(1, 240)
    tts.generate_audio("Hello, world!")
    tts._model.generate.assert_called_once_with(text="Hello, world!", language_id="en", audio_prompt_path=None)


def test_tiny_groups_respect_speaker_changes_and_gaps():
    from videopython.ai.dubbing._synthesis import synthesis_groups
    from videopython.ai.dubbing.models import TranslatedSegment
    from videopython.base.transcription import TranscriptionSegment

    def seg(start, end, text, speaker):
        original = TranscriptionSegment(start=start, end=end, text=text, words=[], speaker=speaker)
        return TranslatedSegment(original_segment=original, translated_text=text, source_lang="en", target_lang="pl")

    segments = [
        seg(0, 0.2, "I", "A"),
        seg(0.2, 3, "dalsza część zdania", "A"),
        seg(3, 3.2, "I", "B"),
        seg(4, 4.2, "I", "B"),
        seg(4.2, 6, "kolejne zdanie", "B"),
    ]
    assert synthesis_groups(segments) == [[0, 1], [2], [3, 4]]
    assert segments[0].translated_text == "I"


def test_chained_tiny_groups_have_bounded_span_and_members():
    from videopython.ai.dubbing._synthesis import synthesis_groups
    from videopython.ai.dubbing.models import TranslatedSegment
    from videopython.base.transcription import TranscriptionSegment

    segments = [
        TranslatedSegment(
            original_segment=TranscriptionSegment(start=i * 0.9, end=(i + 1) * 0.9, text="a", words=[], speaker="A"),
            translated_text="a",
            source_lang="en",
            target_lang="pl",
        )
        for i in range(60)
    ]
    groups = synthesis_groups(segments)
    assert [i for group in groups for i in group] == list(range(60))
    assert max(map(len, groups)) <= 4
    assert all(segments[g[-1]].end - segments[g[0]].start <= 10 for g in groups)
    segments[1].end = 30
    assert synthesis_groups(segments)[0] == [0]


def test_invalid_speech_tokens_never_reach_vocoder():
    from videopython.ai.generation._speech_tokens import InvalidSpeechTokens, guard_speech_tokens

    vocoder = Mock(return_value="audio")
    guarded = guard_speech_tokens(vocoder, 6561)
    for tokens in (torch.tensor([6727]), torch.tensor([-1]), torch.tensor([])):
        with pytest.raises(InvalidSpeechTokens):
            guarded(speech_tokens=tokens)
    vocoder.assert_not_called()
    valid = torch.tensor([0, 6560])
    assert guarded(speech_tokens=valid) == "audio"
    vocoder.assert_called_once_with(speech_tokens=valid)


def test_invalid_token_retry_is_bounded_and_keeps_settings():
    from videopython.ai.generation._speech_tokens import InvalidSpeechTokens

    tts = TextToSpeech(language="pl")
    tts._model = Mock()
    tts._model.generate.side_effect = [InvalidSpeechTokens("bad"), torch.ones(1, 240)]
    tts.generate_audio("za", voice_sample_path="speaker.wav", exaggeration=0.7)
    assert tts._model.generate.call_args_list[0] == tts._model.generate.call_args_list[1]
    tts._model.generate.reset_mock()
    tts._model.generate.side_effect = InvalidSpeechTokens("bad")
    with pytest.raises(InvalidSpeechTokens):
        tts.generate_audio("za")
    assert tts._model.generate.call_count == 3


def test_sampling_excludes_invalid_ids_but_preserves_eos_and_valid_logits():
    from videopython.ai.generation._speech_tokens import restrict_speech_vocabulary

    head = Mock(out_features=10)
    original = torch.tensor([[0.1 * i for i in range(10)]])
    before = original.copy()
    restrict_speech_vocabulary(head, 3, 4)
    hook = head.register_forward_hook.call_args.args[0]
    masked = hook(head, (), original)
    assert masked is original
    np.testing.assert_array_equal(masked[:, :3], before[:, :3])
    np.testing.assert_array_equal(masked[:, 4], before[:, 4])
    assert (masked[:, 3] == -(2**15)).all()
    assert (masked[:, 5:] == -(2**15)).all()
    assert np.isfinite(masked + 0.5 * (masked - masked)).all()

import pytest

from videopython.ai.dubbing._phrases import timed_phrases
from videopython.base.transcription import TranscriptionSegment, TranscriptionWord


def make_segment(text, start=0, speaker="A"):
    words = [
        TranscriptionWord(word=w, start=start + i * 0.5, end=start + (i + 1) * 0.5, speaker=speaker)
        for i, w in enumerate(text.split())
    ]
    return TranscriptionSegment.from_words(words, speaker=speaker, avg_logprob=-0.2)


def test_sentences_win_over_commas_and_preserve_all_words():
    source = make_segment("Well, this is our first sentence. Now this is our second sentence.")
    before = source.model_dump()
    phrases, parents = timed_phrases([source])
    assert [s.text for s in phrases] == ["Well, this is our first sentence.", "Now this is our second sentence."]
    assert parents == [0, 0]
    assert [w for s in phrases for w in s.words] == source.words
    assert [(s.start, s.end) for s in phrases] == [(0, 3), (3, 6)]
    assert all(s.speaker == "A" and s.avg_logprob == -0.2 for s in phrases)
    assert source.model_dump() == before


def test_long_speaker_turn_has_multiple_real_timing_anchors():
    source = make_segment(" ".join(["one two three four five six."] * 10))
    phrases, parents = timed_phrases([source])
    assert len(phrases) == 10
    assert parents == [0] * 10
    assert phrases[-1].start == 27
    assert phrases[-1].end == source.end
    assert all(s.end - s.start <= 8 for s in phrases)


def test_partial_or_invalid_word_alignment_keeps_original_text():
    source = make_segment("These are the full words. And these are more words.")
    for changes in ({"words": source.words[:-1]}, {"words": []}, {"start": 2}, {"text": source.text + " missing"}):
        segment = source.model_copy(update=changes)
        phrases, parents = timed_phrases([segment])
        assert phrases == [segment]
        assert parents == [0]


def test_does_not_create_tiny_tail_or_mix_speakers():
    a = make_segment("This is a full sentence. Yes.")
    b = make_segment("Here is another complete sentence. Here is one more sentence.", start=10, speaker="B")
    phrases, parents = timed_phrases([a, b])
    assert phrases[0] is a
    assert parents == [0, 1, 1]
    assert [s.speaker for s in phrases] == ["A", "B", "B"]


def test_translation_failures_map_to_source_turns():
    from unittest.mock import Mock

    from videopython.ai.dubbing.config import DubbingConfig
    from videopython.ai.dubbing.models import TranslatedSegment
    from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
    from videopython.base.transcription import Transcription

    source = make_segment("This is the first sentence. Here is the second sentence.")
    pipeline = LocalDubbingPipeline(DubbingConfig())
    translator = Mock()
    translator.translation_failures = [1]
    translator.translate_segments.side_effect = lambda **kw: [
        TranslatedSegment(original_segment=s, translated_text="hello", source_lang="en", target_lang="pl")
        for s in kw["segments"]
    ]
    pipeline._translator = translator
    translated, failures = pipeline._translate(Transcription(segments=[source]), "en", "pl", lambda *args: None)
    assert failures == [0]
    assert len(translated) == 2
    assert [s.source_segment_index for s in translated] == [0, 0]
    assert source.text == "This is the first sentence. Here is the second sentence."


def test_common_title_does_not_split_a_name():
    source = make_segment("Here we meet Dr. Smith who is speaking. This is the next sentence.")
    phrases, _ = timed_phrases([source])
    assert phrases[0].text == "Here we meet Dr. Smith who is speaking."


@pytest.mark.parametrize(
    "tokens, first, second",
    [
        (["私", "は", "今日", "走る。", "明日", "も", "外を", "走る。"], "私は今日走る。", "明日も外を走る。"),
        (["我", "今天", "在", "工作。", "明天", "我", "也要", "工作。"], "我今天在工作。", "明天我也要工作。"),
        (["ราคา", "นี้", "คือ", "3,000.", "ราคา", "นั้น", "คือ", "5,000."], "ราคานี้คือ3,000.", "ราคานั้นคือ5,000."),
        (["It", "costs", "3", ",000.", "That", "costs", "5", ",000."], "It  costs 3,000.", "That costs 5,000."),
    ],
)
def test_phrase_text_preserves_source_spacing(tokens, first, second):
    words = [TranscriptionWord(word=w, start=i * 0.5, end=(i + 1) * 0.5) for i, w in enumerate(tokens)]
    source = TranscriptionSegment(start=0, end=4, text=first + "\n" + second, words=words)
    phrases, parents = timed_phrases([source])
    assert [phrase.text for phrase in phrases] == [first, second]
    assert [word for phrase in phrases for word in phrase.words] == words
    assert [(phrase.start, phrase.end) for phrase in phrases] == [(0, 2), (2, 4)]
    assert parents == [0, 0]

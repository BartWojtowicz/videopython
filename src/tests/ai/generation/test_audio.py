import pytest
from soundpython import Audio

from videopython.ai.generation.audio import SUPPORTED_TTS_LANGUAGES, TextToSpeech


class TestTextToSpeech:
    def test_default_language_is_english(self):
        tts = TextToSpeech()
        assert tts.language == "eng"

    def test_german_language_initialization(self):
        tts = TextToSpeech(language="deu")
        assert tts.language == "deu"

    def test_polish_language_initialization(self):
        tts = TextToSpeech(language="pol")
        assert tts.language == "pol"

    def test_unsupported_language_raises_error(self):
        with pytest.raises(ValueError) as exc_info:
            TextToSpeech(language="xyz")
        assert "Language 'xyz' is not supported" in str(exc_info.value)
        assert "eng" in str(exc_info.value)

    def test_all_supported_languages_can_be_initialized(self):
        for lang_code in SUPPORTED_TTS_LANGUAGES.keys():
            tts = TextToSpeech(language=lang_code)
            assert tts.language == lang_code
            assert tts.pipeline is not None
            assert tts.tokenizer is not None


class TestTextToSpeechGeneration:
    def test_generate_audio_english(self):
        tts = TextToSpeech(language="eng")
        audio = tts.generate_audio("Hello world")

        assert isinstance(audio, Audio)
        assert audio.metadata.sample_rate > 0
        assert audio.metadata.channels == 1
        assert audio.metadata.duration_seconds > 0
        assert len(audio.data) > 0

    def test_generate_audio_german(self):
        tts = TextToSpeech(language="deu")
        audio = tts.generate_audio("Hallo Welt")

        assert isinstance(audio, Audio)
        assert audio.metadata.sample_rate > 0
        assert audio.metadata.channels == 1
        assert audio.metadata.duration_seconds > 0
        assert len(audio.data) > 0

    def test_generate_audio_polish(self):
        tts = TextToSpeech(language="pol")
        audio = tts.generate_audio("Witaj swiecie")

        assert isinstance(audio, Audio)
        assert audio.metadata.sample_rate > 0
        assert audio.metadata.channels == 1
        assert audio.metadata.duration_seconds > 0
        assert len(audio.data) > 0

    def test_different_texts_produce_different_audio(self):
        tts = TextToSpeech(language="eng")
        audio1 = tts.generate_audio("Hello")
        audio2 = tts.generate_audio("Goodbye")

        assert audio1.metadata.duration_seconds != audio2.metadata.duration_seconds
        assert len(audio1.data) != len(audio2.data)

import pytest
from soundpython import Audio

from videopython.ai.generation.audio import TextToSpeech


class TestTextToSpeech:
    def test_default_initialization(self):
        tts = TextToSpeech()
        assert tts.model is not None
        assert tts.processor is not None

    def test_device_parameter(self):
        tts = TextToSpeech(device="cpu")
        assert tts.device == "cpu"

    def test_model_size_base(self):
        tts = TextToSpeech(model_size="base")
        assert tts.model is not None

    def test_model_size_small(self):
        tts = TextToSpeech(model_size="small")
        assert tts.model is not None

    def test_invalid_model_size_raises_error(self):
        with pytest.raises(ValueError) as exc_info:
            TextToSpeech(model_size="invalid")
        assert "model_size must be 'base' or 'small'" in str(exc_info.value)


class TestTextToSpeechGeneration:
    def test_generate_audio_basic(self):
        tts = TextToSpeech(device="cpu", model_size="small")
        audio = tts.generate_audio("Hello world")

        assert isinstance(audio, Audio)
        assert audio.metadata.sample_rate == 24000
        assert audio.metadata.channels == 1
        assert audio.metadata.duration_seconds > 0
        assert len(audio.data) > 0

    def test_generate_audio_with_emotion(self):
        tts = TextToSpeech(device="cpu", model_size="small")
        audio = tts.generate_audio("That's funny! [laughs]")

        assert isinstance(audio, Audio)
        assert audio.metadata.sample_rate == 24000
        assert len(audio.data) > 0

    def test_generate_audio_multilingual(self):
        tts = TextToSpeech(device="cpu", model_size="small")

        # Test different languages
        audio_en = tts.generate_audio("Hello world")
        audio_de = tts.generate_audio("Hallo Welt")
        audio_pl = tts.generate_audio("Witaj swiecie")

        assert isinstance(audio_en, Audio)
        assert isinstance(audio_de, Audio)
        assert isinstance(audio_pl, Audio)

    def test_different_texts_produce_different_audio(self):
        tts = TextToSpeech(device="cpu", model_size="small")
        audio1 = tts.generate_audio("Hello")
        audio2 = tts.generate_audio("Goodbye")

        assert audio1.metadata.duration_seconds != audio2.metadata.duration_seconds
        assert len(audio1.data) != len(audio2.data)

    def test_generate_audio_with_voice_preset(self):
        tts = TextToSpeech(device="cpu", model_size="small")
        audio = tts.generate_audio("Hello world", voice_preset="v2/en_speaker_6")

        assert isinstance(audio, Audio)
        assert len(audio.data) > 0

    def test_generate_audio_without_sampling(self):
        tts = TextToSpeech(device="cpu", model_size="small")
        audio = tts.generate_audio("Hello world", do_sample=False)

        assert isinstance(audio, Audio)
        assert len(audio.data) > 0

    def test_audio_metadata(self):
        tts = TextToSpeech(device="cpu", model_size="small")
        audio = tts.generate_audio("Test")

        assert audio.metadata.sample_rate == 24000
        assert audio.metadata.channels == 1
        assert audio.metadata.sample_width == 2
        assert audio.metadata.frame_count > 0
        assert audio.metadata.duration_seconds > 0

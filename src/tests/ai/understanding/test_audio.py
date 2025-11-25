from videopython.ai.understanding.audio import AudioToText


class TestAudioToText:
    """Integration tests for AudioToText class with real transcription."""

    def test_initialization_without_diarization(self):
        """Test AudioToText initialization without diarization."""
        transcriber = AudioToText(model_name="tiny", enable_diarization=False)
        assert transcriber.model_name == "tiny"
        assert transcriber.enable_diarization is False
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "float32"

    def test_initialization_with_custom_parameters(self):
        """Test AudioToText initialization with custom parameters."""
        transcriber = AudioToText(model_name="tiny", enable_diarization=False, device="cpu", compute_type="float32")
        assert transcriber.model_name == "tiny"
        assert transcriber.enable_diarization is False
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "float32"

    def test_initialization_with_diarization(self):
        """Test AudioToText initialization with diarization enabled."""
        transcriber = AudioToText(model_name="tiny", enable_diarization=True)

        assert transcriber.model_name == "tiny"
        assert transcriber.enable_diarization is True
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "float32"

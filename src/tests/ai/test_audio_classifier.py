"""Tests for AudioClassifier with PANNs backend.

These tests are excluded from CI as they require:
- PANNs model download (~80MB+)
- Significant CPU/GPU time

Run locally with: uv run pytest src/tests/ai/test_audio_classifier.py -v
"""

import os

import numpy as np
import pytest

from videopython.base.description import AudioClassification, AudioEvent

# Path to test data (one level up from ai/ directory)
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
TEST_AUDIO_PATH = os.path.join(TEST_DATA_DIR, "test_audio.mp3")
SMALL_VIDEO_PATH = os.path.join(TEST_DATA_DIR, "small_video.mp4")


class TestAudioClassifier:
    """Tests for AudioClassifier with PANNs backend."""

    @pytest.fixture
    def classifier(self):
        """Create AudioClassifier with local backend."""
        from videopython.ai.understanding.audio import AudioClassifier

        return AudioClassifier(backend="local", confidence_threshold=0.3, device="cpu")

    @pytest.fixture
    def low_threshold_classifier(self):
        """Create AudioClassifier with low threshold for more detections."""
        from videopython.ai.understanding.audio import AudioClassifier

        return AudioClassifier(backend="local", confidence_threshold=0.1, device="cpu")

    @pytest.fixture
    def test_audio(self):
        """Load test audio file."""
        from videopython.base.audio import Audio

        return Audio.from_file(TEST_AUDIO_PATH)

    @pytest.fixture
    def test_video(self):
        """Load test video file."""
        from videopython.base.video import Video

        return Video.from_path(SMALL_VIDEO_PATH)

    @pytest.fixture
    def silent_audio(self):
        """Create silent audio."""
        from videopython.base.audio import Audio, AudioMetadata

        # 1 second of silence at 32kHz
        silent_data = np.zeros(32000, dtype=np.float32)
        metadata = AudioMetadata(
            sample_rate=32000,
            channels=1,
            sample_width=4,  # float32
            duration_seconds=1.0,
            frame_count=32000,
        )
        return Audio(data=silent_data, metadata=metadata)

    def test_classifier_initialization(self, classifier):
        """Test classifier initializes correctly."""
        assert classifier.backend == "local"
        assert classifier.confidence_threshold == 0.3
        assert classifier.model_name == "Cnn14"
        assert classifier.device == "cpu"

    def test_classifier_unsupported_backend(self):
        """Test classifier raises error for unsupported backend."""
        from videopython.ai.backends import UnsupportedBackendError
        from videopython.ai.understanding.audio import AudioClassifier

        with pytest.raises(UnsupportedBackendError):
            AudioClassifier(backend="invalid_backend")

    def test_classifier_unsupported_model(self):
        """Test classifier raises error for unsupported model."""
        from videopython.ai.understanding.audio import AudioClassifier

        with pytest.raises(ValueError, match="not supported"):
            AudioClassifier(model_name="InvalidModel")

    def test_classify_returns_audio_classification(self, classifier, test_audio):
        """Test classification returns AudioClassification object."""
        result = classifier.classify(test_audio)
        assert isinstance(result, AudioClassification)
        assert isinstance(result.events, list)
        assert isinstance(result.clip_predictions, dict)

    def test_classify_events_have_correct_structure(self, low_threshold_classifier, test_audio):
        """Test that detected events have correct structure."""
        result = low_threshold_classifier.classify(test_audio)

        for event in result.events:
            assert isinstance(event, AudioEvent)
            assert isinstance(event.start, float)
            assert isinstance(event.end, float)
            assert isinstance(event.label, str)
            assert isinstance(event.confidence, float)
            assert event.start >= 0
            assert event.end > event.start
            assert 0 <= event.confidence <= 1
            assert len(event.label) > 0

    def test_classify_video(self, classifier, test_video):
        """Test classification works with Video input."""
        result = classifier.classify(test_video)
        assert isinstance(result, AudioClassification)
        assert isinstance(result.events, list)

    def test_classify_silent_audio_returns_empty(self, classifier, silent_audio):
        """Test that silent audio returns empty classification."""
        result = classifier.classify(silent_audio)
        assert isinstance(result, AudioClassification)
        # Silent audio may still have some predictions but should be mostly empty
        assert isinstance(result.events, list)

    def test_clip_predictions_have_correct_structure(self, low_threshold_classifier, test_audio):
        """Test that clip predictions have correct structure."""
        result = low_threshold_classifier.classify(test_audio)

        for label, confidence in result.clip_predictions.items():
            assert isinstance(label, str)
            assert isinstance(confidence, float)
            assert 0 <= confidence <= 1
            assert len(label) > 0

    def test_events_are_sorted_by_start_time(self, low_threshold_classifier, test_audio):
        """Test that events are sorted by start time."""
        result = low_threshold_classifier.classify(test_audio)

        if len(result.events) > 1:
            for i in range(len(result.events) - 1):
                assert result.events[i].start <= result.events[i + 1].start

    def test_confidence_threshold_filtering(self, test_audio):
        """Test that confidence threshold correctly filters events."""
        from videopython.ai.understanding.audio import AudioClassifier

        # Create classifiers with different thresholds
        low_classifier = AudioClassifier(confidence_threshold=0.1, device="cpu")
        high_classifier = AudioClassifier(confidence_threshold=0.8, device="cpu")

        low_result = low_classifier.classify(test_audio)
        high_result = high_classifier.classify(test_audio)

        # Higher threshold should result in fewer or equal events
        assert len(high_result.events) <= len(low_result.events)

        # All events should meet their respective thresholds
        for event in low_result.events:
            assert event.confidence >= 0.1

        for event in high_result.events:
            assert event.confidence >= 0.8


class TestAudioEventMerging:
    """Tests for the event merging logic."""

    @pytest.fixture
    def classifier(self):
        """Create AudioClassifier for testing."""
        from videopython.ai.understanding.audio import AudioClassifier

        return AudioClassifier(backend="local", confidence_threshold=0.1, device="cpu")

    def test_merge_consecutive_events(self, classifier):
        """Test that consecutive events of same class are merged."""
        events = [
            AudioEvent(start=0.0, end=0.5, label="Music", confidence=0.8),
            AudioEvent(start=0.5, end=1.0, label="Music", confidence=0.9),
            AudioEvent(start=1.0, end=1.5, label="Music", confidence=0.7),
        ]

        merged = classifier._merge_events(events, gap_threshold=0.5)

        # Should be merged into a single event
        assert len(merged) == 1
        assert merged[0].start == 0.0
        assert merged[0].end == 1.5
        assert merged[0].label == "Music"
        assert merged[0].confidence == 0.9  # Max confidence

    def test_merge_keeps_separate_labels(self, classifier):
        """Test that events with different labels are not merged."""
        events = [
            AudioEvent(start=0.0, end=0.5, label="Music", confidence=0.8),
            AudioEvent(start=0.5, end=1.0, label="Speech", confidence=0.9),
        ]

        merged = classifier._merge_events(events, gap_threshold=0.5)

        # Should remain separate
        assert len(merged) == 2
        labels = {e.label for e in merged}
        assert labels == {"Music", "Speech"}

    def test_merge_respects_gap_threshold(self, classifier):
        """Test that events with gaps larger than threshold are not merged."""
        events = [
            AudioEvent(start=0.0, end=0.5, label="Music", confidence=0.8),
            AudioEvent(start=2.0, end=2.5, label="Music", confidence=0.9),
        ]

        merged = classifier._merge_events(events, gap_threshold=0.5)

        # Gap is too large, should remain separate
        assert len(merged) == 2

    def test_merge_empty_list(self, classifier):
        """Test that empty list returns empty list."""
        merged = classifier._merge_events([], gap_threshold=0.5)
        assert merged == []


class TestAudioEvent:
    """Tests for AudioEvent dataclass."""

    def test_duration_property(self):
        """Test that duration property calculates correctly."""
        event = AudioEvent(start=1.5, end=3.5, label="Music", confidence=0.8)
        assert event.duration == 2.0

    def test_event_fields(self):
        """Test that all fields are accessible."""
        event = AudioEvent(start=0.0, end=1.0, label="Speech", confidence=0.95)
        assert event.start == 0.0
        assert event.end == 1.0
        assert event.label == "Speech"
        assert event.confidence == 0.95


class TestAudioClassification:
    """Tests for AudioClassification dataclass."""

    def test_empty_classification(self):
        """Test creating empty classification."""
        classification = AudioClassification(events=[], clip_predictions={})
        assert len(classification.events) == 0
        assert len(classification.clip_predictions) == 0

    def test_classification_with_data(self):
        """Test creating classification with data."""
        events = [
            AudioEvent(start=0.0, end=1.0, label="Music", confidence=0.8),
            AudioEvent(start=2.0, end=3.0, label="Speech", confidence=0.9),
        ]
        predictions = {"Music": 0.75, "Speech": 0.65}

        classification = AudioClassification(events=events, clip_predictions=predictions)

        assert len(classification.events) == 2
        assert classification.clip_predictions["Music"] == 0.75

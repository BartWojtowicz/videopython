"""Tests for AudioClassifier with AST (Audio Spectrogram Transformer) backend.

Covers only what runs on a GitHub runner: construction and the pure event-merging
logic. Anything needing the AST weights is verified by the real-model harness
instead (see CLAUDE.md), not by a test that downloads a model.
"""

import pytest

from videopython.base.description import AudioClassification, AudioEvent


class TestAudioClassifierInit:
    """Lightweight tests for AudioClassifier initialization (no model download needed)."""

    def test_classifier_accepts_arbitrary_model_name(self):
        """model_name is no longer validated at construction (deferred to model load)."""
        from videopython.ai.understanding.classification import AudioClassifier

        assert AudioClassifier(model_name="some/other-ast-model").model_name == "some/other-ast-model"


class TestAudioEventMerging:
    """Pure logic over hand-built AudioEvents -- no model, no download.

    Previously carried ``requires_model_download`` and so never ran in CI, purely
    because the fixture constructs an ``AudioClassifier``. That construction is
    lazy (``_model = None`` until first use), so these always could have run: 4
    tests in 0.01s.
    """

    @pytest.fixture
    def classifier(self):
        """Create AudioClassifier for testing."""
        from videopython.ai.understanding.classification import AudioClassifier

        return AudioClassifier(confidence_threshold=0.1, device="cpu")

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

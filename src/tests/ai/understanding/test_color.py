import numpy as np
import pytest

from videopython.ai.understanding.color import ColorAnalyzer
from videopython.base.description import ColorHistogram


class TestColorAnalyzer:
    """Tests for ColorAnalyzer class."""

    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        # Create a 200x200 RGB frame with random colors
        return np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    @pytest.fixture
    def solid_color_frame(self):
        """Create a frame with a solid color."""
        # Create a red frame
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Red channel
        return frame

    def test_color_analyzer_initialization(self):
        """Test ColorAnalyzer initialization."""
        analyzer = ColorAnalyzer(num_dominant_colors=5)
        assert analyzer.num_dominant_colors == 5

    def test_extract_color_features_basic(self, sample_frame):
        """Test basic color feature extraction."""
        analyzer = ColorAnalyzer(num_dominant_colors=5)
        features = analyzer.extract_color_features(sample_frame)

        assert isinstance(features, ColorHistogram)
        assert len(features.dominant_colors) == 5
        assert 0 <= features.avg_hue <= 180
        assert 0 <= features.avg_saturation <= 255
        assert 0 <= features.avg_value <= 255
        assert features.hsv_histogram is None  # Not included by default

    def test_extract_color_features_with_histogram(self, sample_frame):
        """Test color feature extraction with full histogram."""
        analyzer = ColorAnalyzer(num_dominant_colors=5)
        features = analyzer.extract_color_features(sample_frame, include_full_histogram=True)

        assert isinstance(features, ColorHistogram)
        assert features.hsv_histogram is not None
        assert len(features.hsv_histogram) == 3  # H, S, V histograms

    def test_dominant_colors_format(self, sample_frame):
        """Test that dominant colors are in correct format."""
        analyzer = ColorAnalyzer(num_dominant_colors=3)
        features = analyzer.extract_color_features(sample_frame)

        assert len(features.dominant_colors) == 3
        for color in features.dominant_colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            for channel in color:
                assert 0 <= channel <= 255

    def test_solid_color_dominant(self, solid_color_frame):
        """Test that solid color frame has correct dominant color."""
        analyzer = ColorAnalyzer(num_dominant_colors=1)
        features = analyzer.extract_color_features(solid_color_frame)

        # The dominant color should be red (255, 0, 0) or very close
        dominant = features.dominant_colors[0]
        assert dominant[0] > 200  # Red channel
        assert dominant[1] < 50  # Green channel
        assert dominant[2] < 50  # Blue channel

    def test_calculate_histogram_difference_identical(self, sample_frame):
        """Test histogram difference between identical frames."""
        analyzer = ColorAnalyzer()
        diff = analyzer.calculate_histogram_difference(sample_frame, sample_frame)

        # Identical frames should have very low difference (close to 0)
        assert 0 <= diff <= 0.1

    def test_calculate_histogram_difference_different(self, sample_frame, solid_color_frame):
        """Test histogram difference between different frames."""
        analyzer = ColorAnalyzer()
        diff = analyzer.calculate_histogram_difference(sample_frame, solid_color_frame)

        # Different frames should have higher difference
        assert diff > 0.1

    def test_custom_num_dominant_colors(self, sample_frame):
        """Test custom number of dominant colors."""
        analyzer = ColorAnalyzer(num_dominant_colors=10)
        features = analyzer.extract_color_features(sample_frame)

        assert len(features.dominant_colors) == 10

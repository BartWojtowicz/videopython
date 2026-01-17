"""Tests for AI-powered transforms (face tracking)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from videopython.ai.transforms import (
    AutoFramingCrop,
    FaceTracker,
    FaceTrackingCrop,
    SplitScreenComposite,
)
from videopython.base.video import Video


class MockBoundingBox:
    """Mock bounding box for testing."""

    def __init__(self, center: tuple[float, float], width: float, height: float):
        self.center = center
        self.width = width
        self.height = height


class MockDetectedFace:
    """Mock detected face for testing."""

    def __init__(self, center: tuple[float, float], width: float, height: float):
        self.bounding_box = MockBoundingBox(center, width, height)


class TestFaceTracker:
    """Tests for FaceTracker utility."""

    def test_init_default_params(self):
        """Test default initialization."""
        tracker = FaceTracker()
        assert tracker.selection_strategy == "largest"
        assert tracker.face_index == 0
        assert tracker.smoothing == 0.8
        assert tracker.detection_interval == 3
        assert tracker.min_face_size == 30

    def test_init_custom_params(self):
        """Test custom initialization."""
        tracker = FaceTracker(
            selection_strategy="centered",
            face_index=1,
            smoothing=0.5,
            detection_interval=5,
            min_face_size=50,
        )
        assert tracker.selection_strategy == "centered"
        assert tracker.face_index == 1
        assert tracker.smoothing == 0.5
        assert tracker.detection_interval == 5
        assert tracker.min_face_size == 50

    def test_select_face_largest(self):
        """Test largest face selection strategy."""
        tracker = FaceTracker(selection_strategy="largest")

        # Create mock faces (already sorted by area, largest first)
        faces = [
            MockDetectedFace((0.5, 0.5), 0.3, 0.3),  # Largest
            MockDetectedFace((0.2, 0.2), 0.1, 0.1),  # Smaller
        ]

        result = tracker._select_face(faces, 1920, 1080)
        assert result is not None
        assert result[:2] == (0.5, 0.5)  # Center
        assert result[2:] == (0.3, 0.3)  # Width, height

    def test_select_face_centered(self):
        """Test centered face selection strategy."""
        tracker = FaceTracker(selection_strategy="centered")

        # Create mock faces with different positions
        faces = [
            MockDetectedFace((0.9, 0.9), 0.3, 0.3),  # Far from center
            MockDetectedFace((0.5, 0.5), 0.1, 0.1),  # Near center
        ]

        result = tracker._select_face(faces, 1920, 1080)
        assert result is not None
        # Should select the centered face
        assert result[:2] == (0.5, 0.5)

    def test_select_face_by_index(self):
        """Test face selection by index."""
        tracker = FaceTracker(selection_strategy="index", face_index=1)

        faces = [
            MockDetectedFace((0.5, 0.5), 0.3, 0.3),
            MockDetectedFace((0.2, 0.2), 0.1, 0.1),
        ]

        result = tracker._select_face(faces, 1920, 1080)
        assert result is not None
        assert result[:2] == (0.2, 0.2)  # Second face

    def test_select_face_index_out_of_bounds(self):
        """Test face selection with index out of bounds falls back to largest."""
        tracker = FaceTracker(selection_strategy="index", face_index=10)

        faces = [
            MockDetectedFace((0.5, 0.5), 0.3, 0.3),
            MockDetectedFace((0.2, 0.2), 0.1, 0.1),
        ]

        result = tracker._select_face(faces, 1920, 1080)
        assert result is not None
        assert result[:2] == (0.5, 0.5)  # Falls back to first (largest)

    def test_select_face_empty_list(self):
        """Test face selection with no faces."""
        tracker = FaceTracker()
        result = tracker._select_face([], 1920, 1080)
        assert result is None

    def test_reset_clears_state(self):
        """Test reset clears all tracking state."""
        tracker = FaceTracker()
        tracker._last_position = (0.5, 0.5)
        tracker._last_size = (0.1, 0.1)
        tracker._smoothed_position = (0.5, 0.5)
        tracker._smoothed_size = (0.1, 0.1)

        tracker.reset()

        assert tracker._last_position is None
        assert tracker._last_size is None
        assert tracker._smoothed_position is None
        assert tracker._smoothed_size is None

    @patch("videopython.ai.transforms.FaceTracker._init_detector")
    def test_detect_and_track_with_mock(self, mock_init):
        """Test detect_and_track with mocked detector."""
        tracker = FaceTracker(smoothing=0.0, detection_interval=1)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            MockDetectedFace((0.5, 0.5), 0.2, 0.2),
        ]
        tracker._detector = mock_detector

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = tracker.detect_and_track(frame, 0)

        assert result is not None
        assert result == (0.5, 0.5, 0.2, 0.2)

    @patch("videopython.ai.transforms.FaceTracker._init_detector")
    def test_detect_and_track_smoothing(self, mock_init):
        """Test smoothing over multiple frames."""
        tracker = FaceTracker(smoothing=0.5, detection_interval=1)

        mock_detector = MagicMock()
        tracker._detector = mock_detector

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # First frame - face at (0.3, 0.3)
        mock_detector.detect.return_value = [MockDetectedFace((0.3, 0.3), 0.1, 0.1)]
        result1 = tracker.detect_and_track(frame, 0)
        assert result1 == (0.3, 0.3, 0.1, 0.1)  # No smoothing on first frame

        # Second frame - face at (0.5, 0.5)
        mock_detector.detect.return_value = [MockDetectedFace((0.5, 0.5), 0.1, 0.1)]
        result2 = tracker.detect_and_track(frame, 1)

        # Should be smoothed: 0.3 * 0.5 + 0.5 * 0.5 = 0.4
        assert result2[0] == pytest.approx(0.4)
        assert result2[1] == pytest.approx(0.4)

    @patch("videopython.ai.transforms.FaceTracker._init_detector")
    def test_detection_interval_skips_frames(self, mock_init):
        """Test detection is only run on interval frames."""
        tracker = FaceTracker(smoothing=0.0, detection_interval=3)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [MockDetectedFace((0.5, 0.5), 0.1, 0.1)]
        tracker._detector = mock_detector

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Frame 0 - should detect
        tracker.detect_and_track(frame, 0)
        assert mock_detector.detect.call_count == 1

        # Frame 1 - should skip detection
        tracker.detect_and_track(frame, 1)
        assert mock_detector.detect.call_count == 1

        # Frame 2 - should skip detection
        tracker.detect_and_track(frame, 2)
        assert mock_detector.detect.call_count == 1

        # Frame 3 - should detect again
        tracker.detect_and_track(frame, 3)
        assert mock_detector.detect.call_count == 2


class TestFaceTrackingCrop:
    """Tests for FaceTrackingCrop transformation."""

    def test_init_default_params(self):
        """Test default initialization."""
        crop = FaceTrackingCrop()
        assert crop.target_aspect == (9, 16)
        assert crop.face_selection == "largest"
        assert crop.padding == 0.3
        assert crop.vertical_offset == -0.1
        assert crop.smoothing == 0.8
        assert crop.fallback == "last_position"

    def test_init_custom_params(self):
        """Test custom initialization."""
        crop = FaceTrackingCrop(
            target_aspect=(1, 1),
            face_selection="centered",
            padding=0.5,
            vertical_offset=0.0,
            smoothing=0.5,
            fallback="center",
        )
        assert crop.target_aspect == (1, 1)
        assert crop.face_selection == "centered"
        assert crop.padding == 0.5
        assert crop.vertical_offset == 0.0
        assert crop.smoothing == 0.5
        assert crop.fallback == "center"

    def test_calculate_crop_region_basic(self):
        """Test crop region calculation."""
        crop = FaceTrackingCrop(target_aspect=(9, 16), padding=0.3, vertical_offset=0.0)

        # Face at center of 1920x1080 frame
        x, y, w, h = crop._calculate_crop_region(
            face_cx=0.5,
            face_cy=0.5,
            face_w=0.1,
            face_h=0.15,
            frame_w=1920,
            frame_h=1080,
        )

        # For 9:16 from 1920x1080: crop_h = 1080, crop_w = 1080 * 9/16 = 607.5
        # Rounded down to even for H.264 compatibility
        assert w == 606
        assert h == 1080

    def test_calculate_crop_region_clamped_to_bounds(self):
        """Test crop region is clamped to frame bounds."""
        crop = FaceTrackingCrop(target_aspect=(9, 16), padding=0.3, vertical_offset=0.0)

        # Face at edge of frame
        x, y, w, h = crop._calculate_crop_region(
            face_cx=0.05,  # Near left edge
            face_cy=0.5,
            face_w=0.1,
            face_h=0.15,
            frame_w=1920,
            frame_h=1080,
        )

        # x should be clamped to 0
        assert x >= 0
        assert x + w <= 1920

    @patch("videopython.ai.transforms.FaceTracker")
    def test_apply_creates_correct_output_shape(self, mock_tracker_class):
        """Test apply produces correct output dimensions."""
        mock_tracker = MagicMock()
        mock_tracker.detect_and_track.return_value = (0.5, 0.5, 0.1, 0.15)
        mock_tracker_class.return_value = mock_tracker

        # Create test video 1920x1080, 10 frames
        frames = np.zeros((10, 1080, 1920, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=30)

        crop = FaceTrackingCrop(target_aspect=(9, 16))
        result = crop.apply(video)

        # Output should be 9:16 aspect ratio
        out_h, out_w = result.frame_shape[:2]
        assert abs(out_w / out_h - 9 / 16) < 0.01

    @patch("videopython.ai.transforms.FaceTracker")
    def test_apply_with_fallback_center(self, mock_tracker_class):
        """Test fallback to center when no face detected."""
        mock_tracker = MagicMock()
        mock_tracker.detect_and_track.return_value = None  # No face
        mock_tracker_class.return_value = mock_tracker

        frames = np.zeros((10, 1080, 1920, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=30)

        crop = FaceTrackingCrop(target_aspect=(9, 16), fallback="center")
        result = crop.apply(video)

        # Should still produce valid output
        assert len(result.frames) == 10


class TestSplitScreenComposite:
    """Tests for SplitScreenComposite transformation."""

    def test_init_default_params(self):
        """Test default initialization."""
        composite = SplitScreenComposite()
        assert composite.layout == "2x1"
        assert composite.gap == 4
        assert composite.gap_color == (0, 0, 0)
        assert composite.border_width == 0

    def test_get_cell_rects_2x1(self):
        """Test cell rectangles for 2x1 layout."""
        composite = SplitScreenComposite(layout="2x1", gap=4)
        cells = composite._get_cell_rects(1920, 1080)

        assert len(cells) == 2
        # Left cell
        assert cells[0][0] == 0  # x
        assert cells[0][1] == 0  # y
        assert cells[0][2] == (1920 - 4) // 2  # width
        assert cells[0][3] == 1080  # height
        # Right cell should start after gap
        assert cells[1][0] == cells[0][2] + 4

    def test_get_cell_rects_1x2(self):
        """Test cell rectangles for 1x2 layout."""
        composite = SplitScreenComposite(layout="1x2", gap=4)
        cells = composite._get_cell_rects(1920, 1080)

        assert len(cells) == 2
        # Top cell
        assert cells[0][0] == 0
        assert cells[0][1] == 0
        assert cells[0][2] == 1920
        assert cells[0][3] == (1080 - 4) // 2
        # Bottom cell
        assert cells[1][1] == cells[0][3] + 4

    def test_get_cell_rects_2x2(self):
        """Test cell rectangles for 2x2 layout."""
        composite = SplitScreenComposite(layout="2x2", gap=4)
        cells = composite._get_cell_rects(1920, 1080)

        assert len(cells) == 4
        # All cells should have same dimensions (approximately)
        cell_w = (1920 - 4) // 2
        cell_h = (1080 - 4) // 2
        for cell in cells:
            assert abs(cell[2] - cell_w) <= 1
            assert abs(cell[3] - cell_h) <= 1

    def test_get_cell_rects_1_plus_2(self):
        """Test cell rectangles for 1+2 layout."""
        composite = SplitScreenComposite(layout="1+2", gap=4)
        cells = composite._get_cell_rects(1920, 1080)

        assert len(cells) == 3
        # First cell (large) should be on left, 2/3 width
        assert cells[0][0] == 0
        assert cells[0][3] == 1080  # Full height

    def test_get_cell_rects_2_plus_1(self):
        """Test cell rectangles for 2+1 layout."""
        composite = SplitScreenComposite(layout="2+1", gap=4)
        cells = composite._get_cell_rects(1920, 1080)

        assert len(cells) == 3
        # Last cell (large) should be on right, full height
        assert cells[2][3] == 1080

    def test_get_required_sources(self):
        """Test required sources for different layouts."""
        assert SplitScreenComposite(layout="2x1")._get_required_sources() == 2
        assert SplitScreenComposite(layout="1x2")._get_required_sources() == 2
        assert SplitScreenComposite(layout="2x2")._get_required_sources() == 4
        assert SplitScreenComposite(layout="1+2")._get_required_sources() == 3
        assert SplitScreenComposite(layout="2+1")._get_required_sources() == 3

    def test_invalid_layout_raises(self):
        """Test invalid layout raises error."""
        composite = SplitScreenComposite(layout="invalid")
        with pytest.raises(ValueError, match="Unknown layout"):
            composite._get_cell_rects(1920, 1080)

    @patch("videopython.ai.transforms.FaceTracker")
    def test_apply_requires_enough_videos(self, mock_tracker_class):
        """Test apply raises error with insufficient videos."""
        mock_tracker = MagicMock()
        mock_tracker.detect_and_track.return_value = (0.5, 0.5, 0.1, 0.1)
        mock_tracker_class.return_value = mock_tracker

        frames = np.zeros((10, 1080, 1920, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=30)

        composite = SplitScreenComposite(layout="2x1")
        with pytest.raises(ValueError, match="requires 2 videos"):
            composite.apply(video)  # Only 1 video provided

    @patch("videopython.ai.transforms.FaceTracker")
    def test_apply_2x1_layout(self, mock_tracker_class):
        """Test apply with 2x1 layout."""
        mock_tracker = MagicMock()
        mock_tracker.detect_and_track.return_value = (0.5, 0.5, 0.1, 0.1)
        mock_tracker_class.return_value = mock_tracker

        frames1 = np.full((10, 480, 640, 3), 100, dtype=np.uint8)
        frames2 = np.full((10, 480, 640, 3), 200, dtype=np.uint8)
        video1 = Video.from_frames(frames1, fps=30)
        video2 = Video.from_frames(frames2, fps=30)

        composite = SplitScreenComposite(layout="2x1", gap=4)
        result = composite.apply(video1, video2)

        # Should produce correct frame count
        assert len(result.frames) == 10


class TestAutoFramingCrop:
    """Tests for AutoFramingCrop transformation."""

    def test_init_default_params(self):
        """Test default initialization."""
        crop = AutoFramingCrop()
        assert crop.target_aspect == (9, 16)
        assert crop.framing_rule == "headroom"
        assert crop.headroom == 0.15
        assert crop.lead_room == 0.1
        assert crop.smoothing == 0.85
        assert crop.max_speed == 0.1

    def test_apply_framing_offset_center(self):
        """Test center framing returns same position."""
        crop = AutoFramingCrop(framing_rule="center")
        result = crop._apply_framing_offset(0.5, 0.5, 0.1)
        assert result == (0.5, 0.5)

    def test_apply_framing_offset_headroom(self):
        """Test headroom framing adds space above face."""
        crop = AutoFramingCrop(framing_rule="headroom", headroom=0.15)
        result = crop._apply_framing_offset(0.5, 0.5, 0.1)

        # Face should be positioned lower in frame (target_y < face_cy)
        assert result[0] == 0.5  # x unchanged
        assert result[1] == 0.5 - 0.15  # y offset by headroom

    def test_apply_framing_offset_thirds(self):
        """Test thirds framing places on rule of thirds."""
        crop = AutoFramingCrop(framing_rule="thirds")
        result = crop._apply_framing_offset(0.5, 0.5, 0.1)

        # Should shift y to place face on upper third
        expected_y = 0.5 - (1 / 3 - 0.5)
        assert result[1] == pytest.approx(expected_y)

    def test_clamp_speed_within_limit(self):
        """Test speed clamping when movement is within limit."""
        crop = AutoFramingCrop(max_speed=0.1)
        result = crop._clamp_speed((0.5, 0.5), (0.55, 0.55))

        # Movement is small enough, should reach target
        assert result == (0.55, 0.55)

    def test_clamp_speed_exceeds_limit(self):
        """Test speed clamping when movement exceeds limit."""
        crop = AutoFramingCrop(max_speed=0.1)
        result = crop._clamp_speed((0.0, 0.0), (1.0, 0.0))

        # Movement would be 1.0, but max_speed is 0.1
        assert result[0] == pytest.approx(0.1, abs=0.01)
        assert result[1] == pytest.approx(0.0, abs=0.01)

    def test_clamp_speed_diagonal(self):
        """Test speed clamping with diagonal movement."""
        crop = AutoFramingCrop(max_speed=0.1)
        # Movement of (0.5, 0.5) = distance of 0.707
        result = crop._clamp_speed((0.0, 0.0), (0.5, 0.5))

        # Distance from origin to result should be max_speed
        distance = (result[0] ** 2 + result[1] ** 2) ** 0.5
        assert distance == pytest.approx(0.1, abs=0.01)

    @patch("videopython.ai.transforms.FaceTracker")
    def test_apply_creates_correct_output_shape(self, mock_tracker_class):
        """Test apply produces correct output dimensions."""
        mock_tracker = MagicMock()
        mock_tracker.detect_and_track.return_value = (0.5, 0.5, 0.1, 0.15)
        mock_tracker_class.return_value = mock_tracker

        frames = np.zeros((10, 1080, 1920, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=30)

        crop = AutoFramingCrop(target_aspect=(9, 16))
        result = crop.apply(video)

        # Output should be 9:16 aspect ratio
        out_h, out_w = result.frame_shape[:2]
        assert abs(out_w / out_h - 9 / 16) < 0.01

    @patch("videopython.ai.transforms.FaceTracker")
    def test_apply_without_face_uses_center(self, mock_tracker_class):
        """Test apply uses center position when no face detected."""
        mock_tracker = MagicMock()
        mock_tracker.detect_and_track.return_value = None
        mock_tracker_class.return_value = mock_tracker

        frames = np.zeros((10, 1080, 1920, 3), dtype=np.uint8)
        video = Video.from_frames(frames, fps=30)

        crop = AutoFramingCrop(target_aspect=(9, 16))
        result = crop.apply(video)

        # Should still produce valid output
        assert len(result.frames) == 10

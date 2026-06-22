"""Tests for AI-powered transforms (face tracking)."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from videopython.ai.transforms import (
    FaceSmoothingTracker,
    FaceTrackingCrop,
)
from videopython.base.video import VideoMetadata


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
    """Tests for FaceSmoothingTracker utility."""

    def test_init_default_params(self):
        """Test default initialization."""
        tracker = FaceSmoothingTracker()
        assert tracker.selection_strategy == "largest"
        assert tracker.face_index == 0
        assert tracker.smoothing == 0.8
        assert tracker.detection_interval == 3
        assert tracker.min_face_size == 30

    def test_init_custom_params(self):
        """Test custom initialization."""
        tracker = FaceSmoothingTracker(
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
        tracker = FaceSmoothingTracker(selection_strategy="largest")

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
        tracker = FaceSmoothingTracker(selection_strategy="centered")

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
        tracker = FaceSmoothingTracker(selection_strategy="index", face_index=1)

        faces = [
            MockDetectedFace((0.5, 0.5), 0.3, 0.3),
            MockDetectedFace((0.2, 0.2), 0.1, 0.1),
        ]

        result = tracker._select_face(faces, 1920, 1080)
        assert result is not None
        assert result[:2] == (0.2, 0.2)  # Second face

    def test_select_face_index_out_of_bounds(self):
        """Test face selection with index out of bounds falls back to largest."""
        tracker = FaceSmoothingTracker(selection_strategy="index", face_index=10)

        faces = [
            MockDetectedFace((0.5, 0.5), 0.3, 0.3),
            MockDetectedFace((0.2, 0.2), 0.1, 0.1),
        ]

        result = tracker._select_face(faces, 1920, 1080)
        assert result is not None
        assert result[:2] == (0.5, 0.5)  # Falls back to first (largest)

    def test_select_face_empty_list(self):
        """Test face selection with no faces."""
        tracker = FaceSmoothingTracker()
        result = tracker._select_face([], 1920, 1080)
        assert result is None

    def test_reset_clears_state(self):
        """Test reset clears all tracking state."""
        tracker = FaceSmoothingTracker()
        tracker._last_position = (0.5, 0.5)
        tracker._last_size = (0.1, 0.1)
        tracker._smoothed_position = (0.5, 0.5)
        tracker._smoothed_size = (0.1, 0.1)

        tracker.reset()

        assert tracker._last_position is None
        assert tracker._last_size is None
        assert tracker._smoothed_position is None
        assert tracker._smoothed_size is None

    @patch("videopython.ai.transforms.FaceSmoothingTracker._init_detector")
    def test_detect_and_track_with_mock(self, mock_init):
        """Test detect_and_track with mocked detector."""
        tracker = FaceSmoothingTracker(smoothing=0.0, detection_interval=1)

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            MockDetectedFace((0.5, 0.5), 0.2, 0.2),
        ]
        tracker._detector = mock_detector

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = tracker.detect_and_track(frame, 0)

        assert result is not None
        assert result == (0.5, 0.5, 0.2, 0.2)

    @patch("videopython.ai.transforms.FaceSmoothingTracker._init_detector")
    def test_detect_and_track_smoothing(self, mock_init):
        """Test smoothing over multiple frames."""
        tracker = FaceSmoothingTracker(smoothing=0.5, detection_interval=1)

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

    @patch("videopython.ai.transforms.FaceSmoothingTracker._init_detector")
    def test_detection_interval_skips_frames(self, mock_init):
        """Test detection is only run on interval frames."""
        tracker = FaceSmoothingTracker(smoothing=0.0, detection_interval=3)

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
        assert crop.framing_rule == "offset"
        assert crop.headroom == 0.15
        assert crop.smoothing == 0.8
        assert crop.max_speed is None
        assert crop.fallback == "last_position"

    def test_init_custom_params(self):
        """Test custom initialization."""
        crop = FaceTrackingCrop(
            target_aspect=(1, 1),
            face_selection="centered",
            padding=0.5,
            vertical_offset=0.0,
            framing_rule="headroom",
            headroom=0.2,
            smoothing=0.5,
            max_speed=0.1,
            fallback="center",
        )
        assert crop.target_aspect == (1, 1)
        assert crop.face_selection == "centered"
        assert crop.padding == 0.5
        assert crop.vertical_offset == 0.0
        assert crop.framing_rule == "headroom"
        assert crop.headroom == 0.2
        assert crop.smoothing == 0.5
        assert crop.max_speed == 0.1
        assert crop.fallback == "center"

    def test_track_positions_fixed_crop_size_and_centering(self):
        """The crop window is the fixed aspect-fit box, centered on the face."""
        crop = FaceTrackingCrop(target_aspect=(9, 16), framing_rule="center", smoothing=0.0)
        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)] * 3
        with patch("videopython.ai.transforms.FaceSmoothingTracker") as tracker_cls:
            tracker_cls.return_value.detect_and_track.return_value = (0.5, 0.5, 0.1, 0.15)
            positions = crop._track_crop_positions(frames, 1920, 1080)

        # For 9:16 from 1920x1080: crop = 606x1080 (even-floored), centered.
        assert positions == [(657, 0)] * 3  # int(0.5*1920 - 606/2) = 657

    def test_track_positions_clamped_to_bounds(self):
        crop = FaceTrackingCrop(target_aspect=(9, 16), framing_rule="center", smoothing=0.0)
        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)]
        with patch("videopython.ai.transforms.FaceSmoothingTracker") as tracker_cls:
            tracker_cls.return_value.detect_and_track.return_value = (0.05, 0.5, 0.1, 0.15)
            positions = crop._track_crop_positions(frames, 1920, 1080)

        assert positions == [(0, 0)]  # clamped at the left edge

    @pytest.mark.parametrize(
        "src_w, src_h, aspect",
        [
            (1920, 1080, (9, 16)),
            (1920, 1080, (1, 1)),
            (1280, 720, (4, 5)),
            (1080, 1920, (16, 9)),
        ],
    )
    def test_predict_metadata_output_dims(self, src_w, src_h, aspect):
        """The dry-run output dims are the fixed crop window the streaming filter
        emits: the requested aspect ratio, even (ffmpeg requires it), and fitting
        within the source. Everything but the dimensions is identity.

        Asserts these invariants directly rather than re-deriving them via
        ``_resolved_output_dims`` (which ``predict_metadata`` calls), so a bug in
        that shared helper can actually surface here."""
        crop = FaceTrackingCrop(target_aspect=aspect)
        meta = VideoMetadata(height=src_h, width=src_w, fps=30, frame_count=4, total_seconds=4 / 30)
        predicted = crop.predict_metadata(meta)

        out_w, out_h = predicted.width, predicted.height
        assert abs(out_w / out_h - aspect[0] / aspect[1]) < 0.02
        assert out_w % 2 == 0 and out_h % 2 == 0
        assert out_w <= src_w and out_h <= src_h
        # Identity for everything except dimensions.
        assert predicted.fps == meta.fps
        assert predicted.frame_count == meta.frame_count
        assert predicted.total_seconds == meta.total_seconds

    @patch("videopython.ai.transforms.FaceSmoothingTracker")
    def test_track_positions_fallback_center(self, mock_tracker_class):
        """With no face detected, ``center`` fallback centers the crop window."""
        mock_tracker = MagicMock()
        mock_tracker.detect_and_track.return_value = None  # No face
        mock_tracker_class.return_value = mock_tracker

        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)] * 10

        crop = FaceTrackingCrop(target_aspect=(9, 16), fallback="center")
        positions = crop._track_crop_positions(frames, 1920, 1080)

        # 9:16 of 1920x1080 -> 606x1080, centered: ((1920-606)//2, 0) = (657, 0).
        assert positions == [(657, 0)] * 10


class TestFaceTrackingCropFraming:
    """Tests for framing/speed features merged into FaceTrackingCrop."""

    def test_apply_framing_offset_center(self):
        crop = FaceTrackingCrop(framing_rule="center")
        result = crop._apply_framing_offset(0.5, 0.5, 0.1)
        assert result == (0.5, 0.5)

    def test_apply_framing_offset_headroom(self):
        crop = FaceTrackingCrop(framing_rule="headroom", headroom=0.15)
        result = crop._apply_framing_offset(0.5, 0.5, 0.1)
        assert result[0] == 0.5
        assert result[1] == 0.5 - 0.15

    def test_apply_framing_offset_thirds(self):
        crop = FaceTrackingCrop(framing_rule="thirds")
        result = crop._apply_framing_offset(0.5, 0.5, 0.1)
        expected_y = 0.5 - (1 / 3 - 0.5)
        assert result[1] == pytest.approx(expected_y)

    def test_clamp_speed_within_limit(self):
        crop = FaceTrackingCrop(max_speed=0.1)
        result = crop._clamp_speed((0.5, 0.5), (0.55, 0.55))
        assert result == (0.55, 0.55)

    def test_clamp_speed_exceeds_limit(self):
        crop = FaceTrackingCrop(max_speed=0.1)
        result = crop._clamp_speed((0.0, 0.0), (1.0, 0.0))
        assert result[0] == pytest.approx(0.1, abs=0.01)
        assert result[1] == pytest.approx(0.0, abs=0.01)

    def test_clamp_speed_diagonal(self):
        crop = FaceTrackingCrop(max_speed=0.1)
        result = crop._clamp_speed((0.0, 0.0), (0.5, 0.5))
        distance = (result[0] ** 2 + result[1] ** 2) ** 0.5
        assert distance == pytest.approx(0.1, abs=0.01)

    @patch("videopython.ai.transforms.FaceSmoothingTracker")
    def test_track_positions_headroom_without_face_uses_fallback(self, mock_tracker_class):
        mock_tracker = MagicMock()
        mock_tracker.detect_and_track.return_value = None
        mock_tracker_class.return_value = mock_tracker

        frames = [np.zeros((1080, 1920, 3), dtype=np.uint8)] * 10

        crop = FaceTrackingCrop(target_aspect=(9, 16), framing_rule="headroom", fallback="center")
        positions = crop._track_crop_positions(frames, 1920, 1080)

        # No face -> centered crop regardless of framing rule: ((1920-606)//2, 0).
        assert positions == [(657, 0)] * 10


class TestGPUFaceTracking:
    """Tests for GPU-accelerated face tracking features."""

    def test_face_tracker_gpu_params(self):
        """Test FaceSmoothingTracker accepts GPU parameters."""
        tracker = FaceSmoothingTracker(
            backend="gpu",
            sample_rate=5,
            batch_size=8,
        )
        assert tracker.backend == "gpu"
        assert tracker.sample_rate == 5
        assert tracker.batch_size == 8

    def test_face_tracker_default_backend_is_auto(self):
        """Test FaceSmoothingTracker defaults to auto backend."""
        tracker = FaceSmoothingTracker()
        assert tracker.backend == "auto"

    def test_face_tracking_crop_gpu_params(self):
        """Test FaceTrackingCrop accepts and stores GPU parameters."""
        crop = FaceTrackingCrop(
            backend="gpu",
            sample_rate=5,
        )
        assert crop.backend == "gpu"
        assert crop.sample_rate == 5

    def test_interpolate_bbox(self):
        """Test bounding box interpolation."""
        bbox1 = (0.0, 0.0, 0.1, 0.1)
        bbox2 = (1.0, 1.0, 0.2, 0.2)

        # t=0 should return bbox1
        result = FaceSmoothingTracker._interpolate_bbox(bbox1, bbox2, 0.0)
        assert result == bbox1

        # t=1 should return bbox2
        result = FaceSmoothingTracker._interpolate_bbox(bbox1, bbox2, 1.0)
        assert result == bbox2

        # t=0.5 should return midpoint
        result = FaceSmoothingTracker._interpolate_bbox(bbox1, bbox2, 0.5)
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(0.15)
        assert result[3] == pytest.approx(0.15)

    def test_interpolate_bbox_quarter(self):
        """Test bounding box interpolation at t=0.25."""
        bbox1 = (0.2, 0.2, 0.1, 0.1)
        bbox2 = (0.6, 0.6, 0.2, 0.2)

        result = FaceSmoothingTracker._interpolate_bbox(bbox1, bbox2, 0.25)
        assert result[0] == pytest.approx(0.3)  # 0.2 + 0.25 * 0.4
        assert result[1] == pytest.approx(0.3)
        assert result[2] == pytest.approx(0.125)  # 0.1 + 0.25 * 0.1
        assert result[3] == pytest.approx(0.125)

    @patch("videopython.ai.transforms.FaceSmoothingTracker._init_detector")
    def test_track_video_with_mock(self, mock_init):
        """Test track_video with mocked detector."""
        tracker = FaceSmoothingTracker(smoothing=0.0, sample_rate=1, backend="cpu")

        mock_detector = MagicMock()
        # Return faces for batched detection
        mock_detector.detect_batch.return_value = [
            [MockDetectedFace((0.5, 0.5), 0.2, 0.2)],
            [MockDetectedFace((0.5, 0.5), 0.2, 0.2)],
            [MockDetectedFace((0.5, 0.5), 0.2, 0.2)],
        ]
        tracker._detector = mock_detector

        frames = np.zeros((3, 1080, 1920, 3), dtype=np.uint8)
        results = tracker.track_video(frames)

        assert len(results) == 3
        assert all(r is not None for r in results)
        # Check detector was called with batched frames
        mock_detector.detect_batch.assert_called_once()

    @patch("videopython.ai.transforms.FaceSmoothingTracker._init_detector")
    def test_track_video_with_sampling(self, mock_init):
        """Test track_video with frame sampling and interpolation."""
        tracker = FaceSmoothingTracker(smoothing=0.0, sample_rate=3, backend="gpu")

        mock_detector = MagicMock()
        # Return faces only for sampled frames (frames 0, 3, 6, 9)
        # For 10 frames with sample_rate=3: indices 0, 3, 6, 9
        mock_detector.detect_batch.return_value = [
            [MockDetectedFace((0.3, 0.3), 0.1, 0.1)],  # frame 0
            [MockDetectedFace((0.5, 0.5), 0.1, 0.1)],  # frame 3
            [MockDetectedFace((0.7, 0.7), 0.1, 0.1)],  # frame 6
            [MockDetectedFace((0.7, 0.7), 0.1, 0.1)],  # frame 9
        ]
        tracker._detector = mock_detector

        frames = np.zeros((10, 1080, 1920, 3), dtype=np.uint8)
        results = tracker.track_video(frames)

        assert len(results) == 10
        # Sampled frames should have exact values
        assert results[0][0] == pytest.approx(0.3, abs=0.01)
        # Interpolated frames should be between sampled values
        # Frame 1 should be interpolated between frame 0 and 3
        assert 0.3 < results[1][0] < 0.5

    @patch("videopython.ai.transforms.FaceSmoothingTracker._init_detector")
    def test_track_video_empty_frames(self, mock_init):
        """Test track_video with empty frame list."""
        tracker = FaceSmoothingTracker()
        mock_detector = MagicMock()
        tracker._detector = mock_detector

        results = tracker.track_video(np.array([]))
        assert results == []

    @patch("videopython.ai.understanding.faces._FaceDetector")
    def test_face_tracker_passes_gpu_params_to_detector(self, mock_detector_class):
        """Test FaceSmoothingTracker passes GPU params to internal detector backend."""
        mock_detector = MagicMock()
        mock_detector.detect_batch.return_value = [[]]
        mock_detector_class.return_value = mock_detector

        tracker = FaceSmoothingTracker(
            backend="gpu",
            min_face_size=50,
        )

        # Initialize detector by calling track_video
        frames = np.zeros((1, 100, 100, 3), dtype=np.uint8)
        tracker.track_video(frames)

        # Verify detector backend was created with correct params
        mock_detector_class.assert_called_once_with(
            min_face_size=50,
            backend="gpu",
        )

    @patch("videopython.ai.transforms.FaceSmoothingTracker")
    def test_face_tracking_crop_passes_params_to_tracker(self, mock_tracker_class):
        """Test FaceTrackingCrop passes GPU params to FaceSmoothingTracker."""
        mock_tracker = MagicMock()
        mock_tracker.detect_and_track.return_value = (0.5, 0.5, 0.1, 0.1)
        mock_tracker_class.return_value = mock_tracker

        frames = [np.zeros((480, 640, 3), dtype=np.uint8)] * 5

        crop = FaceTrackingCrop(
            backend="gpu",
            sample_rate=5,
        )
        crop._track_crop_positions(frames, 640, 480)

        # Verify FaceSmoothingTracker was created with GPU params
        call_kwargs = mock_tracker_class.call_args[1]
        assert call_kwargs["backend"] == "gpu"
        assert call_kwargs["sample_rate"] == 5


class TestFaceCropSubtitleValidateGap:
    """Step 0 + Step 2 together, for the exact scenario in TODO.md.

    Importing ``videopython.ai.transforms`` (top of this file) registers the
    ``face_crop`` op, so a plan combining it with ``add_subtitles`` can be
    dry-run. Lives here, not in the editing suite, to keep that suite free of
    the optional ``[ai]`` extra.
    """

    @staticmethod
    def _plan(ops: list[dict[str, Any]]) -> dict[str, Any]:
        return {"segments": [{"source": "fake.mp4", "start": 0.0, "end": 2.0, "operations": ops}]}

    @staticmethod
    def _transcription():
        from videopython.base.transcription import Transcription, TranscriptionSegment, TranscriptionWord

        words = [
            TranscriptionWord(start=s, end=e, word=w)
            for s, e, w in [(0.0, 0.4, "Hello"), (0.4, 0.8, "there"), (0.8, 1.2, "world")]
        ]
        return Transcription(segments=[TranscriptionSegment.from_words(words)])

    def test_reasonable_plan_passes_and_reports_cropped_dims(self):
        from videopython.editing.video_edit import VideoEdit

        plan = self._plan([{"op": "face_crop", "target_aspect": [9, 16]}, {"op": "add_subtitles"}])
        source = VideoMetadata(height=1080, width=1920, fps=30, frame_count=60, total_seconds=2.0)
        out = VideoEdit.from_dict(plan).validate_with_metadata(source, context={"transcription": self._transcription()})
        assert (out.width, out.height) == (606, 1080)


class TestFaceCropStreaming:
    """face_crop compiles to a sendcmd-driven crop track at plan build."""

    @staticmethod
    def _plan():
        from videopython.editing import VideoEdit

        return VideoEdit.model_validate(
            {
                "segments": [
                    {
                        "source": "src/tests/test_data/small_video.mp4",
                        "start": 2.0,
                        "end": 6.0,
                        "operations": [
                            {
                                "op": "face_crop",
                                "target_aspect": [9, 16],
                                "framing_rule": "center",
                                "smoothing": 0.0,
                            }
                        ],
                    }
                ]
            }
        )

    def test_classifies_as_filter(self):
        from videopython.editing import StreamingClass

        report = self._plan().streamability()
        assert report.entries[0].streaming_class is StreamingClass.FILTER
        assert report.streamable

    def test_streams_and_tracks_the_face(self, tmp_path):
        import glob
        import tempfile as _tempfile
        from unittest.mock import patch as _patch

        from videopython.base.video import Video

        before = set(glob.glob(_tempfile.gettempdir() + "/*.cmd"))
        plan = self._plan()
        with _patch("videopython.ai.transforms.FaceSmoothingTracker") as tracker_cls:
            # Face drifts left -> right across the clip.
            tracker_cls.return_value.detect_and_track.side_effect = lambda frame, i: (
                0.3 + 0.4 * (i / 96),
                0.5,
                0.1,
                0.15,
            )
            out = plan.run_to_file(tmp_path / "out.mp4")
        assert set(glob.glob(_tempfile.gettempdir() + "/*.cmd")) == before, "sendcmd file leaked"

        video = Video.from_path(str(out))
        assert video.frames.shape == (96, 500, 280, 3)  # 9:16 of 800x500

        source = Video.from_path("src/tests/test_data/small_video.mp4", start_second=2.0, end_second=6.0)

        def best_x(out_frame: np.ndarray, src_frame: np.ndarray, width: int = 280) -> int:
            errors = [
                np.abs(out_frame.astype(np.float32) - src_frame[:, x : x + width].astype(np.float32)).mean()
                for x in range(0, 800 - width, 20)
            ]
            return int(np.argmin(errors)) * 20

        x_early = best_x(video.frames[5], source.frames[5])
        x_late = best_x(video.frames[90], source.frames[90])
        assert x_late > x_early + 100, f"crop did not follow the face: {x_early} -> {x_late}"

    def test_behind_frame_effects_is_rejected(self, tmp_path):
        """face_crop cannot reproduce post-effect frames at compile time."""
        from unittest.mock import patch as _patch

        import pytest as _pytest

        from videopython.base.exceptions import PlanValidationError
        from videopython.editing import VideoEdit

        plan = VideoEdit.model_validate(
            {
                "segments": [
                    {
                        "source": "src/tests/test_data/small_video.mp4",
                        "start": 2.0,
                        "end": 4.0,
                        "operations": [
                            {"op": "fade", "mode": "in", "duration": 0.5},
                            {"op": "face_crop", "target_aspect": [9, 16]},
                        ],
                    }
                ]
            }
        )
        with _patch("videopython.ai.transforms.FaceSmoothingTracker") as tracker_cls:
            tracker_cls.return_value.detect_and_track.return_value = (0.5, 0.5, 0.1, 0.15)
            with _pytest.raises(PlanValidationError, match="cannot stream"):
                plan.run_to_file(tmp_path / "out.mp4")

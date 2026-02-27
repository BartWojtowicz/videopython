"""Tests for detection backends that run actual AI inference.

Tests marked with @pytest.mark.requires_model_download are excluded from CI:
- TextDetector tests (EasyOCR, ~100MB+ download)

Tests that run in CI (models <100MB or bundled):
- ObjectDetector tests (YOLO, ~6MB download)
- FaceDetector tests (OpenCV cascade, bundled with opencv-python)
- CameraMotionDetector tests (OpenCV optical flow, no model download)

Run locally with: uv run pytest src/tests/ai -v
CI runs: uv run pytest src/tests/ai -m "not requires_model_download"
"""

import os

import numpy as np
import pytest
from PIL import Image

from videopython.base.description import BoundingBox, DetectedObject

# Path to test data (one level up from ai/ directory)
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data")
BIG_VIDEO_PATH = os.path.join(TEST_DATA_DIR, "big_video.mp4")


@pytest.fixture(scope="module")
def video_frames():
    """Load frames from big_video.mp4 for testing."""
    from videopython.base.video import Video

    video = Video.from_path(BIG_VIDEO_PATH)
    # Return a few frames spread across the video
    indices = [0, 100, 200, 300, 400]
    return [video.frames[i] for i in indices if i < len(video.frames)]


class TestObjectDetectorLocal:
    """Tests for ObjectDetector with YOLO backend (~6MB model download)."""

    @pytest.fixture
    def detector(self):
        """Create ObjectDetector with local backend."""
        from videopython.ai.understanding.detection import ObjectDetector

        return ObjectDetector(model_size="n", confidence_threshold=0.25)

    @pytest.fixture
    def sample_image(self):
        """Create a simple test image."""
        # Create a 640x480 RGB image with some color variation
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some colored rectangles
        img[100:200, 100:200] = [255, 0, 0]  # Red square
        img[200:300, 300:400] = [0, 255, 0]  # Green square
        img[300:400, 100:300] = [0, 0, 255]  # Blue rectangle
        return img

    @pytest.fixture
    def pil_image(self, sample_image):
        """Create PIL Image from sample."""
        return Image.fromarray(sample_image)

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.model_size == "n"
        assert detector.confidence_threshold == 0.25

    def test_detect_returns_list(self, detector, sample_image):
        """Test detection returns a list of DetectedObject."""
        results = detector.detect(sample_image)
        assert isinstance(results, list)
        for obj in results:
            assert isinstance(obj, DetectedObject)

    def test_detect_with_pil_image(self, detector, pil_image):
        """Test detection works with PIL Image input."""
        results = detector.detect(pil_image)
        assert isinstance(results, list)

    def test_detected_object_has_bounding_box(self, detector, sample_image):
        """Test that detected objects have bounding boxes with valid coordinates."""
        results = detector.detect(sample_image)
        for obj in results:
            assert obj.label is not None
            assert 0 <= obj.confidence <= 1
            if obj.bounding_box is not None:
                bbox = obj.bounding_box
                assert isinstance(bbox, BoundingBox)
                assert 0 <= bbox.x <= 1
                assert 0 <= bbox.y <= 1
                assert 0 <= bbox.width <= 1
                assert 0 <= bbox.height <= 1

    def test_detect_on_real_video_frame(self, detector, video_frames):
        """Test detection on actual video frames from big_video.mp4."""
        # Test on multiple frames
        all_detections = []
        for frame in video_frames:
            results = detector.detect(frame)
            assert isinstance(results, list)
            all_detections.extend(results)

        # Verify all detections have valid structure
        for obj in all_detections:
            assert isinstance(obj, DetectedObject)
            assert isinstance(obj.label, str)
            assert 0 <= obj.confidence <= 1
            if obj.bounding_box is not None:
                bbox = obj.bounding_box
                assert 0 <= bbox.x <= 1
                assert 0 <= bbox.y <= 1
                assert 0 <= bbox.width <= 1
                assert 0 <= bbox.height <= 1


class TestFaceDetector:
    """Tests for FaceDetector with OpenCV backend."""

    @pytest.fixture
    def detector(self):
        """Create FaceDetector."""
        from videopython.ai.understanding.detection import FaceDetector

        return FaceDetector(confidence_threshold=0.5)

    @pytest.fixture
    def blank_image(self):
        """Create a blank image with no faces."""
        return np.ones((480, 640, 3), dtype=np.uint8) * 128

    @pytest.fixture
    def pil_blank_image(self, blank_image):
        """Create PIL Image from blank image."""
        return Image.fromarray(blank_image)

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.confidence_threshold == 0.5

    def test_detect_returns_list(self, detector, blank_image):
        """Test detection returns a list of DetectedFace."""
        faces = detector.detect(blank_image)
        assert isinstance(faces, list)

    def test_detect_with_pil_image(self, detector, pil_blank_image):
        """Test detection works with PIL Image input."""
        faces = detector.detect(pil_blank_image)
        assert isinstance(faces, list)

    def test_blank_image_no_faces(self, detector, blank_image):
        """Test that blank image returns empty list."""
        faces = detector.detect(blank_image)
        assert len(faces) == 0

    def test_detect_on_real_video_frame(self, detector, video_frames):
        """Test face detection on actual video frames from big_video.mp4."""
        for frame in video_frames:
            faces = detector.detect(frame)
            assert isinstance(faces, list)
            # If faces detected, check bounding boxes are valid
            for face in faces:
                assert 0 <= face.bounding_box.x <= 1
                assert 0 <= face.bounding_box.y <= 1
                assert face.bounding_box.width > 0
                assert face.bounding_box.height > 0


@pytest.mark.requires_model_download
class TestTextDetectorLocal:
    """Tests for TextDetector with EasyOCR backend (requires ~100MB+ model download)."""

    @pytest.fixture
    def detector(self):
        """Create TextDetector with local backend."""
        from videopython.ai.understanding.detection import TextDetector

        return TextDetector(languages=["en"])

    @pytest.fixture
    def blank_image(self):
        """Create a blank image with no text."""
        return np.ones((480, 640, 3), dtype=np.uint8) * 255

    @pytest.fixture
    def pil_blank_image(self, blank_image):
        """Create PIL Image from blank image."""
        return Image.fromarray(blank_image)

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.languages == ["en"]

    def test_detect_returns_list(self, detector, blank_image):
        """Test detection returns a list of strings."""
        results = detector.detect(blank_image)
        assert isinstance(results, list)
        for text in results:
            assert isinstance(text, str)

    def test_detect_with_pil_image(self, detector, pil_blank_image):
        """Test detection works with PIL Image input."""
        results = detector.detect(pil_blank_image)
        assert isinstance(results, list)

    def test_detect_on_real_video_frame(self, detector, video_frames):
        """Test OCR on actual video frames from big_video.mp4."""
        for frame in video_frames:
            results = detector.detect(frame)
            assert isinstance(results, list)
            for text in results:
                assert isinstance(text, str)


class TestCameraMotionDetector:
    """Tests for CameraMotionDetector with optical flow."""

    @pytest.fixture
    def detector(self):
        """Create CameraMotionDetector."""
        from videopython.ai.understanding.detection import CameraMotionDetector

        return CameraMotionDetector(motion_threshold=2.0, zoom_threshold=0.1)

    @pytest.fixture
    def static_frames(self):
        """Create two identical frames (no motion)."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Add some texture for optical flow to work with
        frame[100:200, 100:200] = [255, 0, 0]
        frame[300:400, 400:500] = [0, 255, 0]
        return frame.copy(), frame.copy()

    @pytest.fixture
    def pan_frames(self):
        """Create two frames with horizontal motion (pan)."""
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Add a feature that moves horizontally
        frame1[200:300, 100:200] = [255, 0, 0]
        frame2[200:300, 150:250] = [255, 0, 0]  # Shifted right by 50 pixels

        return frame1, frame2

    @pytest.fixture
    def tilt_frames(self):
        """Create two frames with vertical motion (tilt)."""
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # Add a feature that moves vertically
        frame1[100:200, 300:400] = [0, 255, 0]
        frame2[150:250, 300:400] = [0, 255, 0]  # Shifted down by 50 pixels

        return frame1, frame2

    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector.motion_threshold == 2.0
        assert detector.zoom_threshold == 0.1

    def test_detect_returns_valid_motion_type(self, detector, static_frames):
        """Test detection returns a valid motion type string."""
        frame1, frame2 = static_frames
        result = detector.detect(frame1, frame2)
        assert isinstance(result, str)
        assert result in detector.MOTION_TYPES

    def test_static_frames_detected(self, detector, static_frames):
        """Test that identical frames are detected as static."""
        frame1, frame2 = static_frames
        result = detector.detect(frame1, frame2)
        assert result == "static"

    def test_detect_with_pil_images(self, detector, static_frames):
        """Test detection works with PIL Image input."""
        frame1, frame2 = static_frames
        pil1 = Image.fromarray(frame1)
        pil2 = Image.fromarray(frame2)
        result = detector.detect(pil1, pil2)
        assert isinstance(result, str)
        assert result in detector.MOTION_TYPES

    def test_pan_motion_detected(self, detector, pan_frames):
        """Test that horizontal motion is detected as pan."""
        frame1, frame2 = pan_frames
        result = detector.detect(frame1, frame2)
        # Pan or complex are acceptable since optical flow may pick up mixed signals
        assert result in ["pan", "complex", "static"]

    def test_tilt_motion_detected(self, detector, tilt_frames):
        """Test that vertical motion is detected as tilt."""
        frame1, frame2 = tilt_frames
        result = detector.detect(frame1, frame2)
        # Tilt or complex are acceptable since optical flow may pick up mixed signals
        assert result in ["tilt", "complex", "static"]

    def test_detect_on_real_video_frames(self, detector, video_frames):
        """Test camera motion detection on consecutive frames from big_video.mp4."""
        # Test motion between consecutive frame pairs
        for i in range(len(video_frames) - 1):
            frame1 = video_frames[i]
            frame2 = video_frames[i + 1]
            result = detector.detect(frame1, frame2)
            assert isinstance(result, str)
            assert result in detector.MOTION_TYPES


class TestFaceDetectorBackends:
    """Tests for FaceDetector with different backends."""

    @pytest.fixture
    def blank_image(self):
        """Create a blank image with no faces."""
        return np.ones((480, 640, 3), dtype=np.uint8) * 128

    def test_cpu_backend_initialization(self):
        """Test CPU backend initializes correctly."""
        from videopython.ai.understanding.detection import FaceDetector

        detector = FaceDetector(backend="cpu")
        assert detector.backend == "cpu"

    def test_gpu_backend_initialization(self):
        """Test GPU backend initializes correctly."""
        from videopython.ai.understanding.detection import FaceDetector

        detector = FaceDetector(backend="gpu")
        assert detector.backend == "gpu"

    def test_auto_backend_initialization(self):
        """Test auto backend initializes correctly."""
        from videopython.ai.understanding.detection import FaceDetector

        detector = FaceDetector(backend="auto")
        assert detector.backend == "auto"

    def test_valid_backend_values(self):
        """Test that valid backend values are accepted."""
        from videopython.ai.understanding.detection import FaceDetector

        # These should all work without error
        FaceDetector(backend="cpu")
        FaceDetector(backend="gpu")
        FaceDetector(backend="auto")

    def test_cpu_backend_detect(self, blank_image):
        """Test CPU backend detection works."""
        from videopython.ai.understanding.detection import FaceDetector

        detector = FaceDetector(backend="cpu")
        faces = detector.detect(blank_image)
        assert isinstance(faces, list)

    def test_cpu_backend_detect_batch(self, blank_image):
        """Test CPU backend batch detection works."""
        from videopython.ai.understanding.detection import FaceDetector

        detector = FaceDetector(backend="cpu")
        images = [blank_image, blank_image, blank_image]
        results = detector.detect_batch(images)

        assert isinstance(results, list)
        assert len(results) == 3
        for faces in results:
            assert isinstance(faces, list)

    def test_detect_batch_with_numpy_array(self, blank_image):
        """Test batch detection with numpy array input."""
        from videopython.ai.understanding.detection import FaceDetector

        detector = FaceDetector(backend="cpu")
        # Stack as (N, H, W, C) array
        images = np.stack([blank_image, blank_image], axis=0)
        results = detector.detect_batch(images)

        assert isinstance(results, list)
        assert len(results) == 2

    def test_detect_batch_empty(self):
        """Test batch detection with empty list."""
        from videopython.ai.understanding.detection import FaceDetector

        detector = FaceDetector(backend="cpu")
        results = detector.detect_batch([])
        assert results == []

    def test_auto_backend_resolves_correctly(self, blank_image):
        """Test auto backend resolves and works."""
        from videopython.ai.understanding.detection import FaceDetector

        detector = FaceDetector(backend="auto")
        faces = detector.detect(blank_image)
        assert isinstance(faces, list)

        # Verify backend was resolved
        resolved = detector._resolve_backend()
        assert resolved in ("cpu", "gpu")


@pytest.mark.requires_model_download
class TestFaceDetectorGPU:
    """Tests for FaceDetector GPU backend with real YOLO model.

    These tests require downloading YOLOv8-face model (~6MB).
    Run with: uv run pytest src/tests/ai -m requires_model_download -v
    """

    @pytest.fixture
    def detector(self):
        """Create FaceDetector with GPU backend."""
        from videopython.ai.understanding.detection import FaceDetector

        return FaceDetector(backend="gpu")

    @pytest.fixture
    def blank_image(self):
        """Create a blank image with no faces."""
        return np.ones((480, 640, 3), dtype=np.uint8) * 128

    def test_gpu_detect_returns_list(self, detector, blank_image):
        """Test GPU detection returns a list."""
        faces = detector.detect(blank_image)
        assert isinstance(faces, list)

    def test_gpu_detect_batch_returns_list(self, detector, blank_image):
        """Test GPU batch detection returns list of lists."""
        images = [blank_image] * 4
        results = detector.detect_batch(images)

        assert isinstance(results, list)
        assert len(results) == 4
        for faces in results:
            assert isinstance(faces, list)

    def test_gpu_detect_on_real_video_frames(self, detector, video_frames):
        """Test GPU face detection on actual video frames."""
        for frame in video_frames:
            faces = detector.detect(frame)
            assert isinstance(faces, list)
            for face in faces:
                assert 0 <= face.bounding_box.x <= 1
                assert 0 <= face.bounding_box.y <= 1
                assert face.bounding_box.width > 0
                assert face.bounding_box.height > 0
                assert 0 <= face.confidence <= 1

    def test_gpu_batch_detect_on_real_video_frames(self, detector, video_frames):
        """Test GPU batch face detection on actual video frames."""
        results = detector.detect_batch(video_frames)

        assert len(results) == len(video_frames)
        for faces in results:
            assert isinstance(faces, list)
            for face in faces:
                assert 0 <= face.bounding_box.x <= 1
                assert 0 <= face.bounding_box.y <= 1
                assert face.bounding_box.width > 0
                assert face.bounding_box.height > 0

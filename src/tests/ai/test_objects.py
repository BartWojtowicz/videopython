"""Tests for the ObjectDetector understanding primitive (mocked YOLO)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from videopython.ai.understanding.objects import ObjectDetector


class FakeBoxes:
    """Minimal stand-in for an Ultralytics Results.boxes object."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = [np.array(b, dtype=float) for b in xyxy]
        self.conf = list(conf)
        self.cls = list(cls)

    def __len__(self):
        return len(self.xyxy)


class FakeResult:
    def __init__(self, boxes, orig_shape):
        self.boxes = boxes
        self.orig_shape = orig_shape


def _detector_with(results, class_names=None, **kwargs):
    """Build an ObjectDetector whose model is a mock returning ``results``."""
    det = ObjectDetector(**kwargs)
    det._class_names = class_names or {0: "person", 2: "car"}
    det._yolo_model = MagicMock(return_value=results)
    return det


def _two_object_result():
    # person (conf 0.9) and car (conf 0.8) in a 100h x 200w image.
    boxes = FakeBoxes(
        xyxy=[[20.0, 10.0, 120.0, 60.0], [0.0, 0.0, 100.0, 50.0]],
        conf=[0.9, 0.8],
        cls=[0, 2],
    )
    return [FakeResult(boxes, orig_shape=(100, 200))]


class TestObjectDetector:
    def test_detect_normalizes_boxes(self):
        det = _detector_with(_two_object_result())
        objs = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))

        assert [o.label for o in objs] == ["person", "car"]
        person = objs[0]
        assert person.confidence == pytest.approx(0.9)
        bb = person.bounding_box
        assert bb is not None
        # 20/200, 10/100, 100/200, 50/100
        assert bb.x == pytest.approx(0.1)
        assert bb.y == pytest.approx(0.1)
        assert bb.width == pytest.approx(0.5)
        assert bb.height == pytest.approx(0.5)

    def test_results_sorted_by_confidence(self):
        # Provide lower-confidence first; detector should sort descending.
        boxes = FakeBoxes(
            xyxy=[[0, 0, 10, 10], [0, 0, 20, 20]],
            conf=[0.4, 0.95],
            cls=[2, 0],
        )
        det = _detector_with([FakeResult(boxes, (100, 100))])
        objs = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert [o.confidence for o in objs] == [0.95, 0.4]

    def test_class_filter_drops_other_labels(self):
        det = _detector_with(_two_object_result(), class_filter=("person",))
        objs = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        assert [o.label for o in objs] == ["person"]

    def test_detect_handles_empty_results(self):
        det = _detector_with([])
        assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []

    def test_detect_batch_list(self):
        det = _detector_with(_two_object_result() * 2)
        frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(2)]
        batched = det.detect_batch(frames)
        assert len(batched) == 2
        assert all(len(d) == 2 for d in batched)

    def test_detect_batch_ndarray(self):
        det = _detector_with(_two_object_result() * 3)
        frames = np.zeros((3, 100, 200, 3), dtype=np.uint8)
        batched = det.detect_batch(frames)
        assert len(batched) == 3

    def test_detect_batch_empty(self):
        det = _detector_with([])
        assert det.detect_batch([]) == []

    def test_cpu_backend_resolves_without_torch(self):
        det = ObjectDetector(backend="cpu")
        assert det.execution_device() == "cpu"

    def test_confidence_threshold_passed_to_model(self):
        det = _detector_with(_two_object_result(), confidence_threshold=0.7)
        det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        _, kwargs = det._yolo_model.call_args
        assert kwargs["conf"] == 0.7

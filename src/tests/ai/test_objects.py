"""Tests for the ObjectDetector understanding primitive (mocked D-FINE)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from videopython.ai.understanding.objects import ObjectDetector


def _result(scores, labels, boxes):
    """A post_process_object_detection result dict (one per image).

    ``.tolist()`` is all ObjectDetector reads, so numpy arrays stand in for the
    torch tensors transformers returns.
    """
    return {
        "scores": np.array(scores, dtype=float),
        "labels": np.array(labels, dtype=int),
        "boxes": np.array(boxes, dtype=float).reshape(-1, 4),
    }


def _detector_with(results, class_names=None, **kwargs):
    """Build an ObjectDetector whose processor/model are mocked.

    ``results`` is the list (one dict per image) that the mocked
    ``post_process_object_detection`` returns. The real ``_infer`` body (torch
    no_grad, target_sizes) still runs; only the model + processor are faked.
    """
    det = ObjectDetector(**kwargs)
    det._class_names = class_names or {0: "person", 2: "car"}
    processor = MagicMock()
    processor.return_value = {"pixel_values": np.zeros((len(results), 3, 8, 8), dtype=np.float32)}
    processor.post_process_object_detection.return_value = results
    det._processor = processor
    det._model = MagicMock(return_value=MagicMock())
    return det


def _two_object_result():
    # person (conf 0.9) and car (conf 0.8) in a 100h x 200w image.
    return [_result(scores=[0.9, 0.8], labels=[0, 2], boxes=[[20.0, 10.0, 120.0, 60.0], [0.0, 0.0, 100.0, 50.0]])]


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
        results = [_result(scores=[0.4, 0.95], labels=[2, 0], boxes=[[0, 0, 10, 10], [0, 0, 20, 20]])]
        det = _detector_with(results)
        objs = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert [o.confidence for o in objs] == [0.95, 0.4]

    def test_boxes_out_of_bounds_are_clamped(self):
        # D-FINE can emit boxes slightly outside the frame; they must clamp to 0..1.
        results = [_result(scores=[0.9], labels=[0], boxes=[[-5.0, -2.0, 220.0, 110.0]])]
        det = _detector_with(results)
        obj = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))[0]
        bb = obj.bounding_box
        assert bb is not None
        assert bb.x == pytest.approx(0.0)
        assert bb.y == pytest.approx(0.0)
        assert bb.x + bb.width == pytest.approx(1.0)
        assert bb.y + bb.height == pytest.approx(1.0)

    def test_class_filter_drops_other_labels(self):
        det = _detector_with(_two_object_result(), class_filter=("person",))
        objs = det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        assert [o.label for o in objs] == ["person"]

    def test_detect_handles_empty_results(self):
        det = _detector_with([_result(scores=[], labels=[], boxes=[])])
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

    def test_confidence_threshold_passed_to_post_process(self):
        det = _detector_with(_two_object_result(), confidence_threshold=0.7)
        det.detect(np.zeros((100, 200, 3), dtype=np.uint8))
        _, kwargs = det._processor.post_process_object_detection.call_args
        assert kwargs["threshold"] == 0.7

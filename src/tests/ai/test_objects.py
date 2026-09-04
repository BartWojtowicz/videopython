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


class TestClassFilterSpellings:
    """D-FINE emits VOC-style names for six COCO classes.

    Passing the standard COCO spelling used to match nothing and draw nothing,
    with no error to explain the silence -- and the class docstring claimed the
    spellings were normalized when no normalization existed.
    """

    def test_standard_coco_spelling_matches_voc_label(self):
        results = [_result(scores=[0.9], labels=[3], boxes=[[0.0, 0.0, 10.0, 10.0]])]
        det = _detector_with(results, class_names={3: "motorbike"}, class_filter=("motorcycle",))
        assert [o.label for o in det.detect(np.zeros((100, 200, 3), dtype=np.uint8))] == ["motorbike"]

    def test_detector_own_spelling_still_matches(self):
        results = [_result(scores=[0.9], labels=[3], boxes=[[0.0, 0.0, 10.0, 10.0]])]
        det = _detector_with(results, class_names={3: "motorbike"}, class_filter=("motorbike",))
        assert [o.label for o in det.detect(np.zeros((100, 200, 3), dtype=np.uint8))] == ["motorbike"]

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("motorcycle", "motorbike"),
            ("airplane", "aeroplane"),
            ("couch", "sofa"),
            ("potted plant", "pottedplant"),
            ("dining table", "diningtable"),
            ("tv", "tvmonitor"),
        ],
    )
    def test_every_diverging_class_is_aliased(self, given, expected):
        assert ObjectDetector(class_filter=(given,)).class_filter == (expected,)

    def test_case_and_spacing_are_normalized(self):
        det = ObjectDetector(class_filter=("Potted  Plant", " TV "))
        assert det.class_filter == ("pottedplant", "tvmonitor")

    def test_unaliased_name_passes_through(self):
        assert ObjectDetector(class_filter=("person",)).class_filter == ("person",)

    def test_unknown_class_is_reported_once_the_model_is_known(self):
        det = _detector_with([], class_names={0: "person"}, class_filter=("person", "sasquatch"))
        assert det.unknown_filter_classes() == ("sasquatch",)

    def test_no_unknown_classes_before_the_model_loads(self):
        det = ObjectDetector(class_filter=("sasquatch",))
        assert det.unknown_filter_classes() == ()

    def test_unknown_class_warns_on_load(self, caplog):
        det = _detector_with([], class_names={0: "person"}, class_filter=("sasquatch",))
        with caplog.at_level("WARNING"):
            det._warn_unknown_filter_classes()
        assert "sasquatch" in caplog.text

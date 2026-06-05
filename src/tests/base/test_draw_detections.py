"""Tests for the AI-free detection overlay renderer."""

import numpy as np

from videopython.base.description import BoundingBox, DetectedObject
from videopython.base.draw_detections import DetectionStyle, class_color, draw_detections


def _frame(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _person(conf: float = 0.9) -> DetectedObject:
    return DetectedObject(
        label="person",
        confidence=conf,
        bounding_box=BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5),
    )


class TestClassColor:
    def test_reserved_class_is_stable(self):
        assert class_color("person") == (76, 175, 80)

    def test_deterministic_across_calls(self):
        assert class_color("zebra") == class_color("zebra")

    def test_distinct_classes_differ(self):
        assert class_color("person") != class_color("car")
        assert class_color("zebra") != class_color("giraffe")

    def test_unknown_class_is_valid_rgb(self):
        r, g, b = class_color("teapot")
        assert all(0 <= c <= 255 for c in (r, g, b))


class TestDrawDetections:
    def test_preserves_shape_and_dtype(self):
        out = draw_detections(_frame(), [_person()])
        assert out.shape == (240, 320, 3)
        assert out.dtype == np.uint8

    def test_empty_list_is_identity(self):
        frame = _frame()
        assert draw_detections(frame, []) is frame

    def test_all_filtered_out_is_identity(self):
        frame = _frame()
        style = DetectionStyle(min_confidence=0.95)
        # Single detection below threshold -> nothing drawn -> same array back.
        assert draw_detections(frame, [_person(conf=0.5)], style) is frame

    def test_box_color_appears_in_output(self):
        out = draw_detections(_frame(), [_person()])
        person = np.array((76, 175, 80))
        assert np.any(np.all(out == person, axis=-1))

    def test_box_color_override(self):
        style = DetectionStyle(box_color=(255, 0, 0), show_confidence=False)
        out = draw_detections(_frame(), [_person()], style)
        assert np.any(np.all(out == np.array((255, 0, 0)), axis=-1))

    def test_off_frame_box_clips_without_raising(self):
        det = DetectedObject(
            label="car",
            confidence=0.8,
            bounding_box=BoundingBox(x=0.8, y=-0.2, width=0.6, height=0.5),
        )
        out = draw_detections(_frame(), [det])
        assert out.shape == (240, 320, 3)

    def test_label_chip_flips_inside_when_box_at_top_edge(self):
        # Box flush with the top edge: chip would overflow above, so it flips
        # to sit inside the box. Render must still succeed and draw something.
        det = DetectedObject(
            label="dog",
            confidence=0.7,
            bounding_box=BoundingBox(x=0.1, y=0.0, width=0.3, height=0.4),
        )
        out = draw_detections(_frame(), [det])
        assert np.any(out != 0)

    def test_does_not_mutate_input(self):
        frame = _frame()
        original = frame.copy()
        draw_detections(frame, [_person()])
        assert np.array_equal(frame, original)

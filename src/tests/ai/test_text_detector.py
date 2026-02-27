"""Unit tests for TextDetector structured OCR output."""

from __future__ import annotations

import numpy as np

from videopython.ai.understanding.detection import TextDetector
from videopython.base.description import DetectedText


def test_detect_detailed_returns_text_regions_without_model_download() -> None:
    """detect_detailed should expose text, confidence, and normalized boxes."""
    detector = TextDetector(languages=["en"])

    class FakeReader:
        def readtext(self, _image):
            return [
                ([[10, 20], [30, 20], [30, 40], [10, 40]], "EXIT", 0.9),
                ([[50, 50], [90, 50], [90, 70], [50, 70]], "A12", 0.8),
            ]

    detector._reader = FakeReader()
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    regions = detector.detect_detailed(image)
    assert len(regions) == 2
    assert all(isinstance(item, DetectedText) for item in regions)
    assert regions[0].text == "EXIT"
    assert regions[0].confidence == 0.9
    assert regions[0].bounding_box is not None
    assert regions[0].bounding_box.x == 0.05
    assert regions[0].bounding_box.y == 0.2
    assert regions[0].bounding_box.width == 0.1
    assert regions[0].bounding_box.height == 0.2

    # Backward-compatible plain-text API should still work.
    assert detector.detect(image) == ["EXIT", "A12"]

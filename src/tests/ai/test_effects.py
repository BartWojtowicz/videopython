"""Tests for AI-powered effects (object detection overlay).

The detector is mocked, so these run without GPU or model weights and stay in
the ai/ suite (the editing suite must not import the optional ``[ai]`` extra).
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np

from videopython.ai.effects import ObjectDetectionOverlay
from videopython.base.description import BoundingBox, DetectedObject
from videopython.base.video import Video, VideoMetadata
from videopython.editing.operation import Operation


def _detection() -> DetectedObject:
    return DetectedObject(
        label="person",
        confidence=0.92,
        bounding_box=BoundingBox(x=0.25, y=0.25, width=0.5, height=0.5),
    )


def _mock_detector(detections):
    inst = MagicMock()
    inst.detect.return_value = detections
    return inst


class TestRegistrationAndSchema:
    def test_registered(self):
        assert Operation.registry().get("object_detection_overlay") is ObjectDetectionOverlay

    def test_llm_exposed(self):
        assert "object_detection_overlay" in Operation.llm_registry()
        assert "object_detection_overlay" in json.dumps(Operation.json_schema())

    def test_backend_is_llm_hidden(self):
        schema = ObjectDetectionOverlay.llm_json_schema()
        props = schema["properties"]
        assert "backend" not in props
        assert "model_size" in props
        assert "confidence_threshold" in props

    def test_predict_metadata_is_identity(self):
        meta = VideoMetadata(height=480, width=640, fps=30, frame_count=60, total_seconds=2.0)
        out = ObjectDetectionOverlay().predict_metadata(meta)
        assert (out.width, out.height, out.frame_count) == (640, 480, 60)


class TestApply:
    @patch("videopython.ai.effects.ObjectDetector")
    def test_apply_preserves_shape_and_draws(self, mock_cls):
        mock_cls.return_value = _mock_detector([_detection()])
        video = Video.from_frames(np.zeros((5, 200, 200, 3), dtype=np.uint8), fps=30)

        out = ObjectDetectionOverlay().apply(video)

        assert out.frame_shape == (200, 200, 3)
        assert len(out.frames) == 5
        assert out.frames.sum() > 0  # boxes were drawn

    @patch("videopython.ai.effects.ObjectDetector")
    def test_no_detections_is_noop(self, mock_cls):
        mock_cls.return_value = _mock_detector([])
        video = Video.from_frames(np.zeros((4, 64, 64, 3), dtype=np.uint8), fps=30)

        out = ObjectDetectionOverlay().apply(video)
        assert out.frames.sum() == 0

    @patch("videopython.ai.effects.ObjectDetector")
    def test_detection_interval_controls_cadence(self, mock_cls):
        inst = _mock_detector([])
        mock_cls.return_value = inst
        video = Video.from_frames(np.zeros((7, 64, 64, 3), dtype=np.uint8), fps=30)

        ObjectDetectionOverlay(detection_interval=3).apply(video)

        # Frames 0, 3, 6 trigger detection; the rest reuse the cache.
        assert inst.detect.call_count == 3

    @patch("videopython.ai.effects.ObjectDetector")
    def test_class_filter_passed_to_detector(self, mock_cls):
        mock_cls.return_value = _mock_detector([])
        video = Video.from_frames(np.zeros((2, 64, 64, 3), dtype=np.uint8), fps=30)

        ObjectDetectionOverlay(class_filter=["person", "car"]).apply(video)

        _, kwargs = mock_cls.call_args
        assert kwargs["class_filter"] == ("person", "car")
        assert kwargs["model_name"] == "yolov8n.pt"

    @patch("videopython.ai.effects.ObjectDetector")
    def test_model_size_maps_to_weights(self, mock_cls):
        mock_cls.return_value = _mock_detector([])
        video = Video.from_frames(np.zeros((1, 64, 64, 3), dtype=np.uint8), fps=30)

        ObjectDetectionOverlay(model_size="m").apply(video)
        _, kwargs = mock_cls.call_args
        assert kwargs["model_name"] == "yolov8m.pt"


class TestStreamingContract:
    @patch("videopython.ai.effects.ObjectDetector")
    def test_streaming_smoke(self, mock_cls):
        mock_cls.return_value = _mock_detector([_detection()])
        eff = ObjectDetectionOverlay(detection_interval=1)
        eff.streaming_init(3, 30.0, 64, 64)

        for i in range(3):
            out = eff.process_frame(np.zeros((64, 64, 3), dtype=np.uint8), i)
            assert out.shape == (64, 64, 3)
            assert out.dtype == np.uint8

        assert eff.streamable is True
        assert eff.requires == ()  # stays in the streaming plan path

    @patch("videopython.ai.effects.ObjectDetector")
    def test_eager_matches_streaming(self, mock_cls):
        # Shared mock instance -> deterministic detections on both paths.
        mock_cls.return_value = _mock_detector([_detection()])
        frames = np.zeros((6, 96, 96, 3), dtype=np.uint8)

        eager = ObjectDetectionOverlay(detection_interval=2).apply(Video.from_frames(frames.copy(), fps=30))

        eff = ObjectDetectionOverlay(detection_interval=2)
        eff.streaming_init(6, 30.0, 96, 96)
        streamed = np.stack([eff.process_frame(frames[i].copy(), i) for i in range(6)])

        np.testing.assert_array_equal(eager.frames, streamed)


class TestPlanValidation:
    def test_validates_in_a_plan(self):
        from videopython.editing.video_edit import VideoEdit

        plan = {
            "segments": [
                {
                    "source": "fake.mp4",
                    "start": 0.0,
                    "end": 2.0,
                    "operations": [
                        {"op": "object_detection_overlay", "class_filter": ["person"], "detection_interval": 2}
                    ],
                }
            ]
        }
        source = VideoMetadata(height=480, width=640, fps=30, frame_count=60, total_seconds=2.0)
        out = VideoEdit.from_dict(plan).validate_with_metadata(source)
        assert (out.width, out.height) == (640, 480)  # shape-preserving

"""Registry-alignment twin of ``editing/test_streamability.py`` for ai ops.

The editing suite must not import the optional ``[ai]`` extra, so its
alignment test only sees ops registered by the editing layer. This twin runs
in the ai suite, where importing the ai op modules registers ``face_crop``
and ``object_detection_overlay``, and re-checks the whole registry.
"""

from videopython.ai.effects import ObjectDetectionOverlay  # noqa: F401  -- registers the op
from videopython.ai.transforms import FaceTrackingCrop  # noqa: F401  -- registers the op
from videopython.editing.effects import Effect
from videopython.editing.operation import Operation


def test_every_registered_op_streams_structurally_including_ai_ops():
    """Every registered op (incl. ai) is streamable by structure.

    The ``streamable`` ClassVar is gone; an op streams iff it overrides
    ``to_ffmpeg_filter`` (a filter op) or ``process_frame`` (a frame effect).
    ``face_crop`` streams via ``to_ffmpeg_filter``; ``object_detection_overlay``
    via ``process_frame``.
    """
    assert "face_crop" in Operation._registry
    assert "object_detection_overlay" in Operation._registry
    for op_id, cls in Operation._registry.items():
        if not cls.__module__.startswith("videopython"):
            continue  # skip test-defined stub ops that pollute the global registry
        overrides_filter = cls.to_ffmpeg_filter is not Operation.to_ffmpeg_filter
        overrides_pf = issubclass(cls, Effect) and cls.process_frame is not Effect.process_frame
        assert overrides_filter or overrides_pf, f"op '{op_id}' streams via neither path"

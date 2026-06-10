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


def test_streamable_flag_matches_filter_compilation_including_ai_ops():
    """Every registered transform must declare ``streamable`` coherently.

    Both ``analyze_streamability`` and ``_build_streaming_plan`` treat the
    ``streamable`` ClassVar as authoritative, but a flag-True transform
    without a working ``to_ffmpeg_filter`` only fails at runtime (the
    strict-mode drift guard), and a flag-False transform with one carries
    dead filter code.
    """
    assert "face_crop" in Operation._registry
    assert "object_detection_overlay" in Operation._registry
    for op_id, cls in Operation._registry.items():
        if issubclass(cls, Effect):
            continue
        overrides_filter = cls.to_ffmpeg_filter is not Operation.to_ffmpeg_filter
        assert overrides_filter == cls.streamable, (
            f"op '{op_id}': streamable={cls.streamable} but "
            f"{'overrides' if overrides_filter else 'does not override'} to_ffmpeg_filter"
        )

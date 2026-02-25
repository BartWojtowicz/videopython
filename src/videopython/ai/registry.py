from __future__ import annotations

from videopython.base.registry import OperationCategory, get_operation_spec, register, spec_from_class


def _register_ai_operations() -> None:
    from videopython.ai.transforms import AutoFramingCrop, FaceTrackingCrop, SplitScreenComposite

    specs = [
        spec_from_class(
            FaceTrackingCrop,
            op_id="face_crop",
            category=OperationCategory.TRANSFORMATION,
            tags={"requires_faces", "changes_dimensions"},
            aliases=("face_tracking_crop",),
            metadata_method="crop_to_aspect_even",
        ),
        spec_from_class(
            AutoFramingCrop,
            op_id="auto_framing",
            category=OperationCategory.TRANSFORMATION,
            tags={"requires_faces", "changes_dimensions"},
            aliases=("auto_framing_crop",),
            metadata_method="crop_to_aspect_even",
        ),
        spec_from_class(
            SplitScreenComposite,
            op_id="split_screen",
            category=OperationCategory.TRANSFORMATION,
            tags={"requires_faces", "multi_source", "changes_dimensions"},
            aliases=("split_screen_composite",),
        ),
    ]

    for spec in specs:
        # Guard against repeated AI imports when base registry state persists.
        if get_operation_spec(spec.id) is None:
            register(spec)


_register_ai_operations()

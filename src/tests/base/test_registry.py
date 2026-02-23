"""Tests for operation registry metadata and registration behavior."""

from __future__ import annotations

import importlib
import inspect
import re
import sys
from typing import Any

import pytest

from videopython.base.combine import StackVideos
from videopython.base.effects import Blur, FullImageOverlay, KenBurns
from videopython.base.text.overlay import TranscriptionOverlay
from videopython.base.transforms import CutSeconds, PictureInPicture
from videopython.base.transitions import FadeTransition

BASE_OPERATION_IDS = {
    "cut_frames",
    "cut",
    "resize",
    "resample_fps",
    "crop",
    "speed_change",
    "picture_in_picture",
    "blur_effect",
    "zoom_effect",
    "color_adjust",
    "vignette",
    "ken_burns",
    "full_image_overlay",
    "instant_transition",
    "fade_transition",
    "blur_transition",
    "stack_videos",
    "add_subtitles",
}

AI_OPERATION_IDS = {
    "face_crop",
    "auto_framing",
    "split_screen",
}


def _reload_registry() -> Any:
    module = importlib.import_module("videopython.base.registry")
    return importlib.reload(module)


def _drop_ai_modules() -> None:
    for name in list(sys.modules):
        if name == "videopython.ai" or name.startswith("videopython.ai."):
            sys.modules.pop(name)


def test_base_operations_registered_after_import_videopython_base() -> None:
    registry = _reload_registry()
    specs = registry.get_operation_specs()

    assert len(specs) >= 15
    assert BASE_OPERATION_IDS.issubset(set(specs))


def test_every_spec_has_valid_fields() -> None:
    registry = _reload_registry()

    for spec in registry.get_operation_specs().values():
        assert spec.id
        assert spec.class_name
        assert spec.module_path
        assert spec.description
        assert isinstance(spec.category, registry.OperationCategory)


def test_all_ids_are_snake_case() -> None:
    registry = _reload_registry()
    pattern = re.compile(r"^[a-z][a-z0-9_]*$")

    for op_id in registry.get_operation_specs():
        assert pattern.match(op_id)


def test_to_json_schema_returns_object_schema_for_every_operation() -> None:
    registry = _reload_registry()

    for spec in registry.get_operation_specs().values():
        schema = spec.to_json_schema()
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert isinstance(schema.get("properties"), dict)
        assert isinstance(schema.get("required"), list)
        assert schema.get("additionalProperties") is False


def test_to_apply_json_schema_returns_object_schema_for_every_operation() -> None:
    registry = _reload_registry()

    for spec in registry.get_operation_specs().values():
        schema = spec.to_apply_json_schema()
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"
        assert isinstance(schema.get("properties"), dict)
        assert isinstance(schema.get("required"), list)
        assert schema.get("additionalProperties") is False


def test_register_raises_on_duplicate_id() -> None:
    registry = _reload_registry()
    existing_spec = next(iter(registry.get_operation_specs().values()))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(existing_spec)


def test_no_alias_collides_with_another_operation_id() -> None:
    registry = _reload_registry()
    specs = registry.get_operation_specs()
    ids = set(specs)
    seen_aliases: set[str] = set()

    for spec in specs.values():
        for alias in spec.aliases:
            assert alias not in ids
            assert alias not in seen_aliases
            resolved = registry.get_operation_spec(alias)
            assert resolved is not None
            assert resolved.id == spec.id
            seen_aliases.add(alias)


def test_spec_from_class_introspects_cut_seconds() -> None:
    registry = _reload_registry()

    spec = registry.spec_from_class(
        CutSeconds,
        op_id="tmp_cut_seconds",
        category=registry.OperationCategory.TRANSFORMATION,
    )

    assert [param.name for param in spec.params] == ["start", "end"]
    assert all(param.required for param in spec.params)
    assert all(param.json_type == "number" for param in spec.params)
    assert spec.apply_params == ()


def test_effect_apply_schema_contains_start_stop() -> None:
    registry = _reload_registry()

    spec = registry.get_operation_spec("blur_effect")
    assert spec is not None
    assert [param.name for param in spec.apply_params] == ["start", "stop"]
    assert all(not param.required for param in spec.apply_params)
    assert all(param.json_type == "number" for param in spec.apply_params)
    assert all(param.nullable for param in spec.apply_params)

    apply_schema = spec.to_apply_json_schema()
    assert set(apply_schema["properties"]) == {"start", "stop"}
    assert apply_schema["required"] == []
    assert apply_schema["properties"]["start"]["type"] == ["number", "null"]
    assert apply_schema["properties"]["stop"]["type"] == ["number", "null"]
    assert apply_schema["properties"]["start"]["minimum"] == 0
    assert apply_schema["properties"]["stop"]["minimum"] == 0


def test_selected_base_schema_constraints_are_present() -> None:
    registry = _reload_registry()

    cut_spec = registry.get_operation_spec("cut")
    assert cut_spec is not None
    cut_schema = cut_spec.to_json_schema()
    assert cut_schema["properties"]["start"]["minimum"] == 0
    assert cut_schema["properties"]["end"]["minimum"] == 0

    speed_spec = registry.get_operation_spec("speed_change")
    assert speed_spec is not None
    speed_schema = speed_spec.to_json_schema()
    assert speed_schema["properties"]["speed"]["exclusiveMinimum"] == 0
    assert speed_schema["properties"]["end_speed"]["exclusiveMinimum"] == 0

    crop_spec = registry.get_operation_spec("crop")
    assert crop_spec is not None
    crop_schema = crop_spec.to_json_schema()
    assert crop_schema["properties"]["width"]["exclusiveMinimum"] == 0
    assert crop_schema["properties"]["height"]["exclusiveMinimum"] == 0


def test_category_filtering_returns_expected_subset() -> None:
    registry = _reload_registry()

    transformations = registry.get_specs_by_category(registry.OperationCategory.TRANSFORMATION)

    assert "cut" in transformations
    assert "resize" in transformations
    assert "blur_effect" not in transformations


def test_tag_filtering_returns_matching_operations() -> None:
    registry = _reload_registry()

    dimensions_specs = registry.get_specs_by_tag("changes_dimensions")

    assert {"resize", "crop", "stack_videos"}.issubset(set(dimensions_specs))


def test_ai_ops_appear_only_after_import_videopython_ai() -> None:
    _drop_ai_modules()
    registry = _reload_registry()

    specs_before = registry.get_operation_specs()
    assert AI_OPERATION_IDS.isdisjoint(set(specs_before))

    importlib.import_module("videopython.ai")

    specs_after = registry.get_operation_specs()
    assert AI_OPERATION_IDS.issubset(set(specs_after))


@pytest.mark.parametrize(
    ("op_id", "cls", "excluded_params"),
    [
        ("cut", CutSeconds, set()),
        ("picture_in_picture", PictureInPicture, {"overlay"}),
        ("ken_burns", KenBurns, {"start_region", "end_region"}),
        ("full_image_overlay", FullImageOverlay, {"overlay_image"}),
        ("stack_videos", StackVideos, set()),
    ],
)
def test_paramspec_count_matches_init_signature(
    op_id: str,
    cls: type[Any],
    excluded_params: set[str],
) -> None:
    registry = _reload_registry()
    spec = registry.get_operation_spec(op_id)

    assert spec is not None

    constructor_params = [
        name for name in inspect.signature(cls.__init__).parameters if name != "self" and name not in excluded_params
    ]
    assert len(spec.params) == len(constructor_params)


@pytest.mark.parametrize(
    ("op_id", "cls", "excluded_apply_params"),
    [
        ("cut", CutSeconds, {"video"}),
        ("blur_effect", Blur, {"video"}),
        ("fade_transition", FadeTransition, {"videos"}),
        ("add_subtitles", TranscriptionOverlay, {"video", "transcription"}),
    ],
)
def test_paramspec_count_matches_apply_signature(
    op_id: str,
    cls: type[Any],
    excluded_apply_params: set[str],
) -> None:
    registry = _reload_registry()
    spec = registry.get_operation_spec(op_id)

    assert spec is not None

    apply_params = [
        name for name in inspect.signature(cls.apply).parameters if name != "self" and name not in excluded_apply_params
    ]
    assert len(spec.apply_params) == len(apply_params)

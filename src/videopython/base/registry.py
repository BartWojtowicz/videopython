from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from enum import Enum
from types import UnionType
from typing import Any, Iterable, Literal, Mapping, Union, get_args, get_origin, get_type_hints

__all__ = [
    "OperationCategory",
    "ParamSpec",
    "OperationSpec",
    "register",
    "get_operation_specs",
    "get_operation_spec",
    "get_specs_by_category",
    "get_specs_by_tag",
    "spec_from_class",
]


class OperationCategory(str, Enum):
    """Operation execution categories used by downstream executors."""

    TRANSFORMATION = "transformation"
    EFFECT = "effect"
    TRANSITION = "transition"
    SPECIAL = "special"


class _UnsetType:
    pass


UNSET = _UnsetType()


@dataclass(frozen=True)
class ParamSpec:
    """Machine-readable schema metadata for a parameter."""

    name: str
    json_type: str
    description: str
    required: bool
    default: Any = UNSET
    enum: tuple[Any, ...] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    items_type: str | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert parameter metadata into a JSON Schema fragment."""
        schema: dict[str, Any] = {
            "type": self.json_type,
            "description": self.description,
        }

        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.json_type == "array" and self.items_type is not None:
            schema["items"] = {"type": self.items_type}
        if self.default is not UNSET:
            schema["default"] = self.default

        return schema


@dataclass(frozen=True)
class OperationSpec:
    """Machine-readable operation description exported by videopython."""

    id: str
    class_name: str
    module_path: str
    category: OperationCategory
    description: str
    params: tuple[ParamSpec, ...]
    apply_params: tuple[ParamSpec, ...] = ()
    tags: frozenset[str] = frozenset()
    aliases: tuple[str, ...] = ()
    metadata_method: str | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Generate constructor-args JSON schema for this operation."""
        return _params_to_json_schema(self.params)

    def to_apply_json_schema(self) -> dict[str, Any]:
        """Generate apply-args JSON schema for this operation."""
        return _params_to_json_schema(self.apply_params)


def _params_to_json_schema(params: tuple[ParamSpec, ...]) -> dict[str, Any]:
    properties = {param.name: param.to_json_schema() for param in params}
    required = [param.name for param in params if param.required]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _spec_params_from_callable(
    fn: Any,
    *,
    exclude_params: Iterable[str],
    param_overrides: Mapping[str, Mapping[str, Any]],
    description_prefix: str,
) -> tuple[ParamSpec, ...]:
    excluded = set(exclude_params)
    signature = inspect.signature(fn)

    module = inspect.getmodule(fn)
    type_hints: dict[str, Any] = {}
    try:
        globalns = vars(module) if module is not None else None
        type_hints = get_type_hints(fn, globalns=globalns, localns=globalns)
    except (AttributeError, NameError, TypeError):
        type_hints = {}

    params: list[ParamSpec] = []
    for parameter in signature.parameters.values():
        if parameter.name == "self" or parameter.name in excluded:
            continue

        annotation = type_hints.get(parameter.name, parameter.annotation)
        required = parameter.default is inspect.Signature.empty
        default = UNSET if required else _to_json_value(parameter.default)
        json_type, enum_values, items_type = _annotation_to_schema(annotation)

        param_spec = ParamSpec(
            name=parameter.name,
            json_type=json_type,
            description=f"{description_prefix} argument '{parameter.name}'.",
            required=required,
            default=default,
            enum=enum_values,
            items_type=items_type,
        )

        override = param_overrides.get(parameter.name)
        if override is not None:
            param_spec = replace(param_spec, **override)

        params.append(param_spec)

    return tuple(params)


_REGISTRY: dict[str, OperationSpec] = {}
_ALIAS_TO_ID: dict[str, str] = {}


def register(spec: OperationSpec) -> None:
    """Register an operation specification.

    Raises:
        ValueError: If id or alias collides with an existing operation id/alias.
    """
    if spec.id in _REGISTRY:
        raise ValueError(f"Operation id '{spec.id}' is already registered")
    if spec.id in _ALIAS_TO_ID:
        current_id = _ALIAS_TO_ID[spec.id]
        raise ValueError(f"Operation id '{spec.id}' collides with alias for '{current_id}'")

    for alias in spec.aliases:
        if alias == spec.id:
            raise ValueError(f"Alias '{alias}' cannot be the same as operation id")
        if alias in _REGISTRY:
            raise ValueError(f"Alias '{alias}' collides with existing operation id")
        if alias in _ALIAS_TO_ID:
            current_id = _ALIAS_TO_ID[alias]
            raise ValueError(f"Alias '{alias}' already points to '{current_id}'")

    _REGISTRY[spec.id] = spec
    for alias in spec.aliases:
        _ALIAS_TO_ID[alias] = spec.id


def get_operation_specs() -> dict[str, OperationSpec]:
    """Return a copy of all registered operation specs."""
    return dict(_REGISTRY)


def get_operation_spec(op_id: str) -> OperationSpec | None:
    """Lookup operation by id or alias."""
    spec = _REGISTRY.get(op_id)
    if spec is not None:
        return spec

    alias_target = _ALIAS_TO_ID.get(op_id)
    if alias_target is None:
        return None
    return _REGISTRY.get(alias_target)


def get_specs_by_category(category: OperationCategory) -> dict[str, OperationSpec]:
    """Return specs filtered by category."""
    return {op_id: spec for op_id, spec in _REGISTRY.items() if spec.category == category}


def get_specs_by_tag(tag: str) -> dict[str, OperationSpec]:
    """Return specs that contain the given capability tag."""
    return {op_id: spec for op_id, spec in _REGISTRY.items() if tag in spec.tags}


def spec_from_class(
    cls: type[Any],
    *,
    op_id: str,
    category: OperationCategory,
    tags: Iterable[str] | None = None,
    aliases: Iterable[str] | None = None,
    exclude_params: Iterable[str] | None = None,
    param_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    exclude_apply_params: Iterable[str] | None = None,
    apply_param_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    metadata_method: str | None = None,
) -> OperationSpec:
    """Build an operation spec from constructor and apply method type hints."""
    constructor_params = _spec_params_from_callable(
        cls.__init__,
        exclude_params=exclude_params or (),
        param_overrides=param_overrides or {},
        description_prefix="Constructor",
    )
    # Default exclusions cover known first-argument media inputs for current ops.
    # Callers can override this when introducing non-standard apply signatures.
    apply_exclusions = (
        set(exclude_apply_params) if exclude_apply_params is not None else {"video", "videos", "transcription"}
    )
    apply_params = _spec_params_from_callable(
        cls.apply,
        exclude_params=apply_exclusions,
        param_overrides=apply_param_overrides or {},
        description_prefix="Apply",
    )

    description = _first_line(inspect.getdoc(cls))
    if not description:
        description = f"{cls.__name__} operation."

    return OperationSpec(
        id=op_id,
        class_name=cls.__name__,
        module_path=cls.__module__,
        category=category,
        description=description,
        params=constructor_params,
        apply_params=apply_params,
        tags=frozenset(tags or ()),
        aliases=tuple(aliases or ()),
        metadata_method=metadata_method,
    )


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_json_value(item) for item in value]
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_to_json_value(item) for item in value]
    return value


def _first_line(docstring: str | None) -> str:
    if not docstring:
        return ""
    return docstring.strip().splitlines()[0].strip()


def _annotation_to_schema(annotation: Any) -> tuple[str, tuple[Any, ...] | None, str | None]:
    if annotation is inspect.Signature.empty:
        return "object", None, None

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = tuple(_to_json_value(value) for value in args)
        literal_type = _json_type_from_python_types({type(value) for value in values})
        return literal_type, values, None

    if _is_union(origin):
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return _annotation_to_schema(non_none_args[0])

        member_types = {_annotation_to_schema(arg)[0] for arg in non_none_args}
        if member_types <= {"integer", "number"}:
            return "number", None, None
        if len(member_types) == 1:
            return member_types.pop(), None, None
        return "object", None, None

    if origin in (list, set, frozenset):
        items_type = _items_type_from_args(args)
        return "array", None, items_type

    if origin is tuple:
        items_type = _items_type_from_args(args)
        return "array", None, items_type

    if origin is dict:
        return "object", None, None

    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        enum_values = tuple(_to_json_value(item.value) for item in annotation)
        enum_type = _json_type_from_python_types({type(value) for value in enum_values})
        return enum_type, enum_values, None

    if annotation is bool:
        return "boolean", None, None
    if annotation is int:
        return "integer", None, None
    if annotation is float:
        return "number", None, None
    if annotation is str:
        return "string", None, None

    if annotation in (Any, object):
        return "object", None, None

    return "object", None, None


def _is_union(origin: Any) -> bool:
    return origin in (Union, UnionType)


def _items_type_from_args(args: tuple[Any, ...]) -> str:
    if not args:
        return "object"

    candidate_args = [arg for arg in args if arg is not Ellipsis]
    if not candidate_args:
        return "object"

    item_types = {_annotation_to_schema(arg)[0] for arg in candidate_args}
    if item_types <= {"integer", "number"}:
        return "number"
    if len(item_types) == 1:
        return item_types.pop()
    return "object"


def _json_type_from_python_types(types: set[type[Any]]) -> str:
    if types == {int}:
        return "integer"
    if types <= {int, float}:
        return "number"
    if types == {bool}:
        return "boolean"
    if types == {str}:
        return "string"
    return "object"


def _register_base_operations() -> None:
    from videopython.base.combine import StackVideos
    from videopython.base.effects import Blur, ColorGrading, FullImageOverlay, KenBurns, Vignette, Zoom
    from videopython.base.text.overlay import TranscriptionOverlay
    from videopython.base.transforms import (
        Crop,
        CutFrames,
        CutSeconds,
        PictureInPicture,
        ResampleFPS,
        Resize,
        SpeedChange,
    )
    from videopython.base.transitions import BlurTransition, FadeTransition, InstantTransition

    register(
        spec_from_class(
            CutFrames,
            op_id="cut_frames",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_duration"},
            metadata_method="cut_frames",
        )
    )
    register(
        spec_from_class(
            CutSeconds,
            op_id="cut",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_duration"},
            aliases=("cut_seconds",),
            metadata_method="cut",
        )
    )
    register(
        spec_from_class(
            Resize,
            op_id="resize",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_dimensions"},
            metadata_method="resize",
        )
    )
    register(
        spec_from_class(
            ResampleFPS,
            op_id="resample_fps",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_fps"},
            metadata_method="resample_fps",
        )
    )
    register(
        spec_from_class(
            Crop,
            op_id="crop",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_dimensions"},
            metadata_method="crop",
        )
    )
    register(
        spec_from_class(
            SpeedChange,
            op_id="speed_change",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_duration"},
            metadata_method="speed_change",
        )
    )
    register(
        spec_from_class(
            PictureInPicture,
            op_id="picture_in_picture",
            category=OperationCategory.TRANSFORMATION,
            tags={"multi_source"},
            exclude_params={"overlay"},
        )
    )
    register(
        spec_from_class(
            Blur,
            op_id="blur_effect",
            category=OperationCategory.EFFECT,
            aliases=("blur",),
        )
    )
    register(
        spec_from_class(
            Zoom,
            op_id="zoom_effect",
            category=OperationCategory.EFFECT,
            aliases=("zoom",),
        )
    )
    register(
        spec_from_class(
            ColorGrading,
            op_id="color_adjust",
            category=OperationCategory.EFFECT,
            aliases=("color_grading",),
        )
    )
    register(
        spec_from_class(
            Vignette,
            op_id="vignette",
            category=OperationCategory.EFFECT,
        )
    )
    register(
        spec_from_class(
            KenBurns,
            op_id="ken_burns",
            category=OperationCategory.EFFECT,
            exclude_params={"start_region", "end_region"},
        )
    )
    register(
        spec_from_class(
            FullImageOverlay,
            op_id="full_image_overlay",
            category=OperationCategory.EFFECT,
            exclude_params={"overlay_image"},
        )
    )
    register(
        spec_from_class(
            InstantTransition,
            op_id="instant_transition",
            category=OperationCategory.TRANSITION,
            tags={"multi_source_only"},
        )
    )
    register(
        spec_from_class(
            FadeTransition,
            op_id="fade_transition",
            category=OperationCategory.TRANSITION,
            tags={"changes_duration", "multi_source_only"},
        )
    )
    register(
        spec_from_class(
            BlurTransition,
            op_id="blur_transition",
            category=OperationCategory.TRANSITION,
            tags={"changes_duration", "multi_source_only"},
        )
    )
    register(
        spec_from_class(
            StackVideos,
            op_id="stack_videos",
            category=OperationCategory.SPECIAL,
            tags={"multi_source_only", "changes_dimensions"},
        )
    )
    register(
        spec_from_class(
            TranscriptionOverlay,
            op_id="add_subtitles",
            category=OperationCategory.SPECIAL,
            tags={"requires_transcript"},
            aliases=("transcription_overlay",),
        )
    )


# Base ops are registered unconditionally on module import; module reload recreates
# the registry dictionaries before this runs.
_register_base_operations()

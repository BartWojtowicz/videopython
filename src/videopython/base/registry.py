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
    exclusive_minimum: int | float | None = None
    exclusive_maximum: int | float | None = None
    items_type: str | None = None
    nullable: bool = False

    def to_json_schema(self) -> dict[str, Any]:
        """Convert parameter metadata into a JSON Schema fragment."""
        schema_type: str | list[str] = self.json_type
        if self.nullable:
            schema_type = [self.json_type, "null"]

        schema: dict[str, Any] = {
            "type": schema_type,
            "description": self.description,
        }

        if self.enum is not None:
            schema["enum"] = list(self.enum)
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.exclusive_minimum is not None:
            schema["exclusiveMinimum"] = self.exclusive_minimum
        if self.exclusive_maximum is not None:
            schema["exclusiveMaximum"] = self.exclusive_maximum
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

    docstring_args = _parse_google_docstring_args(inspect.getdoc(fn))

    params: list[ParamSpec] = []
    for parameter in signature.parameters.values():
        if parameter.name == "self" or parameter.name in excluded:
            continue

        annotation = type_hints.get(parameter.name, parameter.annotation)
        required = parameter.default is inspect.Signature.empty
        default = UNSET if required else _to_json_value(parameter.default)
        json_type, enum_values, items_type, nullable = _annotation_to_schema(annotation)

        doc_desc = docstring_args.get(parameter.name)
        description = doc_desc if doc_desc else f"{description_prefix} argument '{parameter.name}'."

        param_spec = ParamSpec(
            name=parameter.name,
            json_type=json_type,
            description=description,
            required=required,
            default=default,
            enum=enum_values,
            items_type=items_type,
            nullable=nullable,
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
    description: str | None = None,
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

    if description is None:
        description = _clean_docstring(inspect.getdoc(cls))
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


def _clean_docstring(docstring: str | None) -> str:
    """Return full docstring with Args/Returns/Raises sections stripped."""
    if not docstring:
        return ""
    lines = docstring.strip().splitlines()
    result: list[str] = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped.rstrip(":") in ("Args", "Returns", "Raises", "Attributes", "Examples", "Note", "Notes"):
            skip = True
            continue
        if skip:
            # Non-indented line (or blank after a section) means a new top-level block.
            if stripped and not line[0].isspace():
                skip = False
            else:
                continue
        if not skip:
            result.append(stripped)
    text = " ".join(result).strip()
    # Collapse multiple spaces from joining lines.
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def _parse_google_docstring_args(docstring: str | None) -> dict[str, str]:
    """Extract {param_name: description} from Google-style Args section."""
    if not docstring:
        return {}
    lines = docstring.strip().splitlines()
    in_args = False
    args: dict[str, str] = {}
    current_name: str | None = None
    current_desc_parts: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Detect start of Args: section
        if stripped == "Args:":
            in_args = True
            continue
        if not in_args:
            continue
        # A non-indented non-blank line ends the Args section
        if stripped and not line[0].isspace():
            break
        if not stripped:
            continue
        # Check if this is a new parameter line (name: description) or
        # (name (type): description). Param lines are indented once,
        # continuation lines are indented further.
        # Heuristic: param line contains ": " and is at the first indent level.
        lstripped = line.lstrip()
        indent = len(line) - len(lstripped)
        if ": " in lstripped and indent <= 12:
            # Save previous param
            if current_name is not None:
                args[current_name] = " ".join(current_desc_parts).strip()
            name_part, _, desc = lstripped.partition(": ")
            # Strip type annotation in parens: "name (type)" -> "name"
            current_name = name_part.split("(")[0].split(":")[0].strip()
            current_desc_parts = [desc] if desc else []
        elif current_name is not None:
            # Continuation of previous param description
            current_desc_parts.append(stripped)

    if current_name is not None:
        args[current_name] = " ".join(current_desc_parts).strip()

    return args


def _annotation_to_schema(annotation: Any) -> tuple[str, tuple[Any, ...] | None, str | None, bool]:
    if annotation is inspect.Signature.empty:
        return "object", None, None, False

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        values = tuple(_to_json_value(value) for value in args)
        literal_type = _json_type_from_python_types({type(value) for value in values})
        return literal_type, values, None, False

    if _is_union(origin):
        nullable = any(arg is type(None) for arg in args)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            json_type, enum_values, items_type, _ = _annotation_to_schema(non_none_args[0])
            return json_type, enum_values, items_type, nullable

        member_types = {_annotation_to_schema(arg)[0] for arg in non_none_args}
        if member_types <= {"integer", "number"}:
            return "number", None, None, nullable
        if len(member_types) == 1:
            return member_types.pop(), None, None, nullable
        return "object", None, None, nullable

    if origin in (list, set, frozenset):
        items_type = _items_type_from_args(args)
        return "array", None, items_type, False

    if origin is tuple:
        items_type = _items_type_from_args(args)
        return "array", None, items_type, False

    if origin is dict:
        return "object", None, None, False

    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        enum_values = tuple(_to_json_value(item.value) for item in annotation)
        enum_type = _json_type_from_python_types({type(value) for value in enum_values})
        return enum_type, enum_values, None, False

    if annotation is bool:
        return "boolean", None, None, False
    if annotation is int:
        return "integer", None, None, False
    if annotation is float:
        return "number", None, None, False
    if annotation is str:
        return "string", None, None, False

    if annotation in (Any, object):
        return "object", None, None, False

    return "object", None, None, False


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
    from videopython.base.effects import (
        Blur,
        ColorGrading,
        Fade,
        FullImageOverlay,
        KenBurns,
        TextOverlay,
        Vignette,
        VolumeAdjust,
        Zoom,
    )
    from videopython.base.text.overlay import TranscriptionOverlay
    from videopython.base.transforms import (
        Crop,
        CutFrames,
        CutSeconds,
        FreezeFrame,
        PictureInPicture,
        ResampleFPS,
        Resize,
        Reverse,
        SilenceRemoval,
        SpeedChange,
    )
    from videopython.base.transitions import BlurTransition, FadeTransition, InstantTransition

    # Common apply-param overrides: only numeric constraints (descriptions come from
    # the Effect.apply / AudioEffect.apply docstrings).
    _time_range_apply_overrides = {
        "start": {"minimum": 0},
        "stop": {"minimum": 0},
    }

    # -- Transformations --

    register(
        spec_from_class(
            CutFrames,
            op_id="cut_frames",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_duration"},
            param_overrides={
                "start": {"minimum": 0},
                "end": {"minimum": 0},
            },
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
            param_overrides={
                "start": {"minimum": 0},
                "end": {"minimum": 0},
            },
            metadata_method="cut",
        )
    )
    register(
        spec_from_class(
            Resize,
            op_id="resize",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_dimensions"},
            param_overrides={
                "width": {"exclusive_minimum": 0},
                "height": {"exclusive_minimum": 0},
            },
            metadata_method="resize",
        )
    )
    register(
        spec_from_class(
            ResampleFPS,
            op_id="resample_fps",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_fps"},
            param_overrides={"fps": {"minimum": 1}},
            metadata_method="resample_fps",
        )
    )
    register(
        spec_from_class(
            Crop,
            op_id="crop",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_dimensions"},
            param_overrides={
                "width": {"exclusive_minimum": 0},
                "height": {"exclusive_minimum": 0},
            },
            metadata_method="crop",
        )
    )
    register(
        spec_from_class(
            SpeedChange,
            op_id="speed_change",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_duration"},
            param_overrides={
                "speed": {"exclusive_minimum": 0},
                "end_speed": {"exclusive_minimum": 0},
            },
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
            param_overrides={
                "scale": {"exclusive_minimum": 0, "maximum": 1},
                "border_width": {"minimum": 0},
                "corner_radius": {"minimum": 0},
                "opacity": {"minimum": 0, "maximum": 1},
            },
        )
    )
    register(
        spec_from_class(
            Reverse,
            op_id="reverse",
            category=OperationCategory.TRANSFORMATION,
            metadata_method="reverse",
        )
    )
    register(
        spec_from_class(
            SilenceRemoval,
            op_id="silence_removal",
            category=OperationCategory.TRANSFORMATION,
            tags={"requires_transcript", "changes_duration"},
            aliases=("remove_silence", "jump_cut"),
            param_overrides={
                "min_silence_duration": {"exclusive_minimum": 0},
                "padding": {"minimum": 0},
                "speed_factor": {"exclusive_minimum": 1},
            },
            metadata_method="silence_removal",
        )
    )
    register(
        spec_from_class(
            FreezeFrame,
            op_id="freeze_frame",
            category=OperationCategory.TRANSFORMATION,
            tags={"changes_duration"},
            aliases=("freeze",),
            param_overrides={
                "timestamp": {"minimum": 0},
                "duration": {"exclusive_minimum": 0},
            },
            metadata_method="freeze_frame",
        )
    )

    # -- Effects --

    register(
        spec_from_class(
            Blur,
            op_id="blur_effect",
            category=OperationCategory.EFFECT,
            aliases=("blur",),
            param_overrides={"iterations": {"minimum": 1}},
            apply_param_overrides=_time_range_apply_overrides,
        )
    )
    register(
        spec_from_class(
            Zoom,
            op_id="zoom_effect",
            category=OperationCategory.EFFECT,
            aliases=("zoom",),
            param_overrides={"zoom_factor": {"exclusive_minimum": 1}},
            apply_param_overrides=_time_range_apply_overrides,
        )
    )
    register(
        spec_from_class(
            ColorGrading,
            op_id="color_adjust",
            category=OperationCategory.EFFECT,
            aliases=("color_grading",),
            param_overrides={
                "brightness": {"minimum": -1, "maximum": 1},
                "contrast": {"minimum": 0.5, "maximum": 2.0},
                "saturation": {"minimum": 0, "maximum": 2.0},
                "temperature": {"minimum": -1, "maximum": 1},
            },
            apply_param_overrides=_time_range_apply_overrides,
        )
    )
    register(
        spec_from_class(
            Vignette,
            op_id="vignette",
            category=OperationCategory.EFFECT,
            param_overrides={
                "strength": {"minimum": 0, "maximum": 1},
                "radius": {"minimum": 0.5, "maximum": 2.0},
            },
            apply_param_overrides=_time_range_apply_overrides,
        )
    )
    register(
        spec_from_class(
            KenBurns,
            op_id="ken_burns",
            category=OperationCategory.EFFECT,
            exclude_params={"start_region", "end_region"},
            # BoundingBox forward ref breaks get_type_hints, so fix easing type manually.
            param_overrides={
                "easing": {
                    "json_type": "string",
                    "enum": ("linear", "ease_in", "ease_out", "ease_in_out"),
                },
            },
            apply_param_overrides=_time_range_apply_overrides,
        )
    )
    register(
        spec_from_class(
            FullImageOverlay,
            op_id="full_image_overlay",
            category=OperationCategory.EFFECT,
            exclude_params={"overlay_image"},
            param_overrides={
                "alpha": {"minimum": 0, "maximum": 1},
                "fade_time": {"minimum": 0},
            },
            apply_param_overrides=_time_range_apply_overrides,
        )
    )
    register(
        spec_from_class(
            Fade,
            op_id="fade",
            category=OperationCategory.EFFECT,
            param_overrides={"duration": {"exclusive_minimum": 0}},
            apply_param_overrides=_time_range_apply_overrides,
        )
    )
    register(
        spec_from_class(
            VolumeAdjust,
            op_id="volume_adjust",
            category=OperationCategory.EFFECT,
            aliases=("volume",),
            param_overrides={
                "volume": {"minimum": 0},
                "ramp_duration": {"minimum": 0},
            },
            apply_param_overrides=_time_range_apply_overrides,
        )
    )
    register(
        spec_from_class(
            TextOverlay,
            op_id="text_overlay",
            category=OperationCategory.EFFECT,
            aliases=("lower_third", "title_card"),
            exclude_params={"font_filename"},
            param_overrides={
                "font_size": {"minimum": 1},
                "background_padding": {"minimum": 0},
                "max_width": {"exclusive_minimum": 0, "maximum": 1},
            },
            apply_param_overrides=_time_range_apply_overrides,
        )
    )

    # -- Transitions --

    register(
        spec_from_class(
            InstantTransition,
            op_id="instant_transition",
            category=OperationCategory.TRANSITION,
            tags={"multi_source_only"},
            exclude_params={"args", "kwargs"},
        )
    )
    register(
        spec_from_class(
            FadeTransition,
            op_id="fade_transition",
            category=OperationCategory.TRANSITION,
            tags={"changes_duration", "multi_source_only"},
            param_overrides={"effect_time_seconds": {"exclusive_minimum": 0}},
        )
    )
    register(
        spec_from_class(
            BlurTransition,
            op_id="blur_transition",
            category=OperationCategory.TRANSITION,
            tags={"changes_duration", "multi_source_only"},
            param_overrides={
                "effect_time_seconds": {"exclusive_minimum": 0},
                "blur_iterations": {"minimum": 1},
            },
        )
    )

    # -- Special --

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
            param_overrides={
                "font_size": {"minimum": 1},
                "font_border_size": {"minimum": 0},
                "background_padding": {"minimum": 0},
                "highlight_size_multiplier": {"exclusive_minimum": 0},
            },
        )
    )


# Base ops are registered unconditionally on module import; module reload recreates
# the registry dictionaries before this runs.
_register_base_operations()

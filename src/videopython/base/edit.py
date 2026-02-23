from __future__ import annotations

import copy
import importlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from videopython.base.effects import Effect
from videopython.base.registry import OperationCategory, OperationSpec, ParamSpec, get_operation_spec
from videopython.base.transforms import Transformation
from videopython.base.video import Video, VideoMetadata

__all__ = [
    "SegmentConfig",
    "VideoEdit",
]


@dataclass(frozen=True)
class _StepRecord:
    """Canonical step data + live operation instance.

    `args` and `apply_args` are snapshots of parsed input (deep copied) and are the
    source of truth for serialization.
    """

    op_id: str
    args: dict[str, Any]
    apply_args: dict[str, Any]
    operation: Transformation | Effect

    @classmethod
    def create(
        cls,
        op_id: str,
        args: Mapping[str, Any] | None,
        apply_args: Mapping[str, Any] | None,
        operation: Transformation | Effect,
    ) -> _StepRecord:
        return cls(
            op_id=op_id,
            args=copy.deepcopy(dict(args or {})),
            apply_args=copy.deepcopy(dict(apply_args or {})),
            operation=operation,
        )


@dataclass
class SegmentConfig:
    """Configuration for a single video segment in an editing plan."""

    source_video: Path
    start_second: float
    end_second: float
    transform_records: tuple[_StepRecord, ...] = field(default_factory=tuple)
    effect_records: tuple[_StepRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.transform_records = tuple(self.transform_records)
        self.effect_records = tuple(self.effect_records)
        for record in self.transform_records:
            if not isinstance(record.operation, Transformation):
                raise TypeError(
                    "SegmentConfig.transform_records must contain "
                    f"Transformation operations, got {type(record.operation)}"
                )
        for record in self.effect_records:
            if not isinstance(record.operation, Effect):
                raise TypeError(
                    "SegmentConfig.effect_records must contain "
                    f"Effect operations, got {type(record.operation)}"
                )

    def process_segment(self) -> Video:
        """Load the segment and apply transforms then effects."""
        video = Video.from_path(
            str(self.source_video),
            start_second=self.start_second,
            end_second=self.end_second,
        )
        for record in self.transform_records:
            video = record.operation.apply(video)
        for record in self.effect_records:
            video = record.operation.apply(
                video,
                start=_coerce_optional_number(record.apply_args.get("start"), "start"),
                stop=_coerce_optional_number(record.apply_args.get("stop"), "stop"),
            )
        return video


class VideoEdit:
    """Represents a complete multi-segment video editing plan."""

    def __init__(
        self,
        segments: Sequence[SegmentConfig],
        post_transform_records: Sequence[_StepRecord] | None = None,
        post_effect_records: Sequence[_StepRecord] | None = None,
    ):
        if not segments:
            raise ValueError("VideoEdit requires at least one segment")
        self.segments: tuple[SegmentConfig, ...] = tuple(segments)
        self.post_transform_records: tuple[_StepRecord, ...] = tuple(post_transform_records or ())
        self.post_effect_records: tuple[_StepRecord, ...] = tuple(post_effect_records or ())

        for record in self.post_transform_records:
            if not isinstance(record.operation, Transformation):
                raise TypeError(
                    "VideoEdit.post_transform_records must contain "
                    f"Transformation operations, got {type(record.operation)}"
                )
        for record in self.post_effect_records:
            if not isinstance(record.operation, Effect):
                raise TypeError(
                    f"VideoEdit.post_effect_records must contain Effect operations, got {type(record.operation)}"
                )

    @classmethod
    def from_json(cls, text: str) -> VideoEdit:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid VideoEdit JSON: {e.msg} at line {e.lineno} column {e.colno}") from e
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoEdit:
        if not isinstance(data, dict):
            raise ValueError("VideoEdit plan must be a JSON object")

        segments_data = data.get("segments")
        if segments_data is None:
            raise ValueError("VideoEdit plan is missing required key 'segments'")
        if not isinstance(segments_data, list):
            raise ValueError("VideoEdit plan 'segments' must be a list")
        if not segments_data:
            raise ValueError("VideoEdit plan 'segments' must not be empty")

        post_transforms_data = data.get("post_transforms", [])
        post_effects_data = data.get("post_effects", [])
        if not isinstance(post_transforms_data, list):
            raise ValueError("VideoEdit plan 'post_transforms' must be a list")
        if not isinstance(post_effects_data, list):
            raise ValueError("VideoEdit plan 'post_effects' must be a list")

        segments: list[SegmentConfig] = []
        for i, segment_data in enumerate(segments_data):
            location = f"segments[{i}]"
            segments.append(_parse_segment(segment_data, location))

        post_transform_records = [
            _parse_transform_step(step, f"post_transforms[{i}]") for i, step in enumerate(post_transforms_data)
        ]
        post_effect_records = [
            _parse_effect_step(step, f"post_effects[{i}]") for i, step in enumerate(post_effects_data)
        ]

        return cls(
            segments=segments,
            post_transform_records=post_transform_records,
            post_effect_records=post_effect_records,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to canonical JSON-compatible dict.

        Serialization uses `_StepRecord` snapshots as the source of truth. Mutating
        live operation objects after parsing/construction does not affect output.
        """
        return {
            "segments": [self._segment_to_dict(segment) for segment in self.segments],
            "post_transforms": [_step_to_dict(record, include_apply=False) for record in self.post_transform_records],
            "post_effects": [_step_to_dict(record, include_apply=True) for record in self.post_effect_records],
        }

    def run(self) -> Video:
        """Execute the editing plan and return the final video."""
        video = self._assemble_segments()
        for record in self.post_transform_records:
            video = record.operation.apply(video)
        for record in self.post_effect_records:
            video = record.operation.apply(
                video,
                start=_coerce_optional_number(record.apply_args.get("start"), "start"),
                stop=_coerce_optional_number(record.apply_args.get("stop"), "stop"),
            )
        return video

    def validate(self) -> VideoMetadata:
        """Validate the editing plan without loading video data."""
        segment_metas: list[VideoMetadata] = []
        for i, segment in enumerate(self.segments):
            segment_metas.append(self._validate_segment(i, segment))

        if len(segment_metas) > 1:
            first = segment_metas[0]
            for j, other in enumerate(segment_metas[1:], start=1):
                if first.fps != other.fps:
                    raise ValueError(
                        f"Segment 0 output fps ({first.fps}) != segment {j} output fps ({other.fps}). "
                        f"All segments must have identical fps for concatenation."
                    )
                if (first.width, first.height) != (other.width, other.height):
                    raise ValueError(
                        f"Segment 0 output dimensions ({first.width}x{first.height}) != "
                        f"segment {j} output dimensions ({other.width}x{other.height}). "
                        f"All segments must have identical dimensions for concatenation."
                    )

        meta = VideoMetadata(
            height=segment_metas[0].height,
            width=segment_metas[0].width,
            fps=segment_metas[0].fps,
            frame_count=sum(m.frame_count for m in segment_metas),
            total_seconds=round(sum(m.total_seconds for m in segment_metas), 4),
        )

        for record in self.post_transform_records:
            meta = _predict_transform_metadata(
                meta,
                record.op_id,
                record.args,
                context=f"post-assembly ({record.op_id})",
            )
        for record in self.post_effect_records:
            _validate_effect_bounds(record, meta.total_seconds, context="post-assembly")

        return meta

    def _segment_to_dict(self, segment: SegmentConfig) -> dict[str, Any]:
        return {
            "source": str(segment.source_video),
            "start": segment.start_second,
            "end": segment.end_second,
            "transforms": [_step_to_dict(record, include_apply=False) for record in segment.transform_records],
            "effects": [_step_to_dict(record, include_apply=True) for record in segment.effect_records],
        }

    def _validate_segment(self, index: int, segment: SegmentConfig) -> VideoMetadata:
        ctx = f"Segment {index}"
        if segment.start_second < 0:
            raise ValueError(f"{ctx}: start_second ({segment.start_second}) must be >= 0")
        if segment.end_second <= segment.start_second:
            raise ValueError(
                f"{ctx}: end_second ({segment.end_second}) must be > start_second ({segment.start_second})"
            )

        meta = VideoMetadata.from_path(str(segment.source_video))
        if segment.end_second > meta.total_seconds:
            raise ValueError(
                f"{ctx}: end_second ({segment.end_second}) exceeds source duration ({meta.total_seconds}s)"
            )
        meta = meta.cut(segment.start_second, segment.end_second)

        for record in segment.transform_records:
            meta = _predict_transform_metadata(meta, record.op_id, record.args, context=f"{ctx} ({record.op_id})")
        for record in segment.effect_records:
            _validate_effect_bounds(record, meta.total_seconds, context=ctx)
        return meta

    def _assemble_segments(self) -> Video:
        result: Video | None = None
        for segment in self.segments:
            video = segment.process_segment()
            result = video if result is None else result + video
        assert result is not None
        return result


def _parse_segment(segment_data: Any, location: str) -> SegmentConfig:
    if not isinstance(segment_data, dict):
        raise ValueError(f"{location} must be an object")

    allowed_keys = {"source", "start", "end", "transforms", "effects"}
    unknown = sorted(set(segment_data) - allowed_keys)
    if unknown:
        raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")

    for key in ("source", "start", "end"):
        if key not in segment_data:
            raise ValueError(f"{location} is missing required key '{key}'")

    source = segment_data["source"]
    if not isinstance(source, str):
        raise ValueError(f"{location}.source must be a string path")

    start = _require_number(segment_data["start"], f"{location}.start")
    end = _require_number(segment_data["end"], f"{location}.end")

    transforms_data = segment_data.get("transforms", [])
    effects_data = segment_data.get("effects", [])
    if not isinstance(transforms_data, list):
        raise ValueError(f"{location}.transforms must be a list")
    if not isinstance(effects_data, list):
        raise ValueError(f"{location}.effects must be a list")

    transform_records = [
        _parse_transform_step(step, f"{location}.transforms[{i}]") for i, step in enumerate(transforms_data)
    ]
    effect_records = [_parse_effect_step(step, f"{location}.effects[{i}]") for i, step in enumerate(effects_data)]

    return SegmentConfig(
        source_video=Path(source),
        start_second=start,
        end_second=end,
        transform_records=tuple(transform_records),
        effect_records=tuple(effect_records),
    )


def _parse_transform_step(step: Any, location: str) -> _StepRecord:
    step_dict = _require_step_object(step, location)
    if "apply" in step_dict:
        raise ValueError(f"{location}: transforms do not accept apply params")

    allowed_keys = {"op", "args"}
    unknown = sorted(set(step_dict) - allowed_keys)
    if unknown:
        raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")
    if "op" not in step_dict:
        raise ValueError(f"{location} is missing required key 'op'")

    requested_op = step_dict["op"]
    if not isinstance(requested_op, str):
        raise ValueError(f"{location}.op must be a string")

    args = step_dict.get("args", {})
    spec = _resolve_and_validate_step_spec(requested_op, OperationCategory.TRANSFORMATION, location)

    op_cls = _load_operation_class(spec, requested_op, location)
    _ensure_json_instantiable(op_cls, spec, location)
    _validate_object_arg_map(args, spec.params, f"{location}.args")
    try:
        operation = op_cls(**args)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{location}: Failed to instantiate operation '{requested_op}' "
            f"(canonical '{spec.id}'): {e}"
        ) from e

    if not isinstance(operation, Transformation):
        raise ValueError(f"{location}: Operation '{spec.id}' did not instantiate a Transformation")
    return _StepRecord.create(spec.id, args, {}, operation)


def _parse_effect_step(step: Any, location: str) -> _StepRecord:
    step_dict = _require_step_object(step, location)
    allowed_keys = {"op", "args", "apply"}
    unknown = sorted(set(step_dict) - allowed_keys)
    if unknown:
        raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")
    if "op" not in step_dict:
        raise ValueError(f"{location} is missing required key 'op'")

    requested_op = step_dict["op"]
    if not isinstance(requested_op, str):
        raise ValueError(f"{location}.op must be a string")

    args = step_dict.get("args", {})
    apply_args = step_dict.get("apply", {})

    spec = _resolve_and_validate_step_spec(requested_op, OperationCategory.EFFECT, location)

    op_cls = _load_operation_class(spec, requested_op, location)
    _ensure_json_instantiable(op_cls, spec, location)
    _validate_object_arg_map(args, spec.params, f"{location}.args")
    _validate_object_arg_map(apply_args, spec.apply_params, f"{location}.apply")
    normalized_apply_args = _normalize_effect_apply_args(apply_args, f"{location}.apply")
    try:
        operation = op_cls(**args)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{location}: Failed to instantiate operation '{requested_op}' "
            f"(canonical '{spec.id}'): {e}"
        ) from e

    if not isinstance(operation, Effect):
        raise ValueError(f"{location}: Operation '{spec.id}' did not instantiate an Effect")
    return _StepRecord.create(spec.id, args, normalized_apply_args, operation)


def _require_step_object(step: Any, location: str) -> dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError(f"{location} must be an object")
    return step


def _resolve_and_validate_step_spec(
    requested_op: str,
    expected_category: OperationCategory,
    location: str,
) -> OperationSpec:
    spec = get_operation_spec(requested_op)
    if spec is None:
        raise ValueError(
            f"{location}: Unknown operation '{requested_op}'. If this is an AI operation "
            "(e.g. face_crop, auto_framing), ensure `import videopython.ai` is called before parsing the plan."
        )

    # 1) Category check
    if spec.category != expected_category:
        raise ValueError(
            f"{location}: Expected {expected_category.value} operation, got {spec.category.value} ('{spec.id}')"
        )

    # 2) Tag check
    for tag in ("multi_source", "multi_source_only"):
        if tag in spec.tags:
            raise ValueError(
                f"{location}: Operation '{spec.id}' is not supported in VideoEdit JSON plans (tag '{tag}')"
            )

    return spec


def _load_operation_class(spec: OperationSpec, requested_op: str, location: str) -> type[Any]:
    try:
        module = importlib.import_module(spec.module_path)
        cls = getattr(module, spec.class_name)
        if not isinstance(cls, type):
            raise TypeError(f"Resolved attribute '{spec.class_name}' is not a class")
        return cls
    except (ImportError, AttributeError, TypeError) as e:
        raise ValueError(
            f"{location}: Failed to load operation '{requested_op}' (canonical '{spec.id}'): {e}"
        ) from e


def _ensure_json_instantiable(op_cls: type[Any], spec: OperationSpec, location: str) -> None:
    sig = inspect.signature(op_cls.__init__)
    required_init_params = [
        p.name for p in sig.parameters.values() if p.name != "self" and p.default is inspect.Signature.empty
    ]
    spec_param_names = {param.name for param in spec.params}
    missing = [name for name in required_init_params if name not in spec_param_names]
    if missing:
        first_missing = missing[0]
        raise ValueError(
            f"{location}: Operation '{spec.id}' is registered but not JSON-instantiable because required constructor "
            f"parameter '{first_missing}' is not included in the registry spec."
        )


def _validate_object_arg_map(value: Any, params: tuple[ParamSpec, ...], location: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    allowed = {param.name for param in params}
    required = [param.name for param in params if param.required]
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{location} has unknown keys: {', '.join(unknown)}")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"{location} is missing required keys: {', '.join(missing)}")


def _normalize_effect_apply_args(apply_args: Mapping[str, Any], location: str) -> dict[str, Any]:
    """Normalize effect apply args that are consumed directly (not via constructor).

    Phase 2 defers broad primitive type validation, but `start`/`stop` are read directly
    during execution/validation, so validate/coerce them at parse time for actionable errors.
    """
    normalized = dict(apply_args)
    if "start" in normalized:
        normalized["start"] = _coerce_optional_number_at_location(normalized["start"], f"{location}.start")
    if "stop" in normalized:
        normalized["stop"] = _coerce_optional_number_at_location(normalized["stop"], f"{location}.stop")
    return normalized


def _step_to_dict(record: _StepRecord, *, include_apply: bool) -> dict[str, Any]:
    step: dict[str, Any] = {"op": record.op_id}
    args_copy = copy.deepcopy(record.args)
    if args_copy:
        step["args"] = args_copy
    if include_apply:
        apply_copy = copy.deepcopy(record.apply_args)
        if apply_copy:
            step["apply"] = apply_copy
    return step


def _predict_transform_metadata(
    meta: VideoMetadata,
    op_id: str,
    args: Mapping[str, Any],
    context: str = "",
) -> VideoMetadata:
    spec = get_operation_spec(op_id)
    if spec is None or spec.category != OperationCategory.TRANSFORMATION or spec.metadata_method is None:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}Metadata prediction is not supported for transform op '{op_id}'. "
            "Only transforms with a registered metadata_method can be validated."
        )

    try:
        method = getattr(VideoMetadata, spec.metadata_method)
    except AttributeError as e:
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}Registry metadata_method '{spec.metadata_method}' for op '{op_id}' "
            "is not implemented on VideoMetadata"
        ) from e

    accepted_params = {p.name for p in inspect.signature(method).parameters.values() if p.name != "self"}
    prepared = _prepare_metadata_args(meta, op_id, args, accepted_params)
    try:
        return getattr(meta, spec.metadata_method)(**prepared)
    except (TypeError, ValueError) as e:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}Metadata prediction failed for transform op '{op_id}': {e}") from e


def _prepare_metadata_args(
    meta: VideoMetadata,
    op_id: str,
    args: Mapping[str, Any],
    accepted_params: set[str],
) -> dict[str, Any]:
    filtered = {k: v for k, v in args.items() if k in accepted_params}

    if op_id == "crop":
        width = filtered.get("width")
        height = filtered.get("height")
        if isinstance(width, float) and 0 < width <= 1:
            filtered["width"] = int(width * meta.width)
        elif width is not None:
            filtered["width"] = int(width)

        if isinstance(height, float) and 0 < height <= 1:
            filtered["height"] = int(height * meta.height)
        elif height is not None:
            filtered["height"] = int(height)

    if op_id == "speed_change":
        end_speed = args.get("end_speed")
        if end_speed is not None and "speed" in filtered:
            filtered["speed"] = (filtered["speed"] + end_speed) / 2

    return filtered


def _validate_effect_bounds(
    record: _StepRecord,
    duration: float,
    context: str = "",
) -> None:
    prefix = f"{context}: " if context else ""
    effect_name = type(record.operation).__name__
    op_label = record.op_id

    start = _coerce_optional_number(record.apply_args.get("start"), "start")
    stop = _coerce_optional_number(record.apply_args.get("stop"), "stop")

    if start is not None:
        if start < 0:
            raise ValueError(f"{prefix}Effect '{op_label}' ({effect_name}) start ({start}) must be >= 0")
        if start > duration:
            raise ValueError(
                f"{prefix}Effect '{op_label}' ({effect_name}) start ({start}) exceeds timeline duration ({duration}s)"
            )
    if stop is not None:
        if stop < 0:
            raise ValueError(f"{prefix}Effect '{op_label}' ({effect_name}) stop ({stop}) must be >= 0")
        if stop > duration:
            raise ValueError(
                f"{prefix}Effect '{op_label}' ({effect_name}) stop ({stop}) exceeds timeline duration ({duration}s)"
            )
    if start is not None and stop is not None and start > stop:
        raise ValueError(f"{prefix}Effect '{op_label}' ({effect_name}) start ({start}) must be <= stop ({stop})")


def _require_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    return float(value)


def _coerce_optional_number(value: Any, param_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Effect apply parameter '{param_name}' must be a number")
    return float(value)


def _coerce_optional_number_at_location(value: Any, location: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a number")
    return float(value)

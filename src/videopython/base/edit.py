from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path

from videopython.base.effects import Effect
from videopython.base.registry import (
    OperationCategory,
    get_operation_specs,
)
from videopython.base.transforms import Transformation
from videopython.base.video import Video, VideoMetadata

__all__ = [
    "EffectApplication",
    "SegmentConfig",
    "VideoEdit",
]


@dataclass
class EffectApplication:
    """Pairs an Effect with its apply-time start/stop parameters."""

    effect: Effect
    start: float | None = None
    stop: float | None = None


@dataclass
class SegmentConfig:
    """Configuration for a single video segment in an editing plan."""

    source_video: Path
    start_second: float
    end_second: float
    transforms: list[Transformation] = field(default_factory=list)
    effects: list[EffectApplication] = field(default_factory=list)

    def process_segment(self) -> Video:
        """Load the segment and apply transforms then effects."""
        video = Video.from_path(
            str(self.source_video),
            start_second=self.start_second,
            end_second=self.end_second,
        )
        for transform in self.transforms:
            video = transform.apply(video)
        for ea in self.effects:
            video = ea.effect.apply(video, start=ea.start, stop=ea.stop)
        return video


class VideoEdit:
    """Represents a complete multi-segment video editing plan.

    Execution order at each level (segment and post-assembly):
    transforms first, then effects. This is enforced by the data model.
    """

    def __init__(
        self,
        segments: list[SegmentConfig],
        post_transforms: list[Transformation] | None = None,
        post_effects: list[EffectApplication] | None = None,
    ):
        if not segments:
            raise ValueError("VideoEdit requires at least one segment")
        self.segments: tuple[SegmentConfig, ...] = tuple(segments)
        self.post_transforms: tuple[Transformation, ...] = tuple(post_transforms or [])
        self.post_effects: tuple[EffectApplication, ...] = tuple(post_effects or [])

    def run(self) -> Video:
        """Execute the editing plan and return the final video."""
        video = self._assemble_segments()
        for transform in self.post_transforms:
            video = transform.apply(video)
        for ea in self.post_effects:
            video = ea.effect.apply(video, start=ea.start, stop=ea.stop)
        return video

    def validate(self) -> VideoMetadata:
        """Validate the editing plan without loading video data.

        Uses VideoMetadata prediction to simulate the pipeline. Raises on
        any validation error with an actionable message. Returns the predicted
        final VideoMetadata on success.
        """
        segment_metas: list[VideoMetadata] = []

        for i, segment in enumerate(self.segments):
            meta = self._validate_segment(i, segment)
            segment_metas.append(meta)

        # Assembly compatibility: strict check matching Video.__add__ semantics
        # (exact fps and frame_shape equality, not the rounded check in can_be_merged_with)
        if len(segment_metas) > 1:
            first = segment_metas[0]
            for j in range(1, len(segment_metas)):
                other = segment_metas[j]
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

        # Compute assembled metadata
        meta = VideoMetadata(
            height=segment_metas[0].height,
            width=segment_metas[0].width,
            fps=segment_metas[0].fps,
            frame_count=sum(m.frame_count for m in segment_metas),
            total_seconds=round(sum(m.total_seconds for m in segment_metas), 4),
        )

        # Apply post-transforms
        for transform in self.post_transforms:
            meta = _predict_transform_metadata(meta, transform)

        # Validate post-effects
        for ea in self.post_effects:
            _validate_effect_bounds(ea, meta.total_seconds, context="post-assembly")

        return meta

    def _validate_segment(self, index: int, segment: SegmentConfig) -> VideoMetadata:
        """Validate a single segment and return its predicted output metadata."""
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

        for transform in segment.transforms:
            meta = _predict_transform_metadata(meta, transform, context=ctx)

        for ea in segment.effects:
            _validate_effect_bounds(ea, meta.total_seconds, context=ctx)

        return meta

    def _assemble_segments(self) -> Video:
        """Process all segments and concatenate them."""
        result: Video | None = None
        for segment in self.segments:
            video = segment.process_segment()
            if result is None:
                result = video
            else:
                result = result + video
        assert result is not None  # guaranteed by __init__ check
        return result


# --- Metadata prediction helpers ---

# Cache: maps (module_path, class_name) -> (metadata_method, metadata_method_params)
# Only positive lookups are cached to avoid hiding late-registered operations.
_SPEC_CACHE: dict[tuple[str, str], tuple[str, set[str]]] = {}


def _get_metadata_method_for_class(
    transform: Transformation,
) -> tuple[str, set[str]] | None:
    """Look up the registry metadata_method for a transform's class.

    Returns (method_name, set_of_accepted_param_names) or None if not found.
    Only positive results are cached so that late-registered operations
    (e.g. AI transforms registered after import videopython.ai) are picked up.
    """
    cls = type(transform)
    module_path = cls.__module__
    class_name = cls.__name__
    key = (module_path, class_name)

    if key in _SPEC_CACHE:
        return _SPEC_CACHE[key]

    for spec in get_operation_specs().values():
        if (
            spec.module_path == module_path
            and spec.class_name == class_name
            and spec.category == OperationCategory.TRANSFORMATION
        ):
            if spec.metadata_method:
                method = getattr(VideoMetadata, spec.metadata_method)
                sig = inspect.signature(method)
                accepted = {p.name for p in sig.parameters.values() if p.name != "self"}
                result = (spec.metadata_method, accepted)
                _SPEC_CACHE[key] = result
                return result
            break

    return None


def _extract_transform_args(transform: Transformation) -> dict:
    """Extract constructor arguments from a transform instance by inspecting its __init__."""
    sig = inspect.signature(type(transform).__init__)
    args = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if hasattr(transform, name):
            args[name] = getattr(transform, name)
    return args


def _prepare_metadata_args(
    meta: VideoMetadata,
    transform: Transformation,
    args: dict,
    accepted_params: set[str],
) -> dict:
    """Prepare arguments for a VideoMetadata prediction method.

    Handles special cases like Crop with normalized float values.
    """
    from videopython.base.transforms import Crop, SpeedChange

    filtered = {k: v for k, v in args.items() if k in accepted_params}

    if isinstance(transform, Crop):
        # Convert normalized floats to pixel values for metadata prediction
        if isinstance(filtered.get("width"), float) and 0 < filtered["width"] <= 1:
            filtered["width"] = int(filtered["width"] * meta.width)
        else:
            filtered["width"] = int(filtered.get("width", meta.width))
        if isinstance(filtered.get("height"), float) and 0 < filtered["height"] <= 1:
            filtered["height"] = int(filtered["height"] * meta.height)
        else:
            filtered["height"] = int(filtered.get("height", meta.height))

    if isinstance(transform, SpeedChange):
        # For ramped speed, use average speed for metadata prediction
        end_speed = args.get("end_speed")
        if end_speed is not None and "speed" in filtered:
            filtered["speed"] = (filtered["speed"] + end_speed) / 2

    return filtered


def _predict_transform_metadata(
    meta: VideoMetadata,
    transform: Transformation,
    context: str = "",
) -> VideoMetadata:
    """Predict output metadata after applying a transform."""
    lookup = _get_metadata_method_for_class(transform)
    if lookup is None:
        cls_name = type(transform).__name__
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"{prefix}Metadata prediction is not supported for transform "
            f"'{cls_name}'. Only transforms with a registered metadata_method "
            f"can be validated. Consider removing this transform from the plan "
            f"or skipping validation."
        )

    method_name, accepted_params = lookup
    args = _extract_transform_args(transform)
    prepared = _prepare_metadata_args(meta, transform, args, accepted_params)
    method = getattr(meta, method_name)
    return method(**prepared)


def _validate_effect_bounds(
    ea: EffectApplication,
    duration: float,
    context: str = "",
) -> None:
    """Validate effect start/stop bounds against the timeline duration."""
    prefix = f"{context}: " if context else ""
    effect_name = type(ea.effect).__name__

    if ea.start is not None:
        if ea.start < 0:
            raise ValueError(f"{prefix}Effect '{effect_name}' start ({ea.start}) must be >= 0")
        if ea.start > duration:
            raise ValueError(
                f"{prefix}Effect '{effect_name}' start ({ea.start}) exceeds timeline duration ({duration}s)"
            )

    if ea.stop is not None:
        if ea.stop < 0:
            raise ValueError(f"{prefix}Effect '{effect_name}' stop ({ea.stop}) must be >= 0")
        if ea.stop > duration:
            raise ValueError(f"{prefix}Effect '{effect_name}' stop ({ea.stop}) exceeds timeline duration ({duration}s)")

    if ea.start is not None and ea.stop is not None and ea.start > ea.stop:
        raise ValueError(f"{prefix}Effect '{effect_name}' start ({ea.start}) must be <= stop ({ea.stop})")

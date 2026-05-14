"""Operation: single source of truth for editing primitives.

Every editing primitive is an ``Operation`` subclass -- a Pydantic model whose
fields ARE the JSON wire format. Validation, schema, and serialisation come for
free; subclasses just declare fields and implement ``apply``. Auto-registration
via ``__pydantic_init_subclass__`` builds the ``op_id -> class`` registry as
modules are imported.

Subclass contract::

    class Resize(Operation):
        '''Resize the video.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.
        '''

        op: Literal["resize"] = "resize"
        category: ClassVar[OpCategory] = OpCategory.TRANSFORM
        streamable: ClassVar[bool] = True

        width: int | None = Field(None, gt=0)
        height: int | None = Field(None, gt=0)

        def apply(self, video: Video) -> Video: ...
        def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata: ...
        def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None: ...
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Union, get_args, get_origin

import numpy as np
from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter, model_validator

if TYPE_CHECKING:
    from videopython.base.video import Video, VideoMetadata

__all__ = [
    "OpCategory",
    "TimeRange",
    "FilterCtx",
    "Operation",
    "Effect",
]


class OpCategory(str, Enum):
    """Coarse execution category for an Operation subclass."""

    TRANSFORM = "transform"
    EFFECT = "effect"
    SPECIAL = "special"


class TimeRange(BaseModel):
    """Half-open time window in seconds: ``[start, stop)``.

    Either endpoint may be ``None``, meaning "from the beginning" / "to the
    end" respectively. Used by :class:`Effect.window` and elsewhere.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: float | None = Field(None, ge=0, description="Start time in seconds. None means 0.")
    stop: float | None = Field(None, ge=0, description="Stop time in seconds. None means end of video.")

    @model_validator(mode="after")
    def _validate_order(self) -> TimeRange:
        if self.start is not None and self.stop is not None and self.stop < self.start:
            raise ValueError(f"TimeRange.stop ({self.stop}) must be >= start ({self.start})")
        return self


@dataclass(frozen=True)
class FilterCtx:
    """Current pipeline state (post-prior-ops) when compiling to ffmpeg."""

    width: int
    height: int
    fps: float


def _parse_google_docstring_args(docstring: str | None) -> dict[str, str]:
    """Extract ``{param_name: description}`` from a Google-style ``Args:`` block.

    Used to populate ``model_fields[name].description`` automatically so that
    the LLM-facing JSON schema stays in sync with the prose docstring.
    """
    if not docstring:
        return {}
    lines = docstring.strip().splitlines()
    in_args = False
    args: dict[str, str] = {}
    current_name: str | None = None
    current_desc_parts: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if not in_args:
            continue
        if stripped and not line[0].isspace():
            break
        if not stripped:
            continue
        lstripped = line.lstrip()
        indent = len(line) - len(lstripped)
        if ": " in lstripped and indent <= 12:
            if current_name is not None:
                args[current_name] = " ".join(current_desc_parts).strip()
            name_part, _, desc = lstripped.partition(": ")
            current_name = name_part.split("(")[0].split(":")[0].strip()
            current_desc_parts = [desc] if desc else []
        elif current_name is not None:
            current_desc_parts.append(stripped)
    if current_name is not None:
        args[current_name] = " ".join(current_desc_parts).strip()
    return args


class Operation(BaseModel):
    """Pydantic base for every editing primitive.

    Concrete subclasses MUST declare an ``op`` field with a single-value
    ``Literal[str]`` annotation; that value is the discriminator on the JSON
    wire and the registry key. Subclasses may override the ``category``,
    ``streamable``, and ``requires`` ClassVars.

    The default ``apply`` raises ``NotImplementedError``; ``predict_metadata``
    defaults to identity; ``to_ffmpeg_filter`` defaults to ``None`` (eager).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: str

    category: ClassVar[OpCategory] = OpCategory.SPECIAL
    streamable: ClassVar[bool] = False
    requires: ClassVar[tuple[str, ...]] = ()

    _registry: ClassVar[dict[str, type[Operation]]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        op_field = cls.model_fields.get("op")
        if op_field is None:
            return
        annotation = op_field.annotation
        if get_origin(annotation) is not Literal:
            # Abstract intermediate (e.g. Effect) -- no concrete op_id yet.
            return
        literal_values = get_args(annotation)
        if len(literal_values) != 1 or not isinstance(literal_values[0], str):
            raise TypeError(f"{cls.__name__}.op must be Literal of a single str, got {literal_values!r}")
        op_id = literal_values[0]

        existing = Operation._registry.get(op_id)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Duplicate op_id '{op_id}': "
                f"{cls.__module__}.{cls.__qualname__} vs "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        Operation._registry[op_id] = cls

        descs = _parse_google_docstring_args(inspect.getdoc(cls))
        for name, desc in descs.items():
            field = cls.model_fields.get(name)
            if field is None or field.description:
                continue
            field.description = desc

    @property
    def op_id(self) -> str:
        """Wire / registry identifier. Mirrors ``self.op``."""
        return self.op

    @classmethod
    def registry(cls) -> dict[str, type[Operation]]:
        """Snapshot of ``{op_id: subclass}`` for every registered Operation."""
        return dict(Operation._registry)

    @classmethod
    def get(cls, op_id: str) -> type[Operation]:
        """Look up the Operation subclass for ``op_id``."""
        try:
            return Operation._registry[op_id]
        except KeyError as exc:
            known = ", ".join(sorted(Operation._registry)) or "(none)"
            raise KeyError(f"Unknown op_id {op_id!r}. Known ops: [{known}]") from exc

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        """Discriminated-union JSON schema over every registered Operation.

        ``op`` is the discriminator tag. This is the LLM-facing schema for
        validating a single operation payload.
        """
        if not Operation._registry:
            return {"type": "object"}
        ops = sorted(Operation._registry.values(), key=lambda c: c.__name__)
        annotated = Annotated[Union[tuple(ops)], Discriminator("op")]  # type: ignore[valid-type]  # noqa: UP007
        return TypeAdapter(annotated).json_schema()

    def apply(self, video: Video, **context: Any) -> Video:
        """Run this operation on ``video``.

        The runner passes pipeline-context values listed in ``cls.requires``
        as keyword arguments (e.g. ``transcription=...``). Subclasses with
        explicit dependencies should declare them as typed kwargs and ignore
        ``**context``.
        """
        raise NotImplementedError(f"{type(self).__name__}.apply not implemented")

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        """Predict output metadata from input metadata. Default: identity."""
        return meta

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        """Compile to an ffmpeg ``-vf`` filter expression, or ``None`` for eager.

        Streamable transforms override this. Effects use ``process_frame``
        instead -- they do not go through ffmpeg filters.
        """
        return None


class Effect(Operation):
    """Operation that preserves shape and frame count, with optional streaming.

    Subclasses override :meth:`_apply` for in-memory execution and may
    additionally override :meth:`streaming_init` / :meth:`process_frame` for
    bounded-memory streaming via ``base/streaming.py``. The base
    :meth:`apply` resolves :attr:`window`, slices the video, runs
    ``_apply`` on the slice, splices the result back, and asserts the
    shape-preserving invariant.
    """

    category: ClassVar[OpCategory] = OpCategory.EFFECT

    window: TimeRange | None = Field(
        None,
        description="Time window for the effect in seconds. Omit to apply across the full duration.",
    )

    def apply(self, video: Video, **context: Any) -> Video:
        from videopython.base.video import Video as _Video

        original_shape = video.video_shape

        if self.window is None or (self.window.start is None and self.window.stop is None):
            result = self._apply(video)
        else:
            start_s, stop_s = self._resolved_window(video.total_seconds)
            start_f = round(start_s * video.fps)
            end_f = round(stop_s * video.fps)
            inner = self._apply(video[start_f:end_f])
            old_audio = video.audio
            result = _Video.from_frames(
                np.r_["0,2", video.frames[:start_f], inner.frames, video.frames[end_f:]],
                fps=video.fps,
            )
            result.audio = old_audio

        if result.video_shape != original_shape:
            raise RuntimeError(
                f"{type(self).__name__} changed video shape from {original_shape} "
                f"to {result.video_shape}; effects must preserve shape and frame count."
            )
        return result

    def _resolved_window(self, total_seconds: float) -> tuple[float, float]:
        win = self.window or TimeRange()
        start_s = 0.0 if win.start is None else float(win.start)
        stop_s = total_seconds if win.stop is None else float(win.stop)
        start_s = min(start_s, total_seconds)
        stop_s = min(stop_s, total_seconds)
        if stop_s < start_s:
            raise ValueError(f"Effect stop ({stop_s}) must be >= start ({start_s})")
        return start_s, stop_s

    def _apply(self, video: Video) -> Video:
        """Apply the effect to ``video`` in memory. Override in subclasses."""
        raise NotImplementedError(f"{type(self).__name__}._apply not implemented")

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        """Hook for per-stream precomputation (per-frame alphas, sigma curves...).

        Default: no-op. Override in subclasses that need it.
        """

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Process one ``(H, W, 3) uint8`` frame in streaming mode.

        ``frame_index`` is 0-based within this effect's active window.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

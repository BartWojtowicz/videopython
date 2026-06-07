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

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, NamedTuple, Union, get_args, get_origin

import numpy as np
from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter
from tqdm import tqdm

if TYPE_CHECKING:
    from videopython.base.video import Video, VideoMetadata

__all__ = [
    "OpCategory",
    "TimeRange",
    "BoundedTimeField",
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

    Parsing is deliberately permissive: ``start``/``stop`` are plain floats
    with no ``ge=0`` or ordering constraint. The plan skeleton accepts the
    *shape*; the numeric bounds (``>= 0``, ``stop >= start``, in-duration) are
    owned by :meth:`VideoEdit.validate` / :meth:`VideoEdit.check`, which report
    them as structured, collectable, repairable :class:`PlanError`s instead of
    aborting at ``from_dict``. :meth:`Effect._resolved_window` still clamps at
    run time, so a plan run without validation degrades rather than crashes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: float | None = Field(None, description="Start time in seconds. None means 0.")
    stop: float | None = Field(None, description="Stop time in seconds. None means end of video.")


class BoundedTimeField(NamedTuple):
    """Declares a time-valued (seconds) op field that :meth:`VideoEdit.repair` clamps.

    ``name`` is the field; the lower bound is always ``0``. ``exclusive_end``
    picks the upper bound: ``False`` clamps to the clip duration (the value may
    equal it); ``True`` clamps to the last addressable frame
    ``(frame_count - 1) / fps`` -- for fields that index a frame and so must be
    *strictly* less than the duration (e.g. ``freeze_frame.timestamp``).
    """

    name: str
    exclusive_end: bool


@dataclass(frozen=True)
class FilterCtx:
    """Current pipeline state (post-prior-ops) when compiling to ffmpeg."""

    width: int
    height: int
    fps: float


LLM_HIDDEN_KEY = "llm_hidden"


def _strip_llm_hidden(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively drop fields marked ``llm_hidden`` from a generated JSON schema.

    A field declared with ``Field(json_schema_extra={"llm_hidden": True})`` is a
    valid wire field -- it still parses and runs -- but advanced / non-LLM, e.g. a
    raw font *path* when a ``font`` name enum already covers the LLM case. This
    walks ``properties`` / ``$defs`` at any depth, removes such fields, and prunes
    them from sibling ``required`` lists, so the LLM-facing schema never shows
    them. Mutates and returns ``schema`` (callers pass a freshly generated dict).
    """
    props = schema.get("properties")
    if isinstance(props, dict):
        hidden = [name for name, sub in props.items() if isinstance(sub, dict) and sub.get(LLM_HIDDEN_KEY)]
        for name in hidden:
            del props[name]
        required = schema.get("required")
        if hidden and isinstance(required, list):
            schema["required"] = [r for r in required if r not in hidden]
    for value in schema.values():
        if isinstance(value, dict):
            _strip_llm_hidden(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _strip_llm_hidden(item)
    return schema


def _to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a generated JSON schema into a provider strict-mode grammar.

    Strict structured-output modes (OpenAI/OpenRouter ``json_schema``) require:
    every object closed (``additionalProperties: false``); every declared
    property listed in ``required``; and unions expressed as ``anyOf`` without a
    ``discriminator`` keyword. The ``default`` keyword (which strict mode
    rejects, and which is moot once every field is required) is dropped. Numeric
    constraints already emitted by Pydantic are kept verbatim.

    Optionality is taken verbatim from what Pydantic emitted, *not* synthesized:
    strict mode represents an optional field as a nullable required field, and
    Pydantic already encodes exactly that -- an ``Optional`` field carries a
    ``{"type": "null"}`` branch while a defaulted-but-non-``Optional`` field
    (e.g. ``operations: list = []``, ``match_to_lowest_fps: bool = True``) does
    not. So we force every property into ``required`` without adding null
    branches: synthesizing null for a non-``Optional`` field would let a grammar
    emit a null the Pydantic model then rejects -- reintroducing the very
    re-prompt strict mode exists to remove. The union discriminator ``op`` is a
    defaulted ``const`` and is likewise kept required and non-nullable for free.

    Returns a new schema; the input is not mutated. Pydantic ``$ref``/``$defs``
    indirection is left intact (providers resolve it); the per-``$defs`` object
    bodies are rewritten in place of their definitions.
    """
    import copy

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node

        out = {k: walk(v) for k, v in node.items()}

        # A discriminated union: Pydantic emits `oneOf` + `discriminator`.
        # Strict mode wants a plain `anyOf` of variants and no discriminator.
        if "oneOf" in out:
            out["anyOf"] = out.pop("oneOf")
        # Drop keywords strict mode rejects (or that are moot once everything is
        # required): the discriminator tag, `default`, custom `format`s like
        # "path", and any `$schema`/`$id` envelope.
        for key in ("discriminator", "default", "format", "$schema", "$id"):
            out.pop(key, None)

        # Close every object and require all of its properties. Nullability is
        # left exactly as Pydantic emitted it (see the docstring) -- no synthesis.
        if isinstance(out.get("properties"), dict):
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out

    return walk(copy.deepcopy(schema))


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
    llm_exposed: ClassVar[bool] = True
    time_fields: ClassVar[tuple[BoundedTimeField, ...]] = ()
    """Time-valued (seconds) fields :meth:`VideoEdit.repair` may clamp into range.

    Declaring a :class:`BoundedTimeField` here lets ``repair`` clamp an
    out-of-range timestamp (e.g. ``freeze_frame.timestamp`` past the clip end)
    without per-op special-casing -- the repair pass reads the declaration,
    clamps to ``[0, bound]``, and records a :class:`PlanRepair`. Empty by
    default; ops with no time-valued params declare nothing.
    """

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

    @property
    def op_id(self) -> str:
        """Wire / registry identifier. Mirrors ``self.op``."""
        return self.op

    @classmethod
    def registry(cls) -> dict[str, type[Operation]]:
        """Snapshot of ``{op_id: subclass}`` for every registered Operation."""
        return dict(Operation._registry)

    @classmethod
    def llm_registry(cls) -> dict[str, type[Operation]]:
        """Snapshot of ``{op_id: subclass}`` for LLM-exposed Operations only.

        A subset of :meth:`registry` filtered to subclasses with
        ``llm_exposed`` True. Server-only ops (e.g. those needing a
        server-resolved ``source`` path) are excluded so they never leak into
        the LLM-facing schema.
        """
        return {op_id: sub for op_id, sub in Operation._registry.items() if sub.llm_exposed}

    @classmethod
    def get(cls, op_id: str) -> type[Operation]:
        """Look up the Operation subclass for ``op_id``."""
        try:
            return Operation._registry[op_id]
        except KeyError as exc:
            known = ", ".join(sorted(Operation._registry)) or "(none)"
            raise KeyError(f"Unknown op_id {op_id!r}. Known ops: [{known}]") from exc

    @classmethod
    def json_schema(cls, include_server_only: bool = False, *, strict: bool = False) -> dict[str, Any]:
        """Discriminated-union JSON schema over registered Operations.

        ``op`` is the discriminator tag. This is the LLM-facing schema for
        validating a single operation payload. By default the union covers only
        LLM-exposed ops (:meth:`llm_registry`); pass ``include_server_only=True``
        to build the union from the full :meth:`registry`. Fields marked
        ``llm_hidden`` (advanced overrides like raw font paths) are stripped.

        With ``strict=True`` the schema is rewritten for use as a provider
        structured-output **grammar** (OpenAI/OpenRouter ``json_schema`` strict
        mode): every object is closed (``additionalProperties: false``), every
        property is listed in ``required`` with its optionality kept exactly as
        Pydantic emitted it (an ``Optional`` field keeps its nullable branch; a
        defaulted non-``Optional`` field -- including the ``op`` discriminator --
        stays required and non-nullable), and the discriminated union is
        expressed as a plain ``anyOf`` of closed variants (``discriminator``,
        ``default``, custom ``format``, and ``$schema`` -- all unsupported or moot
        in strict mode -- are dropped). Numeric constraints
        (``minimum``/``maximum``/``exclusiveMinimum``) are preserved, so an
        entire class of bound violations becomes impossible at decode time.

        Note: the strict result is a *root-level* ``anyOf`` union -- an embeddable
        schema fragment, not a submittable strict root (providers require the root
        to be a closed object). It is consumed inside
        :meth:`VideoEdit.json_schema(strict=True) <VideoEdit.json_schema>`, which
        *is* a submittable object root; use that to constrain a whole plan.
        """
        source = Operation._registry if include_server_only else cls.llm_registry()
        if not source:
            return {"type": "object"}
        ops = sorted(source.values(), key=lambda c: c.__name__)
        annotated = Annotated[Union[tuple(ops)], Discriminator("op")]  # type: ignore[valid-type]  # noqa: UP007
        schema = _strip_llm_hidden(TypeAdapter(annotated).json_schema())
        return _to_strict_schema(schema) if strict else schema

    @classmethod
    def llm_json_schema(cls) -> dict[str, Any]:
        """Per-op JSON schema with ``llm_hidden`` fields removed.

        Like ``cls.model_json_schema()`` but drops advanced / non-LLM fields
        (e.g. raw font paths) so a single op can be exposed to an LLM directly
        without leaking a field the model shouldn't fill in.
        """
        return _strip_llm_hidden(cls.model_json_schema())

    def apply(self, video: Video) -> Video:
        """Run this operation on ``video``.

        The runner passes pipeline-context values listed in ``cls.requires``
        as keyword arguments (e.g. ``transcription=...``). Subclasses that
        declare ``requires`` widen the signature accordingly -- e.g.
        ``def apply(self, video, transcription=None) -> Video``.
        """
        raise NotImplementedError(f"{type(self).__name__}.apply not implemented")

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        """Predict output metadata from input metadata. Default: identity.

        Run during ``VideoEdit.validate()``'s dry-run, before any frames are
        decoded. Beyond predicting shape, this is the fail-fast gate, and it
        has one contract: **reject exactly the plans that would otherwise crash
        or do unrecoverable / expensive work in** :meth:`apply` **/** ``run()``;
        anything ``run()`` can absorb by graceful degradation is NOT rejected.
        ``TranscriptionOverlay`` rejects un-fittable subtitles (they used to
        crash mid-render); ``TextOverlay``/``ImageOverlay`` do not reject
        off-frame geometry (it clips to a valid no-op). Keep the check
        metadata-cheap -- no frame decode.

        Duration bounds checks use the shared
        :data:`videopython.editing.transforms.DURATION_EPS` tolerance: a value
        is rejected only when it exceeds the limit by more than ``DURATION_EPS``
        seconds, so sub-millisecond float drift at an exact boundary passes
        consistently across the editing layer.
        """
        return meta

    def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None:
        """Compile to an ffmpeg ``-vf`` filter expression, or ``None`` for eager.

        Streamable transforms override this. Effects use ``process_frame``
        instead -- they do not go through ffmpeg filters.
        """
        return None


class Effect(Operation):
    """Operation that preserves shape and frame count, driven by per-frame streaming.

    Subclasses implement the streaming contract -- :meth:`process_frame` (and
    :meth:`streaming_init` for any precomputed per-stream state) -- which is the
    single source of truth for the effect's pixel logic. The base
    :meth:`_apply` runs that same contract over the in-memory frames, so
    in-memory execution comes for free; the same code path feeds
    ``editing/streaming.py`` for bounded-memory streaming. The base
    :meth:`apply` resolves :attr:`window`, slices the video, runs ``_apply`` on
    the slice, splices the result back, and asserts the shape-preserving
    invariant.

    Override :meth:`_apply` only when eager execution must genuinely differ from
    a frame-by-frame replay -- e.g. extra validation, a batched vectorisation,
    or audio handling (``Fade``/``VolumeAdjust`` override :meth:`apply` outright
    so the audio splice stays coherent with the window).
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

    def predict_metadata(self, meta: VideoMetadata, **_context: Any) -> VideoMetadata:
        """Effects preserve shape and frame count, so the prediction is identity.

        Accepts ``**_context`` so requires-aware effects (``TranscriptionOverlay``)
        validate without subclasses needing to override just to widen the
        signature. Mirrors :meth:`Effect.apply`'s ``**context`` accept-all.
        """
        return meta

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
        """Apply the effect to ``video`` in memory by replaying the streaming path.

        Runs :meth:`streaming_init` once, then :meth:`process_frame` over every
        frame in order -- the same logic streaming uses, so eager and streaming
        cannot drift. Subclasses that need a genuinely different eager path
        (extra validation, batched vectorisation) override this.
        """
        height, width = video.frame_shape[:2]
        self.streaming_init(len(video.frames), video.fps, width, height)
        for i in tqdm(range(len(video.frames)), desc=type(self).__name__):
            video.frames[i] = self.process_frame(video.frames[i], i)
        return video

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int) -> None:
        """Hook for per-stream precomputation (per-frame alphas, sigma curves...).

        Default: no-op. Override in subclasses that need it.
        """

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Process one ``(H, W, 3) uint8`` frame in streaming mode.

        ``frame_index`` is 0-based within this effect's active window.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

"""Operation: single source of truth for editing primitives.

Every editing primitive is an ``Operation`` subclass -- a Pydantic model whose
fields ARE the JSON wire format. Validation, schema, and serialisation come for
free; subclasses just declare fields and implement the streaming contract.
Auto-registration via ``__pydantic_init_subclass__`` builds the
``op_id -> class`` registry as modules are imported.

Subclass contract::

    class Resize(Operation):
        '''Resize the video.

        Args:
            width: Target width in pixels.
            height: Target height in pixels.
        '''

        op: Literal["resize"] = "resize"
        category: ClassVar[OpCategory] = OpCategory.TRANSFORM

        width: int | None = Field(None, gt=0)
        height: int | None = Field(None, gt=0)

        def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata: ...
        def to_ffmpeg_filter(self, ctx: FilterCtx) -> str | None: ...
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, NamedTuple, Union, get_args, get_origin

import numpy as np
from pydantic import BaseModel, ConfigDict, Discriminator, Field, TypeAdapter

if TYPE_CHECKING:
    from videopython.base.video import VideoMetadata

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
    aborting at ``from_dict``. The window is still clamped to
    ``min(stop, total_seconds)`` at run time, so a plan run without validation
    degrades rather than crashes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: float | None = Field(None, description="Start time in seconds. None means 0.")
    stop: float | None = Field(None, description="Stop time in seconds. None means end of video.")


class BoundedTimeField(NamedTuple):
    """Declares a time-valued (seconds) op field that :meth:`VideoEdit.repair` clamps.

    ``name`` is the field; the lower bound is always ``0``. ``exclusive_end``
    distinguishes how the upper bound is enforced so repair clamps exactly what
    validation rejects: ``False`` permits the clip duration (reject ``value >
    total_seconds``, clamp to the duration); ``True`` is for a field that indexes
    a frame and so must be *strictly* less than the duration (reject ``value >=
    total_seconds``, clamp to the last addressable frame ``(frame_count - 1) /
    fps``) -- e.g. ``freeze_frame.timestamp``.
    """

    name: str
    exclusive_end: bool


@dataclass(frozen=True)
class FilterCtx:
    """Current pipeline state (post-prior-ops) when compiling to ffmpeg.

    ``frame_count`` is the number of frames entering the filter at this chain
    position (the plan builder folds ``predict_metadata`` through the chain),
    so duration-aware compilations (a speed ramp's time-warp expression, a
    freeze's frame indices) can be exact. ``0`` when unknown -- compilations
    that need it must return ``None`` (no filter compilation) in that case.

    ``context`` carries the resolved, segment-local runtime context (the same
    re-based values ``streaming_init`` receives) so a context-consuming op can
    compile itself into the filter chain (e.g. ``add_subtitles`` consuming the
    transcription to write an ``.ass`` file). Empty when no context applies.

    ``owned_files`` collects temp files a compilation creates (the ``.ass``
    file a ``subtitles=`` entry references); the plan runner deletes them once
    streaming finishes or the plan is abandoned.

    ``source_path``/``start_second``/``end_second`` locate the segment on
    disk, and ``decode_filters`` is the decode-stage filter prefix ahead of
    this op -- together they let a compilation run its own bounded decode
    pass over exactly the frames the filter will see (``face_crop``'s
    detection). ``decode_filters`` is ``None`` when those frames are not
    reproducible at compile time (the op sits at the encode stage, behind
    per-frame Python effects); such compilations must return ``None``.
    """

    width: int
    height: int
    fps: float
    frame_count: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    owned_files: list[Path] = field(default_factory=list)
    source_path: Path | None = None
    start_second: float = 0.0
    end_second: float | None = None
    decode_filters: tuple[str, ...] | None = ()
    audio_label: str = "a"
    """A unique-within-the-graph prefix an audio-filter compilation can use to
    name internal ``filter_complex`` labels (``freeze_frame``'s split/concat
    splice). The plan builder sets a distinct value per op so a multi-statement
    audio fragment cannot collide with another op's internal labels. The
    surrounding ``[in]<chain>[out]`` wrapper labels are owned by the builder;
    this names only the op's *internal* intermediate streams."""


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
    wire and the registry key. Subclasses may override the ``category`` and
    ``requires`` ClassVars.

    ``predict_metadata`` defaults to identity; ``to_ffmpeg_filter`` defaults to
    ``None`` (no filter compilation).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    op: str

    category: ClassVar[OpCategory] = OpCategory.SPECIAL
    requires: ClassVar[tuple[str, ...]] = ()
    compiles_from_source: ClassVar[bool] = False
    """Whether the op's filter compile decodes the source itself (face_crop's
    detection pass). Such ops cannot sit at the encode stage -- the frames
    behind per-frame Python effects are not reproducible at compile time --
    so both the plan builder and the streamability report reject them as
    UNSTREAMABLE there."""
    changes_duration: ClassVar[bool] = False
    """Whether the op's output duration differs from its input (speed, freeze).

    The streaming plan builder folds ``predict_metadata`` through the chain
    either way; this flag additionally gates time-based *context*: a
    context-consuming op scheduled after a duration-changing transform would
    receive timestamps on the wrong timeline, so such plans are rejected as
    UNSTREAMABLE until context re-mapping exists. The streamability report
    mirrors the same rule.
    """
    llm_exposed: ClassVar[bool] = True
    internal_only: ClassVar[bool] = False
    """Whether this op is engine-internal and must NOT be a chain op.

    ``CutSeconds``/``CutFrames`` trim a segment, but trimming is the segment's own
    ``start``/``end`` mechanism -- the engine constructs them directly. They have
    no ffmpeg filter and no ``process_frame``, so this flag keeps them OUT of the
    registry: they cannot appear in a plan's ``operations`` list or the LLM
    schema, while direct construction (``CutSeconds(start=..., end=...)``) still
    works. Default False (a normal chain op)."""
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

        if cls.internal_only:
            # Engine-internal op (CutSeconds/CutFrames): the op Literal is still
            # validated above, but the op is kept out of the registry so it
            # cannot be resolved as a chain op or exposed to the LLM.
            return

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

    def predict_metadata(self, meta: VideoMetadata) -> VideoMetadata:
        """Predict output metadata from input metadata. Default: identity.

        Run during ``VideoEdit.validate()``'s dry-run, before any frames are
        decoded. Beyond predicting shape, this is the fail-fast gate, and it
        has one contract: **reject exactly the plans that would otherwise crash
        or do unrecoverable / expensive work in** ``run_to_file()``;
        anything ``run_to_file()`` can absorb by graceful degradation is NOT rejected.
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
        """Compile to an ffmpeg ``-vf`` filter expression, or ``None`` for no filter compilation.

        Streamable transforms override this. Effects use ``process_frame``
        instead -- they do not go through ffmpeg filters.
        """
        return None

    def to_ffmpeg_audio_filter(self, ctx: FilterCtx) -> str | None:
        """Compile the op's audio-domain twin to an ffmpeg audio-filter expression.

        The audio analogue of :meth:`to_ffmpeg_filter`: segment audio now
        streams through the SAME ffmpeg process as the video (a second
        ``-i source`` input routed through ``-filter_complex``), so a
        duration-changing transform expresses its audio effect as a filter on
        that graph instead of mutating an in-memory ``Audio`` array
        (``speed_change`` -> ``atempo``, ``freeze_frame`` -> silence splice,
        ``silence_removal`` -> ``aselect`` keep windows, ``fade`` -> ``afade``,
        ``volume_adjust`` -> ``volume``).

        ``ctx`` is the SAME :class:`FilterCtx` the video side builds at this
        op's plan position -- ``ctx.fps``/``ctx.frame_count`` are the
        already-folded values, ``ctx.context`` carries the resolved,
        segment-local ``requires`` -- so the audio chain stays in lockstep
        with the video chain. The returned expression is a comma-joined
        single-input/single-output filter sub-chain (e.g. ``"atempo=2.0"``);
        the plan builder appends it to the segment's labeled audio graph at
        the same stage (decode/encode) it appends the video filter. ``None``
        means "no audio effect" -- the default, so the builder only emits a
        filter for the four audio-affecting ops.
        """
        return None

    def streams(self) -> bool:
        """Whether this op streams in O(1) memory at its plan position.

        Structural replacement for the former ``streamable`` ClassVar: a transform
        streams iff it overrides :meth:`to_ffmpeg_filter`. :class:`Effect` widens
        this (a frame effect streams via ``process_frame``; a filter effect via
        ``compiles_to_filter``). The one case structure cannot express -- the
        override exists but the filter compiles to ``None`` at this position -- is
        caught at runtime by the ``STREAMING_UNSUPPORTED`` raise in
        ``VideoEdit._compile_streaming_plans``.
        """
        return type(self).to_ffmpeg_filter is not Operation.to_ffmpeg_filter


class Effect(Operation):
    """Operation that preserves shape and frame count, driven by per-frame streaming.

    Subclasses implement the streaming contract -- :meth:`process_frame` (and
    :meth:`streaming_init` for any precomputed per-stream state) -- which is the
    single source of truth for the effect's pixel logic. The streaming engine
    in ``editing/streaming.py`` drives that contract for bounded-memory
    execution, resolving :attr:`window` against the segment timeline so frames
    outside the window pass through untouched.

    Effects that compile to a native ffmpeg filter instead set
    :attr:`compiles_to_filter` and implement :meth:`to_ffmpeg_filter` (and, for
    audio-coupled effects like ``Fade``/``VolumeAdjust``,
    :meth:`to_ffmpeg_audio_filter`) so the window stays coherent across the
    decode/encode graph.
    """

    category: ClassVar[OpCategory] = OpCategory.EFFECT
    audio_coupled: ClassVar[bool] = False
    """Whether the effect mutates audio alongside pixels (``afade``/``volume``)."""

    window: TimeRange | None = Field(
        None,
        description="Time window for the effect in seconds. Omit to apply across the full duration.",
    )

    @property
    def compiles_to_filter(self) -> bool:
        """Whether this effect joins the decode filter chain instead of scheduling per-frame Python.

        When True, the streaming plan builder calls :meth:`to_ffmpeg_filter`
        (with the segment's resolved context on the :class:`FilterCtx`) and, if
        it compiles, appends the result to the vf chain at this op's plan
        position -- the Filter class of the streaming contract. Instance-level
        rather than a ClassVar because it may depend on field values (e.g.
        ``add_subtitles``'s ``renderer``). False by default: effects normally
        stream via ``streaming_init``/``process_frame``.
        """
        return False

    def predict_metadata(self, meta: VideoMetadata, **_context: Any) -> VideoMetadata:
        """Effects preserve shape and frame count, so the prediction is identity.

        Accepts ``**_context`` so requires-aware effects (``TranscriptionOverlay``)
        validate without subclasses needing to override just to widen the
        signature. Mirrors :meth:`Effect.streaming_init`'s ``**_context`` accept-all.
        """
        return meta

    def streaming_init(self, total_frames: int, fps: float, width: int, height: int, **_context: Any) -> None:
        """Hook for per-stream precomputation (per-frame alphas, sigma curves...).

        ``_context`` carries resolved ``requires`` values for context-aware
        effects (e.g. ``transcription=...`` for ``TranscriptionOverlay``),
        already re-based onto the local timeline by the runner. Effects that
        declare no ``requires`` are always called without context kwargs.

        Default: no-op. Override in subclasses that need it.
        """

    def process_frame(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Process one ``(H, W, 3) uint8`` frame in streaming mode.

        ``frame_index`` is 0-based within this effect's active window.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming")

    def streams(self) -> bool:
        """An effect streams via per-frame Python (``process_frame``) or a filter.

        Frame effects override :meth:`process_frame`; filter effects
        (``add_subtitles``, ``vignette``, ...) instead set
        :attr:`compiles_to_filter` and implement :meth:`to_ffmpeg_filter`.
        ``add_subtitles`` streams *only* via the filter path (it does not override
        ``process_frame``), so ``compiles_to_filter`` is consulted per-instance.
        """
        return type(self).process_frame is not Effect.process_frame or self.compiles_to_filter

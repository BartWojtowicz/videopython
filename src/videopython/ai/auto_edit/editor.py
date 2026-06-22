"""AutoEditor: turn source videos + a brief into a runnable VideoEdit."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

from videopython.ai.errors import AiError

from .backend import PlannerError, StructuredVisionLLM
from .catalog import build_catalog
from .models import CatalogBundle, CatalogScene, EditPlan
from .resolve import UnknownSceneIdsError, resolve_plan

if TYPE_CHECKING:
    import numpy as np

    from videopython.ai.video_analysis import VideoAnalysis, VideoAnalyzer
    from videopython.base.video import VideoMetadata
    from videopython.editing import VideoEdit

_SYSTEM_PROMPT = (
    "You are a video editor. You are given a creative brief and a catalog of candidate scenes "
    "from one or more source videos, each shown with its keyframe. Select and order scenes into "
    "an edit that fulfils the brief. Reference scenes only by their catalog id via 'scene_id'; "
    "never invent timestamps or sources."
)

_NormalizeTarget = tuple[int, int] | Literal["first", "largest", "match"]


class AutoEditError(AiError, RuntimeError):
    """The planner could not produce a valid edit within the retry budget."""


class AutoEditor:
    """Plan a VideoEdit from source videos using a structured-vision planner."""

    def __init__(
        self,
        planner: StructuredVisionLLM,
        *,
        analyzer: VideoAnalyzer | None = None,
        max_rounds: int = 3,
        normalize_target: _NormalizeTarget = "largest",
    ) -> None:
        self.planner = planner
        self._analyzer = analyzer
        self.max_rounds = max_rounds
        self.normalize_target = normalize_target

    def edit(self, sources: Sequence[str | Path], brief: str, *, context: dict[str, Any] | None = None) -> VideoEdit:
        """Analyze ``sources`` and plan an edit for ``brief`` (runs the analyzer)."""
        analyses = [self._get_analyzer().analyze_path(source) for source in sources]
        return self.edit_from_analyses(analyses, brief, context=context)

    def edit_from_analyses(
        self, analyses: Sequence[VideoAnalysis], brief: str, *, context: dict[str, Any] | None = None
    ) -> VideoEdit:
        """Plan an edit from precomputed VideoAnalysis results (no model download)."""
        bundle = build_catalog(analyses)
        metadata = _metadata_by_source(analyses)
        run_context = _merge_context(analyses, context)
        schema = EditPlan.json_schema(strict=True)
        base_text, images = _build_prompt(brief, bundle)

        feedback: str | None = None
        for _ in range(self.max_rounds):
            text = base_text if feedback is None else f"{base_text}\n\n{feedback}"
            try:
                raw = self.planner.generate_json(system=_SYSTEM_PROMPT, text=text, images=images or None, schema=schema)
                edit = resolve_plan(EditPlan.model_validate(raw), bundle.catalog)
            except (PlannerError, ValidationError, UnknownSceneIdsError) as exc:
                feedback = _shape_feedback(exc)
                continue
            edit, _ = edit.repair(metadata, context=run_context, clamp_segment_end=True)
            edit, _ = edit.normalize_dimensions(metadata, self.normalize_target, context=run_context)
            errors = edit.check(metadata, context=run_context)
            if not errors:
                return edit
            feedback = "The previous plan had these problems:\n" + "\n".join(e.to_prompt_line() for e in errors)

        raise AutoEditError(f"No valid edit after {self.max_rounds} round(s). Last feedback:\n{feedback}")

    def _get_analyzer(self) -> VideoAnalyzer:
        if self._analyzer is None:
            from videopython.ai.video_analysis import VideoAnalyzer

            self._analyzer = VideoAnalyzer()
        return self._analyzer


def _build_prompt(brief: str, bundle: CatalogBundle) -> tuple[str, list[np.ndarray]]:
    """The planner prompt: scene-line text plus the keyframes, in catalog order."""
    lines = [f"Brief: {brief}\n\nCandidate scenes:"]
    images: list[np.ndarray] = []
    for scene in bundle.catalog.scenes:
        lines.append(_scene_line(scene))
        frame = bundle.keyframes.get(scene.id)
        if frame is not None:
            images.append(frame)
    return "\n\n".join(lines), images


def _scene_line(scene: CatalogScene) -> str:
    bits = [f"[{scene.id}] {scene.duration:.1f}s"]
    if scene.shot_type:
        bits.append(scene.shot_type)
    if scene.caption:
        bits.append(scene.caption)
    line = " | ".join(bits)
    if scene.transcript:
        line += f"\n  speech: {scene.transcript}"
    return line


def _metadata_by_source(analyses: Sequence[VideoAnalysis]) -> dict[str, VideoMetadata]:
    from videopython.base.video import VideoMetadata

    paths: set[str] = set()
    for analysis in analyses:
        if analysis.source.path is not None:
            paths.add(analysis.source.path)
    return {str(Path(p)): VideoMetadata.from_path(p) for p in paths}


def _merge_context(analyses: Sequence[VideoAnalysis], context: dict[str, Any] | None) -> dict[str, Any] | None:
    transcriptions: dict[str, Any] = {}
    for analysis in analyses:
        src = analysis.source
        if src.path is not None and analysis.audio is not None and analysis.audio.transcription is not None:
            transcriptions[str(Path(src.path))] = analysis.audio.transcription
    if not transcriptions:
        return context
    return {"transcription": transcriptions, **(context or {})}


def _shape_feedback(exc: PlannerError | ValidationError | UnknownSceneIdsError) -> str:
    if isinstance(exc, UnknownSceneIdsError):
        return f"Plan used scene ids not in the catalog: {sorted(set(exc.ids))}. Use only listed ids."
    if isinstance(exc, PlannerError):
        return f"Your previous response was not usable JSON ({exc}). Return one matching JSON object."
    return f"The previous plan did not match the schema: {exc.errors()!r}"

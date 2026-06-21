"""videopython MCP server (stdio): auto-edit primitives as MCP tools.

The MCP client's model is the planner. These tools expose the steps it drives:
analyze each source, build a keyframe catalog, then author an ``EditPlan`` that
references catalog scenes by ``scene_id`` and validate / repair / run it. The
server caches analyses + the catalog so the agent passes small payloads (scene
ids), not whole analysis blobs.
"""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from pydantic import ValidationError

from videopython.ai._ollama import _downscale, _encode_png_b64
from videopython.ai.auto_edit import EditPlan, UnknownSceneIdsError, resolve_plan
from videopython.ai.auto_edit import build_catalog as _build_scene_catalog
from videopython.editing import VideoEdit

if TYPE_CHECKING:
    from videopython.ai.auto_edit import CatalogBundle
    from videopython.ai.video_analysis import VideoAnalysis, VideoAnalyzer
    from videopython.base import PlanError, PlanRepair

mcp = FastMCP("videopython")

# Session state for one stdio client (one process per client).
_analyses: dict[str, VideoAnalysis] = {}
_analyzers: dict[str, VideoAnalyzer] = {}
_bundle: CatalogBundle | None = None

_MAX_INLINE_KEYFRAMES = 12  # cap inlined keyframes in build_catalog; pull the rest with scene_keyframes


@mcp.tool()
def analyze_video(path: str, profile: Literal["full", "editing"] = "full") -> dict[str, Any]:
    """Analyze a source video (scenes, transcript, captions) and cache it for build_catalog.

    ``profile="editing"`` skips audio classification (faster on long sources); the
    catalog (captions + transcript + face flag) is unaffected. Returns a short
    summary; call this once per source, then build_catalog.
    """
    # Heavy analyzer deps (e.g. transnetv2-pytorch) bare-print to stdout, which here is
    # the stdio JSON-RPC channel; send that to stderr so the transport stays clean.
    with redirect_stdout(sys.stderr):
        analysis = _get_analyzer(profile).analyze_path(path)
    _analyses[str(Path(path))] = analysis
    src = analysis.source
    return {
        "source": str(Path(path)),
        "duration": src.duration,
        "fps": src.fps,
        "width": src.width,
        "height": src.height,
        "scenes": len(analysis.scenes.samples) if analysis.scenes else 0,
    }


@mcp.tool(structured_output=False)
def build_catalog(sources: list[str] | None = None) -> list[TextContent | ImageContent]:
    """Build the candidate-scene catalog from analyzed videos and cache it.

    The first text block is the full catalog JSON (id/duration/shot_type/caption/
    transcript per scene -- enough to shortlist from text alone). Up to
    ``_MAX_INLINE_KEYFRAMES`` downscaled keyframes follow; a final note names any
    scene ids whose images were omitted (fetch them with scene_keyframes). Pass
    ``sources`` to restrict to specific analyzed paths, or omit for all.
    """
    global _bundle
    analyses = _selected_analyses(sources)
    if not analyses:
        raise ValueError("No analyzed videos cached; call analyze_video first.")
    _bundle = _build_scene_catalog(analyses)

    ids = [scene.id for scene in _bundle.catalog.scenes]
    inlined, omitted = ids[:_MAX_INLINE_KEYFRAMES], ids[_MAX_INLINE_KEYFRAMES:]
    blocks: list[TextContent | ImageContent] = [
        TextContent(type="text", text=_bundle.catalog.model_dump_json()),
        *_keyframe_blocks(inlined),
    ]
    if omitted:
        blocks.append(
            TextContent(
                type="text",
                text=f"Keyframes omitted for {len(omitted)} scenes "
                f"(call scene_keyframes for these ids): {', '.join(omitted)}",
            )
        )
    return blocks


@mcp.tool(structured_output=False)
def scene_keyframes(scene_ids: list[str]) -> list[TextContent | ImageContent]:
    """Return downscaled keyframe images for specific catalog scene ids (use after build_catalog)."""
    if _bundle is None:
        raise ValueError("No catalog cached; call build_catalog first.")
    known = _bundle.catalog.by_id()
    unknown = sorted({sid for sid in scene_ids if sid not in known})
    if unknown:
        error = {"code": "unknown_scene_ids", "value": unknown, "message": f"Unknown scene ids: {unknown}"}
        return [TextContent(type="text", text=json.dumps(error))]
    return _keyframe_blocks(list(dict.fromkeys(scene_ids)))


@mcp.tool()
def validate_edit(plan: dict[str, Any]) -> dict[str, Any]:
    """Validate an edit plan (an EditPlan referencing catalog scene ids).

    Returns every problem at once as structured errors; ``valid`` is True when
    the list is empty. A schema-invalid plan or an unknown scene id is reported
    the same way (error ``code`` ``schema_invalid`` / ``unknown_scene_ids``).
    """
    edit, errors = _resolve(plan)
    if edit is None:
        return {"valid": False, "errors": errors}
    errors = [_error_dict(e) for e in edit.check(_source_metadata(edit), context=_context())]
    return {"valid": not errors, "errors": errors}


@mcp.tool()
def repair_edit(plan: dict[str, Any]) -> dict[str, Any]:
    """Repair the mechanical issues in an edit plan and normalize segment dimensions.

    Returns the resolved+repaired VideoEdit and a changelog, for inspection;
    ``edit`` is None with structured ``errors`` if the plan does not resolve. The
    returned ``edit`` is a concrete VideoEdit, not an EditPlan -- keep refining by
    adjusting the original by-id plan, not by resubmitting this output. run_edit
    applies the same repair before rendering.
    """
    edit, errors = _resolve(plan)
    if edit is None:
        return {"edit": None, "repairs": [], "errors": errors}
    metadata = _source_metadata(edit)
    context = _context()
    edit, repairs = edit.repair(metadata, context=context, clamp_segment_end=True)
    edit, dim_repairs = edit.normalize_dimensions(metadata, "largest", context=context)
    return {"edit": edit.to_dict(), "repairs": [_repair_dict(r) for r in (*repairs, *dim_repairs)], "errors": []}


@mcp.tool()
def run_edit(plan: dict[str, Any], output_path: str) -> dict[str, Any]:
    """Render an edit plan to an MP4 file (the path suffix is normalized to .mp4).

    Resolves scene ids, repairs + normalizes, validates, then renders. If the
    plan does not resolve or validation still fails, returns structured ``errors``
    and ``output_path`` None instead of rendering.
    """
    edit, errors = _resolve(plan)
    if edit is None:
        return {"output_path": None, "errors": errors}
    metadata = _source_metadata(edit)
    context = _context()
    edit, _ = edit.repair(metadata, context=context, clamp_segment_end=True)
    edit, _ = edit.normalize_dimensions(metadata, "largest", context=context)
    errors = [_error_dict(e) for e in edit.check(metadata, context=context)]
    if errors:
        return {"output_path": None, "errors": errors}
    out = edit.run_to_file(output_path, context=context)
    return {"output_path": str(out), "errors": []}


@mcp.resource("schema://videopython/edit-plan", mime_type="application/json")
def edit_plan_schema() -> str:
    """JSON Schema for the edit plan an agent authors (references catalog scene ids)."""
    return json.dumps(EditPlan.json_schema(strict=True))


def _get_analyzer(profile: str = "full") -> VideoAnalyzer:
    if profile not in _analyzers:
        from videopython.ai.video_analysis import VideoAnalysisConfig, VideoAnalyzer

        _analyzers[profile] = VideoAnalyzer(config=VideoAnalysisConfig.for_profile(profile))
    return _analyzers[profile]


def _selected_analyses(sources: list[str] | None) -> list[VideoAnalysis]:
    if sources is None:
        return list(_analyses.values())
    return [_analyses[str(Path(s))] for s in sources if str(Path(s)) in _analyses]


def _keyframe_blocks(scene_ids: list[str]) -> list[TextContent | ImageContent]:
    assert _bundle is not None  # callers guard; narrows the module global for mypy
    blocks: list[TextContent | ImageContent] = []
    for sid in scene_ids:
        frame = _bundle.keyframes.get(sid)
        if frame is not None:
            blocks.append(TextContent(type="text", text=f"scene {sid}:"))
            blocks.append(ImageContent(type="image", data=_encode_png_b64(_downscale(frame)), mimeType="image/png"))
    return blocks


def _resolve(plan: dict[str, Any]) -> tuple[VideoEdit | None, list[dict[str, Any]]]:
    """Resolve a by-id plan, or return (None, structured errors) for the agent's plan mistakes."""
    if _bundle is None:
        raise ValueError("No catalog cached; call build_catalog first.")
    try:
        edit = resolve_plan(EditPlan.model_validate(plan), _bundle.catalog)
    except ValidationError as exc:
        detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return None, [{"code": "schema_invalid", "detail": detail, "message": str(exc)}]
    except UnknownSceneIdsError as exc:
        return None, [{"code": "unknown_scene_ids", "value": sorted(set(exc.ids)), "message": str(exc)}]
    return edit, []


def _source_metadata(edit: VideoEdit) -> dict[str, Any]:
    from videopython.base.video import VideoMetadata

    return {str(seg.source): VideoMetadata.from_path(seg.source) for seg in edit.segments}


def _context() -> dict[str, Any] | None:
    transcriptions: dict[str, Any] = {}
    for analysis in _analyses.values():
        src = analysis.source
        if src.path is not None and analysis.audio is not None and analysis.audio.transcription is not None:
            transcriptions[str(Path(src.path))] = analysis.audio.transcription
    return {"transcription": transcriptions} if transcriptions else None


def _error_dict(error: PlanError) -> dict[str, Any]:
    return {
        "code": error.code.value,
        "location": error.location,
        "op": error.op,
        "field": error.field,
        "value": error.value,
        "limit": error.limit,
        "detail": error.detail,
        "message": error.to_prompt_line(),
    }


def _repair_dict(repair: PlanRepair) -> dict[str, Any]:
    return {
        "location": repair.location,
        "field": repair.field,
        "old": repair.old,
        "new": repair.new,
        "code": repair.code.value,
    }


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

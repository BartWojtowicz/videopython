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
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from pydantic import ValidationError

from videopython.ai._ollama import _encode_png_b64
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
_analyzer: VideoAnalyzer | None = None
_bundle: CatalogBundle | None = None


@mcp.tool()
def analyze_video(path: str) -> dict[str, Any]:
    """Analyze a source video (scenes, transcript, captions) and cache it for build_catalog.

    Returns a short summary; call this once per source, then build_catalog.
    """
    # Heavy analyzer deps (e.g. transnetv2-pytorch) bare-print to stdout, which here is
    # the stdio JSON-RPC channel; send that to stderr so the transport stays clean.
    with redirect_stdout(sys.stderr):
        analysis = _get_analyzer().analyze_path(path)
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

    Returns the catalog as JSON text plus one keyframe image per scene (each
    preceded by its scene id), so the model can see the footage. Pass ``sources``
    to restrict to specific analyzed paths, or omit for all. Author the edit by
    referencing the returned ``id`` values via the edit-plan schema resource.
    """
    global _bundle
    analyses = _selected_analyses(sources)
    if not analyses:
        raise ValueError("No analyzed videos cached; call analyze_video first.")
    _bundle = _build_scene_catalog(analyses)

    blocks: list[TextContent | ImageContent] = [TextContent(type="text", text=_bundle.catalog.model_dump_json())]
    for scene in _bundle.catalog.scenes:
        frame = _bundle.keyframes.get(scene.id)
        if frame is not None:
            blocks.append(TextContent(type="text", text=f"scene {scene.id}:"))
            blocks.append(ImageContent(type="image", data=_encode_png_b64(frame), mimeType="image/png"))
    return blocks


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


def _get_analyzer() -> VideoAnalyzer:
    global _analyzer
    if _analyzer is None:
        from videopython.ai.video_analysis import VideoAnalyzer

        _analyzer = VideoAnalyzer()
    return _analyzer


def _selected_analyses(sources: list[str] | None) -> list[VideoAnalysis]:
    if sources is None:
        return list(_analyses.values())
    return [_analyses[str(Path(s))] for s in sources if str(Path(s)) in _analyses]


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

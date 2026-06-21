"""Tests for the videopython MCP server tools (call the tool functions directly)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp.types import ImageContent, TextContent  # noqa: E402

from tests.test_config import SMALL_VIDEO_PATH  # noqa: E402
from videopython.ai.video_analysis.models import (  # noqa: E402
    AnalysisRunInfo,
    SceneAnalysisSample,
    SceneAnalysisSection,
    VideoAnalysis,
    VideoAnalysisConfig,
    VideoAnalysisSource,
)
from videopython.base import PlanError, PlanErrorCode, PlanRepair  # noqa: E402
from videopython.base.description import SceneDescription  # noqa: E402
from videopython.base.video import VideoMetadata  # noqa: E402
from videopython.mcp import server  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state() -> Any:
    server._analyses = {}
    server._bundle = None
    server._analyzer = None
    yield
    server._analyses = {}
    server._bundle = None
    server._analyzer = None


def _real_analysis() -> VideoAnalysis:
    meta = VideoMetadata.from_path(SMALL_VIDEO_PATH)
    dur = meta.total_seconds
    scenes = [
        SceneAnalysisSample(
            scene_index=0,
            start_second=0.0,
            end_second=dur / 2,
            scene_description=SceneDescription(caption="intro", subjects=[], shot_type="wide"),
        ),
        SceneAnalysisSample(
            scene_index=1,
            start_second=dur / 2,
            end_second=dur,
            scene_description=SceneDescription(caption="outro", subjects=[], shot_type="close-up"),
        ),
    ]
    return VideoAnalysis(
        source=VideoAnalysisSource(
            path=str(SMALL_VIDEO_PATH),
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            frame_count=meta.frame_count,
            duration=dur,
        ),
        config=VideoAnalysisConfig(),
        run_info=AnalysisRunInfo(created_at="2026-01-01T00:00:00Z", mode="path"),
        scenes=SceneAnalysisSection(samples=scenes),
    )


def _stem() -> str:
    return Path(SMALL_VIDEO_PATH).stem


def test_edit_plan_schema_resource() -> None:
    schema = json.loads(server.edit_plan_schema())
    assert schema["type"] == "object"
    assert "segments" in schema["properties"]
    assert "resize" in json.dumps(schema)  # operation union embedded


def test_build_catalog_returns_catalog_and_keyframes() -> None:
    server._analyses = {str(SMALL_VIDEO_PATH): _real_analysis()}
    blocks = server.build_catalog()

    texts = [b for b in blocks if isinstance(b, TextContent)]
    images = [b for b in blocks if isinstance(b, ImageContent)]
    catalog = json.loads(texts[0].text)
    assert len(catalog["scenes"]) == 2
    assert len(images) == 2  # one keyframe image per scene
    assert all(img.mimeType == "image/png" for img in images)
    assert server._bundle is not None  # cached for later tools


def test_build_catalog_without_analyses_errors() -> None:
    with pytest.raises(ValueError, match="analyze_video"):
        server.build_catalog()


def test_validate_edit_clean_plan() -> None:
    server._analyses = {str(SMALL_VIDEO_PATH): _real_analysis()}
    server.build_catalog()
    stem = _stem()
    result = server.validate_edit({"segments": [{"scene_id": f"{stem}#0"}, {"scene_id": f"{stem}#1"}]})
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_edit_without_catalog_errors() -> None:
    with pytest.raises(ValueError, match="catalog"):
        server.validate_edit({"segments": [{"scene_id": "x"}]})


def test_analyze_video_caches_and_summarizes() -> None:
    analysis = _real_analysis()

    class _FakeAnalyzer:
        def analyze_path(self, path: str) -> VideoAnalysis:
            return analysis

    server._analyzer = _FakeAnalyzer()  # type: ignore[assignment]
    out = server.analyze_video(str(SMALL_VIDEO_PATH))
    assert out["scenes"] == 2
    assert out["source"] == str(SMALL_VIDEO_PATH)
    assert str(SMALL_VIDEO_PATH) in server._analyses


def test_run_edit_renders(tmp_path: Path) -> None:
    server._analyses = {str(SMALL_VIDEO_PATH): _real_analysis()}
    server.build_catalog()
    out_path = tmp_path / "out.mp4"
    result = server.run_edit({"segments": [{"scene_id": f"{_stem()}#0"}]}, str(out_path))
    assert result["errors"] == []
    assert result["output_path"]
    assert out_path.exists()


def test_validate_edit_unknown_scene_id_returns_structured_error() -> None:
    server._analyses = {str(SMALL_VIDEO_PATH): _real_analysis()}
    server.build_catalog()
    result = server.validate_edit({"segments": [{"scene_id": "does-not-exist#0"}]})
    assert result["valid"] is False
    assert result["errors"][0]["code"] == "unknown_scene_ids"


def test_validate_edit_schema_invalid_returns_structured_error() -> None:
    server._analyses = {str(SMALL_VIDEO_PATH): _real_analysis()}
    server.build_catalog()
    result = server.validate_edit({"segments": [{}]})  # scene_id is required
    assert result["valid"] is False
    assert result["errors"][0]["code"] == "schema_invalid"


def test_run_edit_resolve_failure_returns_errors(tmp_path: Path) -> None:
    server._analyses = {str(SMALL_VIDEO_PATH): _real_analysis()}
    server.build_catalog()
    result = server.run_edit({"segments": [{"scene_id": "missing#0"}]}, str(tmp_path / "out.mp4"))
    assert result["output_path"] is None
    assert result["errors"][0]["code"] == "unknown_scene_ids"


def test_repair_edit_returns_edit_and_changelog() -> None:
    server._analyses = {str(SMALL_VIDEO_PATH): _real_analysis()}
    server.build_catalog()
    result = server.repair_edit({"segments": [{"scene_id": f"{_stem()}#0"}]})
    assert result["errors"] == []
    assert "segments" in result["edit"]
    assert isinstance(result["repairs"], list)


def test_error_dict_includes_op() -> None:
    err = PlanError(
        code=PlanErrorCode.CUT_EXCEEDS_DURATION,
        location="segments[0].operations[0]",
        op="cut",
        field="end",
        value=5.0,
        limit=2.0,
    )
    out = server._error_dict(err)
    assert out["op"] == "cut"
    assert out["code"] == "cut_exceeds_duration"
    assert "end" in out["message"]


def test_repair_dict_shape() -> None:
    rep = PlanRepair(
        location="segments[0].operations[0]",
        field="window.stop",
        old=5.0,
        new=2.0,
        code=PlanErrorCode.EFFECT_WINDOW_EXCEEDS_DURATION,
    )
    assert server._repair_dict(rep) == {
        "location": "segments[0].operations[0]",
        "field": "window.stop",
        "old": 5.0,
        "new": 2.0,
        "code": "effect_window_exceeds_duration",
    }

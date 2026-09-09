import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.test_config import SMALL_VIDEO_PATH
from videopython.ai.video_analysis import VideoAnalysis, VideoAnalysisConfig, VideoAnalyzer, detectors, models
from videopython.mcp import server


def _analysis(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("Test analyzer failure")

    monkeypatch.setattr(detectors, "AudioToText", fail)
    source = tmp_path / "source.mp4"
    shutil.copyfile(SMALL_VIDEO_PATH, source)
    return VideoAnalyzer(config=VideoAnalysisConfig(enabled_analyzers={"audio_to_text"}), sampling="low").analyze_path(
        source
    )


def test_fresh_mcp_server_imports_exports_and_renders_without_analyzers(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path, monkeypatch)
    saved = tmp_path / "analysis.json"
    analysis.save(saved)
    bootstrap = """
from videopython.mcp import server

def forbidden(*args, **kwargs):
    raise RuntimeError("Saved analysis must not start inference")

server._get_analyzer = forbidden
server.main()
"""

    async def workflow():
        params = StdioServerParameters(command=sys.executable, args=["-c", bootstrap])
        async with stdio_client(params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                loaded = await session.call_tool("import_analysis", {"path": str(saved)})
                assert not loaded.isError
                assert loaded.structuredContent["provenance"] == analysis.provenance.model_dump()
                assert loaded.structuredContent["config"] == analysis.config.model_dump(mode="json")
                outcomes = loaded.structuredContent["analyzers"]
                assert any(row["analyzer"] == "audio_to_text" and row["status"] == "failed" for row in outcomes)
                assert all(value is None for value in loaded.structuredContent["provenance"]["models"].values())
                catalog = await session.call_tool("build_catalog", {"sources": [str(analysis.source.path)]})
                assert not catalog.isError
                scene = json.loads(catalog.content[0].text)["scenes"][0]
                notifications = []

                async def progress(number, total, message):
                    notifications.append((number, total, json.loads(message)))

                rendered = await session.call_tool(
                    "run_edit",
                    {"plan": {"segments": [{"scene_id": scene["id"]}]}, "output_path": str(tmp_path / "out.mp4")},
                    progress_callback=progress,
                )
                assert not rendered.isError
                assert [n for n, _, _ in notifications] == list(range(1, len(notifications) + 1))
                assert notifications[-1][2]["stage"] == "complete" and notifications[-1][2]["finished"]
                assert any(e["stage"] == "segment" and e["completed"] > 0 for _, _, e in notifications)
                assert Path(rendered.structuredContent["output_path"]).exists()
                exported = await session.call_tool(
                    "export_analysis",
                    {"source": str(analysis.source.path), "output_path": str(tmp_path / "export.json")},
                )
                assert not exported.isError
                assert VideoAnalysis.load(tmp_path / "export.json") == analysis

    asyncio.run(workflow())


def test_import_export_hash_once_and_reject_changed_source(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path, monkeypatch)
    saved = tmp_path / "analysis.json"
    analysis.save(saved)
    monkeypatch.setattr(server, "_analyses", {})
    monkeypatch.setattr(server, "_bundle", None)
    original = models.source_digest
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(models, "source_digest", counted)
    server.import_analysis(str(saved))
    assert len(calls) == 1
    server.export_analysis(str(analysis.source.path), str(tmp_path / "export.json"))
    assert len(calls) == 2
    with Path(analysis.source.path).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Source content differs"):
        server.import_analysis(str(saved))
    with pytest.raises(ValueError, match="Source content differs"):
        server.export_analysis(str(analysis.source.path), str(tmp_path / "rejected.json"))
    assert not (tmp_path / "rejected.json").exists()


def test_saved_analysis_requires_current_format_and_file_identity(tmp_path, monkeypatch):
    analysis = _analysis(tmp_path, monkeypatch)
    payload = analysis.model_dump(mode="json")
    payload["provenance"]["format_version"] = 2
    with pytest.raises(ValueError, match="format_version"):
        VideoAnalysis.model_validate(payload)
    del payload["provenance"]
    with pytest.raises(ValueError, match="provenance"):
        VideoAnalysis.model_validate(payload)
    analysis.provenance.source_sha256 = None
    with pytest.raises(ValueError, match="no verified file identity"):
        analysis.verify_source()

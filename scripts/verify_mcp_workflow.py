#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from time import perf_counter
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ollama_version() -> str:
    result = subprocess.run(["ollama", "--version"], check=True, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def _serve(vision_model: str) -> None:
    from videopython.ai.video_analysis import AUDIO_TO_TEXT, SCENE_VLM, VideoAnalysisConfig, VideoAnalyzer
    from videopython.mcp import server

    config = VideoAnalysisConfig.for_profile("editing")
    config.analyzer_params[AUDIO_TO_TEXT] = {"enable_vad": False}
    config.analyzer_params[SCENE_VLM] = {"model": vision_model}
    server._analyzers["editing"] = VideoAnalyzer(config=config)
    server.main()


def _structured(result: Any, tool: str) -> dict[str, Any]:
    if result.isError:
        messages = [block.text for block in result.content if getattr(block, "type", None) == "text"]
        raise RuntimeError(f"{tool} failed: {' '.join(messages)}")
    if result.structuredContent is None:
        raise RuntimeError(f"{tool} did not return structured content")
    return result.structuredContent


def _catalog(result: Any) -> dict[str, Any]:
    if result.isError:
        raise RuntimeError("build_catalog failed")
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return json.loads(block.text)
    raise RuntimeError("build_catalog did not return catalog JSON")


def _require_analyzers(outcomes: list[dict[str, Any]]) -> None:
    actual = {item["analyzer"]: (item["status"], item["reason"]) for item in outcomes}
    expected = {
        "audio_to_text": ("completed", None),
        "semantic_scene_detector": ("completed", None),
        "scene_vlm": ("completed", None),
        "audio_classifier": ("skipped", "disabled"),
        "face_tracker": ("completed", None),
    }
    if actual != expected:
        raise RuntimeError(f"unexpected analyzer outcomes: {actual}")


async def _run(source: Path, workdir: Path, vision_model: str) -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from videopython import VideoMetadata

    source = source.resolve()
    workdir = workdir.resolve()
    report_path = workdir / "report.json"
    output_path = workdir / "rendered.mp4"
    source_metadata = VideoMetadata.from_path(source)
    report: dict[str, Any] = {
        "status": "running",
        "environment": {
            "machine": platform.machine(),
            "ollama": _ollama_version(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "videopython": importlib_metadata.version("videopython"),
            "vision_model": vision_model,
        },
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "bytes": source.stat().st_size,
            "width": source_metadata.width,
            "height": source_metadata.height,
            "fps": source_metadata.fps,
            "frames": source_metadata.frame_count,
            "duration_seconds": source_metadata.total_seconds,
        },
        "steps": {},
    }
    _write_report(report_path, report)

    server = StdioServerParameters(
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "--serve", "--vision-model", vision_model],
    )
    started = perf_counter()
    try:
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                step_started = perf_counter()
                analysis = _structured(
                    await session.call_tool("analyze_video", {"path": str(source), "profile": "editing"}),
                    "analyze_video",
                )
                report["steps"]["analyze_video"] = {
                    "elapsed_seconds": round(perf_counter() - step_started, 3),
                    "result": analysis,
                }
                _write_report(report_path, report)
                _require_analyzers(analysis["analyzers"])

                step_started = perf_counter()
                catalog = _catalog(await session.call_tool("build_catalog", {"sources": [str(source)]}))
                scenes = catalog["scenes"]
                report["steps"]["build_catalog"] = {
                    "elapsed_seconds": round(perf_counter() - step_started, 3),
                    "scene_count": len(scenes),
                    "scenes": scenes,
                }
                _write_report(report_path, report)
                if not scenes or not any(scene["caption"].strip() for scene in scenes):
                    raise RuntimeError("catalog does not contain a model caption")

                plan = {"segments": [{"scene_id": scene["id"]} for scene in scenes[:2]]}
                step_started = perf_counter()
                validation = _structured(await session.call_tool("validate_edit", {"plan": plan}), "validate_edit")
                report["steps"]["validate_edit"] = {
                    "elapsed_seconds": round(perf_counter() - step_started, 3),
                    "plan": plan,
                    "result": validation,
                }
                _write_report(report_path, report)
                if not validation["valid"]:
                    raise RuntimeError(f"representative edit plan is invalid: {validation['errors']}")

                step_started = perf_counter()
                rendered = _structured(
                    await session.call_tool("run_edit", {"plan": plan, "output_path": str(output_path)}),
                    "run_edit",
                )
                if rendered["errors"] or rendered["output_path"] is None:
                    raise RuntimeError(f"run_edit returned errors: {rendered['errors']}")
                rendered_path = Path(rendered["output_path"])
                rendered_metadata = VideoMetadata.from_path(rendered_path)
                report["steps"]["run_edit"] = {
                    "elapsed_seconds": round(perf_counter() - step_started, 3),
                    "result": rendered,
                    "output": {
                        "bytes": rendered_path.stat().st_size,
                        "sha256": _sha256(rendered_path),
                        "width": rendered_metadata.width,
                        "height": rendered_metadata.height,
                        "fps": rendered_metadata.fps,
                        "frames": rendered_metadata.frame_count,
                        "duration_seconds": rendered_metadata.total_seconds,
                    },
                }

        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = round(perf_counter() - started, 3)
        _write_report(report_path, report)

    print(report_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the real MCP editing workflow over stdio.")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--vision-model", default="gemma3:12b")
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.serve and (args.source is None or args.workdir is None):
        parser.error("--source and --workdir are required")
    return args


def main() -> None:
    args = _parse_args()
    if args.serve:
        _serve(args.vision_model)
    else:
        asyncio.run(_run(args.source, args.workdir, args.vision_model))


if __name__ == "__main__":
    main()

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _ffmpeg_output(*args: str) -> str:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def _check_ffmpeg() -> None:
    _ffmpeg_output("ffmpeg", "-version")
    _ffmpeg_output("ffprobe", "-version")
    encoders = _ffmpeg_output("ffmpeg", "-hide_banner", "-encoders")
    filters = _ffmpeg_output("ffmpeg", "-hide_banner", "-filters")

    for encoder in ("libx264", "aac"):
        if encoder not in encoders:
            raise RuntimeError(f"FFmpeg encoder is missing: {encoder}")
    for filter_name in ("xfade", "acrossfade", "subtitles"):
        if filter_name not in filters:
            raise RuntimeError(f"FFmpeg filter is missing: {filter_name}")


def _check_render() -> None:
    import videopython
    import videopython.ai as ai
    import videopython.audio as audio
    import videopython.editing as editing

    assert videopython.__all__ == ["Video", "VideoMetadata"]
    assert {"VideoAnalyzer", "AutoEditor"} <= set(ai.__all__)
    assert {"Audio", "AudioMetadata"} <= set(audio.__all__)
    assert {"Operation", "VideoEdit"} <= set(editing.__all__)

    source = Path(__file__).parents[1] / "src/tests/test_data/small_video.mp4"
    metadata = videopython.VideoMetadata.from_path(source)
    edit = editing.VideoEdit.from_dict(
        {
            "segments": [
                {
                    "source": str(source),
                    "start": 0.0,
                    "end": min(0.5, metadata.total_seconds),
                }
            ]
        }
    )

    with tempfile.TemporaryDirectory() as directory:
        output = edit.run_to_file(Path(directory) / "smoke.mp4")
        rendered = videopython.VideoMetadata.from_path(output)
        assert rendered.frame_count > 0


async def _check_mcp() -> None:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    command = shutil.which("videopython-mcp", path=str(Path(sys.executable).parent))
    if command is None:
        raise RuntimeError("The videopython-mcp console script is missing")

    server = StdioServerParameters(command=command)
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {tool.name for tool in (await session.list_tools()).tools}
            resources = {str(resource.uri) for resource in (await session.list_resources()).resources}

    assert tools == {
        "analyze_video",
        "build_catalog",
        "export_analysis",
        "import_analysis",
        "repair_edit",
        "run_edit",
        "scene_keyframes",
        "scene_transcripts",
        "validate_edit",
    }
    assert resources == {"schema://videopython/edit-plan"}


def main() -> None:
    _check_ffmpeg()
    _check_render()
    asyncio.run(_check_mcp())
    print("wheel smoke passed")


if __name__ == "__main__":
    main()

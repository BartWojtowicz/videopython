"""Regression test: AI ops must be in the op registry / plan schema.

``FaceTrackingCrop`` and ``ObjectDetectionOverlay`` register only as an import
side-effect of their class definition, and ``ai/__init__`` re-exports them
lazily. Before the ``ai.ops`` self-registration shim (imported by
``ai.auto_edit``), a fresh process that built ``EditPlan.json_schema()`` /
``Operation.llm_registry()`` -- e.g. the auto-edit planner or the MCP
``edit_plan_schema`` resource -- silently omitted both ops, so the LLM could
never select them and ``Operation.get("face_crop")`` raised "Unknown op_id".

These run in a clean subprocess so no prior import in the session's interpreter
can mask the bug.
"""

from __future__ import annotations

import subprocess
import sys

_AI_OP_IDS = ("face_crop", "object_detection_overlay")


def _run(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def test_auto_edit_import_registers_ai_ops() -> None:
    """Importing the auto_edit package alone registers the AI ops."""
    out = _run(
        "import videopython.ai.auto_edit  # noqa: F401\n"
        "from videopython.editing.operation import Operation\n"
        "print(' '.join(sorted(Operation.llm_registry())))\n"
    )
    registered = set(out.split())
    for op_id in _AI_OP_IDS:
        assert op_id in registered, f"{op_id!r} missing from llm_registry; registered={sorted(registered)}"


def test_edit_plan_schema_includes_ai_ops() -> None:
    """The strict EditPlan schema (used by the planner and MCP) lists the AI ops."""
    out = _run(
        "import json\n"
        "from videopython.ai.auto_edit import EditPlan\n"
        "print(json.dumps(EditPlan.json_schema(strict=True)))\n"
    )
    for op_id in _AI_OP_IDS:
        assert f'"{op_id}"' in out, f"{op_id!r} missing from EditPlan.json_schema()"

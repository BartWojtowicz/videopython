from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from videopython.editing import PlanError, PlanErrorCode, PlanRepair, VideoEdit

_FIXTURES = Path(__file__).with_name("fixtures")


def _without_schema_annotations(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_schema_annotations(item) for key, item in value.items() if key not in {"description", "title"}
        }
    if isinstance(value, list):
        return [_without_schema_annotations(item) for item in value]
    return value


def test_saved_video_edit_plan_round_trips() -> None:
    text = (_FIXTURES / "video_edit_plan.json").read_text(encoding="utf-8")
    assert VideoEdit.from_json(text).to_dict() == json.loads(text)


def _video_edit_contract() -> dict[str, Any]:
    schema = VideoEdit.json_schema()
    canonical = json.dumps(_without_schema_annotations(schema), sort_keys=True, separators=(",", ":")).encode()
    segment = schema["properties"]["segments"]["items"]
    operations = segment["properties"]["operations"]["items"]

    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "root_properties": sorted(schema["properties"]),
        "segment_properties": sorted(segment["properties"]),
        "operation_ids": sorted(operations["discriminator"]["mapping"]),
        "plan_error_codes": sorted(code.value for code in PlanErrorCode),
        "plan_error_fields": [field.name for field in dataclasses.fields(PlanError)],
        "plan_repair_fields": [field.name for field in dataclasses.fields(PlanRepair)],
    }


def test_video_edit_schema_contract() -> None:
    contract = json.loads((_FIXTURES / "video_edit_schema_contract.json").read_text(encoding="utf-8"))
    result = subprocess.run([sys.executable, __file__], check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == contract


if __name__ == "__main__":
    print(json.dumps(_video_edit_contract()))

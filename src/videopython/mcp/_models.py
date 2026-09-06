from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class SchemaIssue(BaseModel):
    """One Pydantic schema-validation issue."""

    model_config = ConfigDict(extra="forbid")

    loc: list[str | int]
    msg: str
    type: str


class McpError(BaseModel):
    """Structured error returned by an editing tool."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    location: str | None = None
    op: str | None = None
    field: str | None = None
    value: float | list[str] | None = None
    limit: float | None = None
    detail: str | list[SchemaIssue] | None = None


class McpRepair(BaseModel):
    """One change made while repairing an edit plan."""

    model_config = ConfigDict(extra="forbid")

    location: str
    field: str
    old: float | str | None
    new: float | str | None
    code: str


class AnalyzeVideoResult(BaseModel):
    """Summary returned after analysis is cached."""

    model_config = ConfigDict(extra="forbid")

    source: str
    duration: float | None
    fps: float | None
    width: int | None
    height: int | None
    scenes: int


class ValidateEditResult(BaseModel):
    """Result of edit-plan validation."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    errors: list[McpError]


class RepairEditResult(BaseModel):
    """Concrete repaired edit and its change records."""

    model_config = ConfigDict(extra="forbid")

    edit: dict[str, Any] | None
    repairs: list[McpRepair]
    errors: list[McpError]


class RunEditResult(BaseModel):
    """Rendered output path or the errors that prevented rendering."""

    model_config = ConfigDict(extra="forbid")

    output_path: str | None
    errors: list[McpError]

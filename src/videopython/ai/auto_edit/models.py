"""Wire types for LLM-authored editing: the scene catalog and the by-id plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from videopython.editing._schema import array_field_schema, field_schema, optional_model_field_schema
from videopython.editing.operation import Operation, _to_strict_schema
from videopython.editing.video_edit import OperationInput, TransitionSpec


class CatalogScene(BaseModel):
    """A candidate scene the planner picks by id; carries the exact source bounds."""

    id: str
    source: Path
    start: float
    end: float
    duration: float
    shot_type: str | None = None
    caption: str = ""
    transcript: str = ""
    has_speech: bool = False
    has_faces: bool = False


class EditCatalog(BaseModel):
    """The candidate scenes across all source videos."""

    scenes: list[CatalogScene]

    def by_id(self) -> dict[str, CatalogScene]:
        return {scene.id: scene for scene in self.scenes}


@dataclass
class CatalogBundle:
    """An EditCatalog plus its keyframes (numpy stays out of the pydantic models)."""

    catalog: EditCatalog
    keyframes: dict[str, np.ndarray] = field(default_factory=dict)


class PlanSegment(BaseModel):
    """One plan segment: a chosen scene id with its operations and optional transition."""

    model_config = ConfigDict(extra="forbid")

    scene_id: str = Field(description="id of a scene from the catalog to place at this position.")
    operations: list[OperationInput] = Field(default_factory=list, description="Operations to apply to this scene.")
    transition_in: TransitionSpec | None = Field(
        default=None, description="Optional crossfade from the previous segment into this one."
    )


class EditPlan(BaseModel):
    """The planner's output: an ordered selection of catalog scenes, referenced by id."""

    model_config = ConfigDict(extra="forbid")

    segments: list[PlanSegment] = Field(min_length=1, description="Ordered plan segments.")
    post_operations: list[OperationInput] = Field(
        default_factory=list, description="Operations applied once to the whole assembled program."
    )

    @classmethod
    def json_schema(cls, *, strict: bool = False) -> dict[str, Any]:
        """The by-id mirror of VideoEdit.json_schema, reusing the op union and strict rewrite."""
        op_schema = Operation.json_schema()

        segment_schema: dict[str, Any] = {
            "type": "object",
            "description": PlanSegment.__doc__,
            "properties": {
                "scene_id": field_schema(PlanSegment, "scene_id"),
                "operations": array_field_schema(PlanSegment, "operations", op_schema),
                "transition_in": optional_model_field_schema(TransitionSpec, PlanSegment, "transition_in"),
            },
            "required": ["scene_id"],
            "additionalProperties": False,
        }
        segments = field_schema(cls, "segments")
        segments["items"] = segment_schema
        schema: dict[str, Any] = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "description": cls.__doc__,
            "properties": {
                "segments": segments,
                "post_operations": array_field_schema(cls, "post_operations", op_schema),
            },
            "required": ["segments"],
            "additionalProperties": False,
        }
        if not strict:
            return schema
        op_defs = op_schema.pop("$defs", None)
        if op_defs:
            schema["$defs"] = op_defs
        return _to_strict_schema(schema)

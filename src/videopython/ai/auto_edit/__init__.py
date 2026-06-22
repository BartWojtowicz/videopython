"""LLM-authored editing: build a scene catalog and plan a VideoEdit from it."""

from __future__ import annotations

import videopython.ai.ops  # noqa: F401  -- registers the AI ops (face_crop, object_detection_overlay)

from .backend import PlannerError, StructuredVisionLLM
from .catalog import build_catalog
from .editor import AutoEditError, AutoEditor
from .local import OllamaVisionLLM
from .models import CatalogBundle, CatalogScene, EditCatalog, EditPlan, PlanSegment
from .resolve import UnknownSceneIdsError, resolve_plan

__all__ = [
    "AutoEditError",
    "AutoEditor",
    "CatalogBundle",
    "CatalogScene",
    "EditCatalog",
    "EditPlan",
    "OllamaVisionLLM",
    "PlanSegment",
    "PlannerError",
    "StructuredVisionLLM",
    "UnknownSceneIdsError",
    "build_catalog",
    "resolve_plan",
]

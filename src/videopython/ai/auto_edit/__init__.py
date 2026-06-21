"""LLM-authored editing: build a scene catalog and plan a VideoEdit from it."""

from __future__ import annotations

from .backend import ImagePart, Part, PlannerError, StructuredVisionLLM, TextPart
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
    "ImagePart",
    "OllamaVisionLLM",
    "Part",
    "PlanSegment",
    "PlannerError",
    "StructuredVisionLLM",
    "TextPart",
    "UnknownSceneIdsError",
    "build_catalog",
    "resolve_plan",
]

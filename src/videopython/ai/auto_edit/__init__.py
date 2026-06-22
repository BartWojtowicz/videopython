"""LLM-authored editing: build a scene catalog and plan a VideoEdit from it.

Uses eager imports (unlike the leaf packages): it is a thin orchestration
package whose submodules load together, and it MUST eagerly import
``videopython.ai.ops`` so the AI editing ops are registered before any
``EditPlan.json_schema()`` call. Heavy ML deps it pulls in stay lazy.
"""

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

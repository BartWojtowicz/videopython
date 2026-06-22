"""Force-registration of the AI-defined editing operations.

``FaceTrackingCrop`` (``ai/transforms.py``) and ``ObjectDetectionOverlay``
(``ai/effects.py``) are :class:`~videopython.editing.operation.Operation`
subclasses, so they enter ``Operation._registry`` as a side-effect of their
class definition (via ``__pydantic_init_subclass__``). But ``ai/__init__``
re-exports them *lazily* (PEP 562), so in a fresh process that builds the op
schema before anything touches those symbols, they are silently absent from
``Operation.llm_registry()`` / ``EditPlan.json_schema()`` / the MCP
``edit_plan_schema`` resource -- the planner can never select them and
``Operation.get("face_crop")`` raises "Unknown op_id".

Importing this module eagerly imports both leaf modules, guaranteeing the AI
ops are registered. Import it wherever the AI op universe is assembled (it is
imported from ``ai.auto_edit``, which the MCP server depends on transitively).
The imports are light: ``transforms``/``effects`` pull only ``understanding``
leaves whose heavy ML deps load lazily, so this drags in no torch at import.
"""

from __future__ import annotations

import videopython.ai.effects  # noqa: F401  -- registers ObjectDetectionOverlay
import videopython.ai.transforms  # noqa: F401  -- registers FaceTrackingCrop

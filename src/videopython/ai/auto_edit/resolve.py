"""Resolve a by-id EditPlan into a runnable VideoEdit."""

from __future__ import annotations

from videopython.ai.errors import AiError
from videopython.editing import SegmentConfig, VideoEdit

from .models import EditCatalog, EditPlan


class UnknownSceneIdsError(AiError, ValueError):
    """An EditPlan referenced scene ids absent from the catalog."""

    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        super().__init__(f"Plan references unknown scene ids: {sorted(set(ids))}")


def resolve_plan(plan: EditPlan, catalog: EditCatalog) -> VideoEdit:
    """Map each plan segment's scene_id to its exact source/start/end."""
    by_id = catalog.by_id()
    unknown = [seg.scene_id for seg in plan.segments if seg.scene_id not in by_id]
    if unknown:
        raise UnknownSceneIdsError(unknown)
    segments = [
        SegmentConfig(
            source=by_id[seg.scene_id].source,
            start=by_id[seg.scene_id].start,
            end=by_id[seg.scene_id].end,
            operations=seg.operations,
            transition_in=seg.transition_in,
        )
        for seg in plan.segments
    ]
    return VideoEdit(segments=segments, post_operations=plan.post_operations)

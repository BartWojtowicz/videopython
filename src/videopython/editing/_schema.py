"""Shared helpers for building LLM-facing JSON schemas (used by VideoEdit and EditPlan)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


def field_schema(model: type[BaseModel], field_name: str) -> dict[str, Any]:
    """The property block for ``model.field_name`` (description/format/bounds), no ``title``."""
    prop = dict(model.model_json_schema()["properties"][field_name])
    prop.pop("title", None)
    return prop


def array_field_schema(model: type[BaseModel], field_name: str, items: dict[str, Any]) -> dict[str, Any]:
    """An array slot: the model's shape/description with ``items`` inlined and ``default: []``."""
    prop = field_schema(model, field_name)
    prop["items"] = items
    prop["default"] = []
    return prop


def optional_model_field_schema(
    inline_model: type[BaseModel], parent: type[BaseModel], field_name: str
) -> dict[str, Any]:
    """An optional ``Model | None`` slot with the model inlined self-contained.

    Pydantic emits such a field as an ``anyOf`` with a ``$ref`` into a buried
    ``$defs``; inline a closed copy (titles dropped, descriptions kept) so the
    field needs no external ``$defs`` -- the same self-containment the op union
    relies on.
    """
    inline = inline_model.model_json_schema()
    inline.pop("title", None)
    inline.pop("$defs", None)
    for sub in inline.get("properties", {}).values():
        sub.pop("title", None)
    prop = field_schema(parent, field_name)
    prop["anyOf"] = [inline, {"type": "null"}]
    return prop

"""Helpers for guarding optional heavy-dependency imports.

Every heavy ML dependency in ``videopython.ai`` is imported lazily inside the
method that needs it (there are no top-level ``import torch``/``transformers``/
``chatterbox`` statements anywhere under ``ai/``). That keeps leaf modules
importable on a slim install: importing ``ai.understanding.objects`` only needs
core deps until a detector is actually constructed.

The granular extras (``asr``, ``vision``, ``tts``, ...) partition those heavy
deps. When a user installs only one extra and reaches a code path needing a dep
from another, the bare ``import`` would raise a stock ``ModuleNotFoundError``
with no hint about which extra restores it. :func:`require` wraps the import and
turns that into an actionable ``ImportError`` pointing at the right
``pip install 'videopython[<extra>]'``.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str, *, feature: str | None = None) -> ModuleType:
    """Import ``module`` or raise a clear, extra-pointing ``ImportError``.

    Args:
        module: Dotted module name to import (e.g. ``"chatterbox.mtl_tts"``,
            ``"torch"``). The fully-imported module object is returned so the
            call site can bind the symbols it needs.
        extra: The videopython extra that ships ``module`` (e.g. ``"tts"``,
            ``"asr"``). Surfaced verbatim in the install hint.
        feature: Optional human-readable feature name for the message head. When
            omitted the top-level package of ``module`` is used.

    Returns:
        The imported module object.

    Raises:
        ImportError: If ``module`` cannot be imported. The message always
            contains the extra name and a ``pip install 'videopython[<extra>]'``
            hint so the caller knows exactly how to fix it.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        label = feature or module.split(".")[0]
        raise ImportError(f"{label} requires the '{extra}' extra: pip install 'videopython[{extra}]'") from exc

"""Common base for ``videopython.ai`` error types.

Every error the AI layer raises subclasses :class:`AiError` (alongside the
builtin it semantically is — ``RuntimeError`` for operational failures,
``ValueError`` for bad inputs), so a caller can ``except AiError`` to catch any
of them without enumerating each submodule's error type.
"""

from __future__ import annotations


class AiError(Exception):
    """Base class for all ``videopython.ai`` errors."""

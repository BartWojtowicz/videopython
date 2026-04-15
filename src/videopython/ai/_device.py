"""Shared device selection helpers for local AI models."""

from __future__ import annotations

import logging
from typing import Literal, cast

Device = Literal["cpu", "cuda", "mps"]
logger = logging.getLogger(__name__)


def log_device_initialization(
    component: str,
    *,
    requested_device: str | None,
    resolved_device: str,
) -> None:
    """Log resolved device information for model initialization."""
    requested = requested_device.lower() if isinstance(requested_device, str) else "auto"
    logger.info(
        "%s initialized on device=%s (requested=%s)",
        component,
        resolved_device,
        requested,
    )


def release_device_memory(device: str | None) -> None:
    """Release cached allocator memory for the given device.

    Safe to call when torch is not importable or the device is CPU/None.
    """
    try:
        import torch
    except ImportError:
        return

    import gc

    gc.collect()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return

    if device == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            mps_mod = getattr(torch, "mps", None)
            empty_cache = getattr(mps_mod, "empty_cache", None) if mps_mod is not None else None
            if callable(empty_cache):
                empty_cache()


def select_device(
    device: str | None,
    *,
    mps_allowed: bool,
) -> Device:
    """Select an execution device for local inference.

    Selection order for auto mode (`device=None` or `device="auto"`):
    1. CUDA
    2. MPS (only when ``mps_allowed`` is True)
    3. CPU
    """
    requested = device.lower() if isinstance(device, str) else None

    if requested is not None and requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")

    if requested == "cpu":
        return cast(Device, requested)

    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but no CUDA device is available.")
        return "cuda"

    if requested == "mps":
        if not mps_allowed:
            raise ValueError("MPS is not supported for this model.")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("MPS requested but MPS backend is not available.")
        return "mps"

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_allowed and mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"

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

    if requested in {"cpu", "cuda"}:
        return cast(Device, requested)

    if requested == "mps":
        if not mps_allowed:
            raise ValueError("MPS is not supported for this model.")
        return "mps"

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if mps_allowed and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

"""Keyframe encoding helpers for LLM / MCP image payloads.

Public so transport layers (the MCP server) and the shared Ollama client both
reach for the same downscale + PNG-encode helpers instead of importing privates
across package boundaries.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

# Bound a keyframe's longest side before PNG-encoding for an MCP/LLM payload.
KEYFRAME_MAX_DIM = 768


def downscale_keyframe(frame: np.ndarray, max_dim: int = KEYFRAME_MAX_DIM) -> np.ndarray:
    """Shrink an RGB frame so its longest side is at most ``max_dim`` (aspect preserved; never upscales)."""
    h, w = frame.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def encode_png_b64(frame: np.ndarray) -> str:
    """PNG-encode an RGB frame and return base64 ascii. Full resolution (no downscale)."""
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def keyframe_to_png_b64(frame: np.ndarray, max_dim: int = KEYFRAME_MAX_DIM) -> str:
    """Downscale a keyframe then PNG-encode it as base64 -- the MCP transport payload.

    SceneVLM captioning and the local planner deliberately encode full-resolution
    frames via :func:`encode_png_b64` instead.
    """
    return encode_png_b64(downscale_keyframe(frame, max_dim=max_dim))

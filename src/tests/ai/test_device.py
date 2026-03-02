"""Tests for shared device selection helpers."""

from __future__ import annotations

import sys
import types

import pytest

from videopython.ai._device import select_device


def _fake_torch(*, cuda_available: bool, mps_available: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available),
        ),
    )


def test_select_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True, mps_available=True))
    assert select_device(None, mps_allowed=True) == "cuda"


def test_select_device_auto_uses_mps_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False, mps_available=True))
    assert select_device(None, mps_allowed=True) == "mps"


def test_select_device_explicit_cuda_requires_available_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False, mps_available=True))
    with pytest.raises(ValueError, match="CUDA requested"):
        select_device("cuda", mps_allowed=True)


def test_select_device_explicit_mps_requires_available_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False, mps_available=False))
    with pytest.raises(ValueError, match="MPS requested"):
        select_device("mps", mps_allowed=True)

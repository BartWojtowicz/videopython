from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class _Tensor(np.ndarray):
    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)

    def to(self, *_args, **_kwargs):
        return self

    def unsqueeze(self, axis: int):
        return np.expand_dims(self, axis).view(_Tensor)


class _Generator:
    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int):
        self.seed = seed
        return self


def _tensor(value, dtype=None, **_kwargs) -> _Tensor:
    return np.asarray(value, dtype=dtype).view(_Tensor)


def _fake_torch() -> ModuleType:
    torch = ModuleType("torch")
    attributes = {
        "bfloat16": "bfloat16",
        "float32": np.float32,
        "Tensor": _Tensor,
        "Generator": _Generator,
        "no_grad": nullcontext,
        "inference_mode": nullcontext,
        "tensor": _tensor,
        "from_numpy": _tensor,
        "zeros": lambda *shape, dtype=None: _tensor(np.zeros(shape, dtype=dtype)),
        "vstack": lambda tensors: _tensor(np.vstack(tensors)),
        "device": lambda value: value,
        "use_deterministic_algorithms": lambda _enabled: None,
        "cuda": SimpleNamespace(is_available=lambda: False),
        "backends": SimpleNamespace(
            cudnn=SimpleNamespace(is_available=lambda: False),
            mps=SimpleNamespace(is_available=lambda: False),
        ),
    }
    for name, value in attributes.items():
        setattr(torch, name, value)
    return torch


@pytest.fixture(scope="module", autouse=True)
def model_runtime_fakes() -> Iterator[None]:
    # These tests exercise our boundary code, not the optional model runtimes.
    monkeypatch = pytest.MonkeyPatch()
    if importlib.util.find_spec("torch") is None:
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    yield
    monkeypatch.undo()

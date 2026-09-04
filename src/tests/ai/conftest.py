from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class _Tensor(np.ndarray):
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
        "tensor": _tensor,
        "from_numpy": _tensor,
        "zeros": lambda shape, dtype=None: _tensor(np.zeros(shape, dtype=dtype)),
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


class _Tokenizer:
    def encode(self, text: str) -> list[str]:
        return re.findall(r"\w+|[^\w\s]", text)


def _fake_whisper() -> tuple[ModuleType, ModuleType, ModuleType]:
    whisper = ModuleType("whisper")
    setattr(whisper, "__path__", [])

    audio = ModuleType("whisper.audio")
    sample_rate = 16_000
    sample_count = 30 * sample_rate
    setattr(audio, "SAMPLE_RATE", sample_rate)
    setattr(audio, "N_SAMPLES", sample_count)
    setattr(audio, "pad_or_trim", lambda value, length=sample_count: _tensor(np.resize(value, length)))
    setattr(
        audio,
        "log_mel_spectrogram",
        lambda _value, n_mels=80: _tensor(np.zeros((n_mels, 3000), dtype=np.float32)),
    )

    tokenizer = ModuleType("whisper.tokenizer")
    setattr(tokenizer, "get_tokenizer", lambda **_kwargs: _Tokenizer())

    setattr(whisper, "audio", audio)
    setattr(whisper, "tokenizer", tokenizer)
    return whisper, audio, tokenizer


@pytest.fixture(scope="module", autouse=True)
def model_runtime_fakes() -> Iterator[None]:
    # These tests exercise our boundary code, not the optional model runtimes.
    monkeypatch = pytest.MonkeyPatch()
    if importlib.util.find_spec("torch") is None:
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    if importlib.util.find_spec("whisper") is None:
        whisper, audio, tokenizer = _fake_whisper()
        monkeypatch.setitem(sys.modules, "whisper", whisper)
        monkeypatch.setitem(sys.modules, "whisper.audio", audio)
        monkeypatch.setitem(sys.modules, "whisper.tokenizer", tokenizer)
    yield
    monkeypatch.undo()

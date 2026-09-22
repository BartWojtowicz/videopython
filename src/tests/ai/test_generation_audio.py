from __future__ import annotations

import os
from types import SimpleNamespace

from videopython.ai.generation.audio import TextToMusic


def test_musicgen_initialization_does_not_change_mps_environment(monkeypatch) -> None:
    import videopython.ai._optional as optional
    import videopython.ai.generation.audio as generation_audio

    class Loader:
        value = object()

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls.value

    class Model:
        def to(self, _device):
            return self

    class ModelLoader:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return Model()

    transformers = SimpleNamespace(AutoProcessor=Loader, MusicgenForConditionalGeneration=ModelLoader)
    monkeypatch.setattr(optional, "require", lambda *_args, **_kwargs: transformers)
    monkeypatch.setattr(generation_audio, "select_device", lambda *_args, **_kwargs: "mps")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "caller-value")

    TextToMusic()._init_local()

    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "caller-value"

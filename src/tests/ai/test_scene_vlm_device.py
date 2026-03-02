"""Device-initialization tests for SceneVLM."""

from __future__ import annotations

import sys
import types

import pytest

import videopython.ai.understanding.image as image_mod


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_on_device: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    to_calls: list[str] = []
    kwargs_seen: dict[str, str] = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, _name):
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _name, **kwargs):
            if "dtype" in kwargs:
                kwargs_seen["dtype"] = str(kwargs["dtype"])
            return cls()

        def to(self, device):
            to_calls.append(device)
            if fail_on_device is not None and device == fail_on_device:
                raise RuntimeError(f"device failure: {device}")
            return self

        def eval(self):
            return self

    fake_transformers = types.SimpleNamespace(
        AutoProcessor=FakeProcessor,
        AutoModelForImageTextToText=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return to_calls, kwargs_seen


def test_scene_vlm_uses_dtype_and_non_mps_auto_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"mps_allowed": True}

    def fake_select_device(_requested, mps_allowed=False):
        called["mps_allowed"] = mps_allowed
        return "cpu"

    monkeypatch.setattr(image_mod, "select_device", fake_select_device)
    to_calls, kwargs_seen = _install_fake_transformers(monkeypatch)

    scene_vlm = image_mod.SceneVLM(model_name="fake-model", device=None)
    scene_vlm._init_local()

    assert called["mps_allowed"] is False
    assert kwargs_seen["dtype"] == "auto"
    assert to_calls == ["cpu"]
    assert scene_vlm.device == "cpu"


def test_scene_vlm_init_error_is_not_silently_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_mod, "select_device", lambda _requested, mps_allowed=False: "cuda")
    _install_fake_transformers(monkeypatch, fail_on_device="cuda")

    scene_vlm = image_mod.SceneVLM(model_name="fake-model", device=None)
    with pytest.raises(RuntimeError, match="device failure: cuda"):
        scene_vlm._init_local()

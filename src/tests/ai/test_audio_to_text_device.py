"""Device-initialization tests for AudioToText."""

from __future__ import annotations

import pytest

import videopython.ai.understanding.audio as audio_mod


def test_audio_to_text_disables_mps_auto_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, bool] = {"mps_allowed": True}

    def fake_select_device(_requested, mps_allowed=False):
        called["mps_allowed"] = mps_allowed
        return "cpu"

    monkeypatch.setattr(audio_mod, "select_device", fake_select_device)

    transcriber = audio_mod.AudioToText(model_name="small", device=None)

    assert called["mps_allowed"] is False
    assert transcriber.device == "cpu"

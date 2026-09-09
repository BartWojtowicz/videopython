import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from videopython.ai.generation._speech_graphs import SpeechGraphs
from videopython.ai.generation.audio import TextToSpeech


class LlamaMLP:
    training = False

    def __init__(self):
        self.calls = 0
        self.child = SimpleNamespace(_forward_hooks={}, _forward_pre_hooks={})

    def modules(self):
        return [self, self.child]

    def forward(self, x):
        self.calls += 1
        return x.data * 2


class Input:
    def __init__(self, value=1, shape=(2, 1, 4), device="cuda", dtype="float32", contiguous=True):
        self.data = np.full(shape, value, dtype=np.float32)
        self.shape = shape
        self.ndim = len(shape)
        self.device = SimpleNamespace(type=device)
        self.dtype = dtype
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous

    def stride(self):
        return tuple(s // self.data.itemsize for s in self.data.strides)


@pytest.fixture
def runtime(monkeypatch):
    torch = SimpleNamespace(
        float32="float32",
        is_grad_enabled=lambda: False,
        inference_mode=nullcontext,
        cuda=SimpleNamespace(empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    module = LlamaMLP()
    owner = SpeechGraphs(
        SimpleNamespace(t3=SimpleNamespace(tfmr=SimpleNamespace(layers=[SimpleNamespace(mlp=module)])))
    )
    wrapper = owner._wrappers[0]
    captures = []

    def capture(x):
        captures.append(x.shape)
        static = np.empty_like(x.data)
        output = np.empty_like(x.data)
        buffer = SimpleNamespace(copy_=lambda value: np.copyto(static, value.data))
        graph = SimpleNamespace(replay=lambda: np.multiply(static, 2, out=output))
        return graph, buffer, output

    monkeypatch.setattr(wrapper, "_capture", capture)
    return module, owner, captures, torch


def test_replay_copies_new_token_values(runtime):
    module, _, captures, _ = runtime
    x = Input(1)
    assert np.all(module.forward(x) == 2)
    x.data.fill(7)
    assert np.all(module.forward(x) == 14)
    assert len(captures) == 1
    assert module.calls == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"shape": (2, 8, 4)}, {"shape": (1, 1, 4)}, {"device": "cpu"}, {"dtype": "float16"}, {"contiguous": False}],
)
def test_other_layouts_use_original_forward(runtime, kwargs):
    module, _, captures, _ = runtime
    assert np.all(module.forward(Input(**kwargs)) == 2)
    assert not captures
    assert module.calls == 1


def test_training_and_gradients_bypass_capture(runtime):
    module, _, captures, torch = runtime
    module.training = True
    module.forward(Input())
    module.training = False
    torch.is_grad_enabled = lambda: True
    module.forward(Input())
    assert module.calls == 2
    assert not captures


def test_nested_hooks_added_after_capture_still_run_eagerly(runtime):
    module, _, captures, _ = runtime
    module.forward(Input())
    module.child._forward_hooks[1] = object()
    module.forward(Input())
    assert module.calls == 1
    assert len(captures) == 1


def test_changed_shape_does_not_replay_wrong_graph(runtime):
    module, _, captures, _ = runtime
    module.forward(Input())
    assert module.forward(Input(shape=(2, 1, 8))).shape == (2, 1, 8)
    assert module.calls == 1
    assert len(captures) == 1


def test_capture_failure_restores_forwards_and_retries(runtime, monkeypatch):
    module, owner, _, _ = runtime

    def fail(x):
        raise RuntimeError("capture unsupported")

    monkeypatch.setattr(owner._wrappers[0], "_capture", fail)
    assert np.all(module.forward(Input(3)) == 6)
    assert owner.disabled
    assert not owner._wrappers
    assert "forward" not in vars(module)
    owner.close()


def test_close_releases_buffers_and_restores_instance_override(runtime):
    module, owner, _, _ = runtime
    wrapper = owner._wrappers[0]
    module.forward(Input())
    assert wrapper.state is not None
    owner.close()
    assert wrapper.state is None
    assert not owner._wrappers
    assert "forward" not in vars(module)
    previous = module.forward
    module.forward = previous
    owner = SpeechGraphs(
        SimpleNamespace(t3=SimpleNamespace(tfmr=SimpleNamespace(layers=[SimpleNamespace(mlp=module)])))
    )
    owner.close()
    assert module.forward is previous


def test_tts_unload_closes_graphs_before_releasing_model(monkeypatch):
    import videopython.ai._predictor as predictor

    tts = TextToSpeech()
    closed = []
    tts._speech_graphs = SimpleNamespace(close=lambda: closed.append(True))

    def release(device):
        assert closed == [True]
        assert tts._speech_graphs is None
        assert tts._model is None

    monkeypatch.setattr(predictor, "release_device_memory", release)
    tts.unload()

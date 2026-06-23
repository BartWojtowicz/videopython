"""Tests for TextToImage (Qwen-Image), with the diffusers pipeline mocked."""

from unittest.mock import MagicMock, patch

from PIL import Image

from videopython.ai.generation.image import TextToImage

_QWEN_SHA = "25468b98e3276ca6700de15c6628e51b7de54a26"


def _fake_diffusers(pipeline):
    mod = MagicMock()
    mod.QwenImagePipeline.from_pretrained.return_value = pipeline
    return mod


class TestInit:
    @patch("videopython.ai.generation.image.select_device", return_value="cuda")
    @patch("videopython.ai._optional.require")
    def test_cuda_offloads_and_does_not_call_to(self, mock_require, _sd):
        import torch

        pipe = MagicMock()
        fake = _fake_diffusers(pipe)
        mock_require.return_value = fake

        TextToImage(device="cuda")._init_local()

        pipe.enable_model_cpu_offload.assert_called_once()
        pipe.enable_vae_tiling.assert_called_once()
        pipe.to.assert_not_called()

        _, kwargs = fake.QwenImagePipeline.from_pretrained.call_args
        assert kwargs["revision"] == _QWEN_SHA
        assert kwargs["use_safetensors"] is True
        assert kwargs["torch_dtype"] == torch.bfloat16
        assert "variant" not in kwargs  # Qwen-Image has no fp16 variant

    @patch("videopython.ai.generation.image.select_device", return_value="cpu")
    @patch("videopython.ai._optional.require")
    def test_cpu_calls_to_and_does_not_offload(self, mock_require, _sd):
        import torch

        pipe = MagicMock()
        mock_require.return_value = _fake_diffusers(pipe)

        TextToImage(device="cpu")._init_local()

        pipe.to.assert_called_once_with("cpu")
        pipe.enable_model_cpu_offload.assert_not_called()
        _, kwargs = mock_require.return_value.QwenImagePipeline.from_pretrained.call_args
        assert kwargs["torch_dtype"] == torch.float32


class TestGenerate:
    def test_passes_qwen_params_and_appends_magic(self):
        img = Image.new("RGB", (8, 8), (100, 110, 120))
        pipe = MagicMock()
        pipe.return_value.images = [img]

        t = TextToImage()
        t.device = "cpu"
        t._pipeline = pipe

        out = t.generate_image("a cat on a sofa")

        assert out is img
        _, kwargs = pipe.call_args
        assert kwargs["true_cfg_scale"] == 4.0
        assert kwargs["negative_prompt"] == " "
        assert kwargs["width"] == 1328
        assert kwargs["height"] == 1328
        assert kwargs["prompt"].startswith("a cat on a sofa")
        assert "Ultra HD" in kwargs["prompt"]  # magic suffix appended

    def test_add_magic_false_keeps_prompt_verbatim(self):
        pipe = MagicMock()
        pipe.return_value.images = [Image.new("RGB", (4, 4))]

        t = TextToImage()
        t.device = "cpu"
        t._pipeline = pipe

        t.generate_image("plain prompt", add_magic=False)
        assert pipe.call_args.kwargs["prompt"] == "plain prompt"

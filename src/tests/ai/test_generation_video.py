"""Tests for TextToVideo / ImageToVideo (Wan2.2), with the diffusers pipeline mocked."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from videopython.ai.generation.video import ImageToVideo, TextToVideo

_T2V_SHA = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"
_I2V_SHA = "596658fd9ca6b7b71d5057529bbf319ecbc61d74"


def _pil_frames(n, size=(16, 16), color=(120, 130, 140)):
    return [Image.new("RGB", size, color) for _ in range(n)]


class TestTextToVideo:
    @patch("videopython.ai.generation.video.Video")
    def test_frames_convert_to_uint8_rgb(self, mock_video):
        # .frames[0] is a list of PIL images (output_type="pil"); guards against the
        # output_type="np" float[0,1] path that uint8-casts to an all-black video.
        pipe = MagicMock()
        pipe.return_value.frames = [_pil_frames(3)]

        t = TextToVideo()
        t.device = "cpu"
        t._pipeline = pipe

        t.generate_video("a dog running", num_frames=3)

        arr = mock_video.from_frames.call_args.args[0]
        assert arr.dtype == np.uint8
        assert arr.ndim == 4 and arr.shape[-1] == 3
        assert arr.max() > 0  # not all-black
        assert mock_video.from_frames.call_args.kwargs["fps"] == 16.0

        kw = pipe.call_args.kwargs
        assert kw["output_type"] == "pil"
        assert kw["guidance_scale_2"] == 3.0
        assert kw["height"] == 720 and kw["width"] == 1280
        assert kw["negative_prompt"]  # non-empty canonical Wan negative prompt

    @patch("videopython.ai.generation.video.select_device", return_value="cuda")
    @patch("videopython.ai._optional.require")
    def test_cuda_offloads_with_fp32_vae(self, mock_require, _sd):
        import torch

        diff = MagicMock()
        pipe = MagicMock()
        diff.WanPipeline.from_pretrained.return_value = pipe
        mock_require.return_value = diff

        TextToVideo(device="cuda")._init_local()

        pipe.enable_model_cpu_offload.assert_called_once()
        pipe.to.assert_not_called()
        pipe.enable_vae_tiling.assert_not_called()  # Wan has no VAE tiling (Qwen-only)

        _, vae_kw = diff.AutoencoderKLWan.from_pretrained.call_args
        assert vae_kw["subfolder"] == "vae"
        assert vae_kw["torch_dtype"] == torch.float32
        assert vae_kw["revision"] == _T2V_SHA

        _, pipe_kw = diff.WanPipeline.from_pretrained.call_args
        assert pipe_kw["revision"] == _T2V_SHA
        assert pipe_kw["torch_dtype"] == torch.bfloat16

    @patch("videopython.ai.generation.video.select_device", return_value="cpu")
    def test_non_cuda_raises(self, _sd):
        # Wan2.2 is CUDA-only; a non-CUDA device fails loudly (no CPU/MPS fallback).
        with pytest.raises(RuntimeError, match="CUDA"):
            TextToVideo(device="cpu")._init_local()


class TestImageToVideo:
    @patch("videopython.ai.generation.video.Video")
    def test_resizes_to_grid_and_converts_frames(self, mock_video):
        pipe = MagicMock()
        pipe.vae_scale_factor_spatial = 8
        pipe.transformer.config.patch_size = [1, 2, 2]  # real config is a 3-tuple; code reads [1] -> mod_value 16
        pipe.return_value.frames = [_pil_frames(2, color=(50, 60, 70))]

        t = ImageToVideo()
        t.device = "cpu"
        t._pipeline = pipe

        t.generate_video(Image.new("RGB", (832, 480)), prompt="pan left")

        kw = pipe.call_args.kwargs
        assert kw["height"] % 16 == 0 and kw["width"] % 16 == 0
        assert kw["image"].size == (kw["width"], kw["height"])  # image resized to requested dims
        assert kw["output_type"] == "pil"

        arr = mock_video.from_frames.call_args.args[0]
        assert arr.dtype == np.uint8 and arr.max() > 0

    @patch("videopython.ai.generation.video.select_device", return_value="cuda")
    @patch("videopython.ai._optional.require")
    def test_cuda_offloads_with_fp32_vae(self, mock_require, _sd):
        import torch

        diff = MagicMock()
        pipe = MagicMock()
        diff.WanImageToVideoPipeline.from_pretrained.return_value = pipe
        mock_require.return_value = diff

        ImageToVideo(device="cuda")._init_local()

        pipe.enable_model_cpu_offload.assert_called_once()
        pipe.to.assert_not_called()
        pipe.enable_vae_tiling.assert_not_called()  # Wan has no VAE tiling (Qwen-only)

        _, vae_kw = diff.AutoencoderKLWan.from_pretrained.call_args
        assert vae_kw["torch_dtype"] == torch.float32
        assert vae_kw["revision"] == _I2V_SHA
        _, pipe_kw = diff.WanImageToVideoPipeline.from_pretrained.call_args
        assert pipe_kw["revision"] == _I2V_SHA

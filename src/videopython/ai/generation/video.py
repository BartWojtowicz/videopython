import numpy as np
import torch
from diffusers import DiffusionPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from modelscope import AutoencoderKLWan, WanPipeline  # type: ignore
from PIL.Image import Image

from videopython.base.video import Video

TEXT_TO_VIDEO_MODEL = "cerspense/zeroscope_v2_576w"
IMAGE_TO_VIDEO_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
WAN_TEXT_TO_VIDEO_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"  # Local path after download
WAN_IMAGE_TO_VIDEO_MODEL = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"  # Local path after download


class TextToVideo:
    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but TextToVideo model requires CUDA.")

        self.dtype = torch.bfloat16
        self.device = "cuda"

        vae = AutoencoderKLWan.from_pretrained(
            WAN_TEXT_TO_VIDEO_MODEL,
            subfolder="vae",
            torch_dtype=torch.float32
        )

        self.flow_shift = 12.0
        self.pipeline = WanPipeline.from_pretrained(
            WAN_TEXT_TO_VIDEO_MODEL,
            vae=vae,
            torch_dtype=self.dtype
        )
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.pipeline.scheduler.config, flow_shift=self.flow_shift
        )
        self.pipeline.to(self.device)

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        num_steps: int = 10,
        height: int = 480,
        width: int = 832,
        num_frames: int = 49,
        guidance_scale: float = 4.0,
    ) -> Video:
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        ).frames[0]

        video_frames = np.asarray(255 * output, dtype=np.uint8)
        return Video.from_frames(video_frames, fps=30.0)


class ImageToVideo:
    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but ImageToVideo model requires CUDA.")

        self.dtype = torch.bfloat16
        self.device = "cuda"

        vae = AutoencoderKLWan.from_pretrained(
            WAN_IMAGE_TO_VIDEO_MODEL,
            subfolder="vae",
            torch_dtype=torch.float32
        )

        self.pipeline = WanPipeline.from_pretrained(
            WAN_IMAGE_TO_VIDEO_MODEL,
            vae=vae,
            torch_dtype=self.dtype
        )
        self.pipeline.to(self.device)

    def generate_video(
        self,
        image: Image,
        prompt: str,
        negative_prompt: str | None = None,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        guidance_scale: float = 4.0,
        guidance_scale_2: float = 3.0,
        num_inference_steps: int = 40,
        fps: float = 16.0
    ) -> Video:
        output = self.pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            num_inference_steps=num_inference_steps,
        ).frames[0]

        video_frames = np.asarray(255 * output, dtype=np.uint8)
        return Video.from_frames(video_frames, fps=fps)

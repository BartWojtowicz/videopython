import numpy as np
import torch
from diffusers import AutoencoderKLWan, DiffusionPipeline, WanPipeline
from PIL.Image import Image

from videopython.base.video import Video

TEXT_TO_VIDEO_MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
IMAGE_TO_VIDEO_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"


class TextToVideo:
    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but TextToVideo model requires CUDA.")
        vae = AutoencoderKLWan.from_pretrained(TEXT_TO_VIDEO_MODEL, subfolder="vae", torch_dtype=torch.float32)
        self.pipeline = WanPipeline.from_pretrained(TEXT_TO_VIDEO_MODEL, vae=vae, torch_dtype=torch.bfloat16)
        self.pipeline.to("cuda")

    def generate_video(
        self, prompt: str, num_steps: int = 40, height: int = 720, width: int = 1280, num_frames: int = 81
    ) -> Video:
        video_frames = self.pipeline(
            prompt,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=4.0,
            guidance_scale_2=3.0,
            sample_shift=12.0,
            boundary_ratio=0.875,
        ).frames[0]
        video_frames = np.asarray(255 * video_frames, dtype=np.uint8)
        return Video.from_frames(video_frames, fps=16.0)


class ImageToVideo:
    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but ImageToVideo model requires CUDA.")
        self.pipeline = DiffusionPipeline.from_pretrained(
            IMAGE_TO_VIDEO_MODEL, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")

    def generate_video(self, image: Image, fps: int = 24) -> Video:
        video_frames = self.pipeline(image=image, fps=fps, output_type="np").frames[0]
        video_frames = np.asarray(255 * video_frames, dtype=np.uint8)
        return Video.from_frames(video_frames, fps=float(fps))

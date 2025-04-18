import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from PIL.Image import Image

from videopython.base.video import Video

TEXT_TO_VIDEO_MODEL = "cerspense/zeroscope_v2_576w"
IMAGE_TO_VIDEO_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"


class TextToVideo:
    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but TextToVideo model requires CUDA.")
        self.pipeline = DiffusionPipeline.from_pretrained(TEXT_TO_VIDEO_MODEL, torch_dtype=torch.float16)
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.to("cuda")

    def generate_video(
        self, prompt: str, num_steps: int = 25, height: int = 320, width: int = 576, num_frames: int = 24
    ) -> Video:
        video_frames = self.pipeline(
            prompt,
            num_inference_steps=num_steps,
            height=height,
            width=width,
            num_frames=num_frames,
        ).frames[0]
        video_frames = np.asarray(255 * video_frames, dtype=np.uint8)
        return Video.from_frames(video_frames, fps=24.0)


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

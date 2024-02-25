import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler, I2VGenXLPipeline
from PIL.Image import Image

from videopython.base.video import Video

TEXT_TO_VIDEO_MODEL = "cerspense/zeroscope_v2_576w"
IMAGE_TO_VIDEO_MODEL = "ali-vilab/i2vgen-xl"


class TextToVideo:
    def __init__(self, gpu_optimized: bool = True):
        self.pipeline = DiffusionPipeline.from_pretrained(
            TEXT_TO_VIDEO_MODEL, torch_dtype=torch.float16 if gpu_optimized else torch.float32
        )
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(self.pipeline.scheduler.config)
        if gpu_optimized:
            self.pipeline.enable_model_cpu_offload()

    def generate_video(
        self, prompt: str, num_steps: int = 25, height: int = 320, width: int = 576, num_frames: int = 24
    ) -> Video:
        video_frames = np.asarray(
            self.pipeline(
                prompt,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                num_frames=num_frames,
            ).frames
        )
        return Video.from_frames(video_frames, fps=24.0)


class ImageToVideo:
    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but ImageToVideo model requires CUDA.")
        self.pipeline = I2VGenXLPipeline.from_pretrained(
            IMAGE_TO_VIDEO_MODEL, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")

    def generate_video(self, prompt: str, image: Image) -> Video:
        video_frames = self.pipeline(prompt=prompt, image=image, output_type="np").frames[0]
        video_frames = np.asarray(video_frames, dtype=np.uint8)
        return Video.from_frames(video_frames, fps=16.0)

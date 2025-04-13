import torch
from diffusers import DiffusionPipeline
from PIL import Image

TEXT_TO_IMAGE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


class TextToImage:
    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but TextToVideo model requires CUDA.")
        self.pipeline = DiffusionPipeline.from_pretrained(
            TEXT_TO_IMAGE_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        self.pipeline.to("cuda")

    def generate_image(self, prompt: str) -> Image.Image:
        image = self.pipeline(prompt=prompt).images[0]
        return image

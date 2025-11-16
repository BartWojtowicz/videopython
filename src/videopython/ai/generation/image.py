import torch
from diffusers import FluxPipeline
from PIL import Image

TEXT_TO_IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"


class TextToImage:
    def __init__(self):
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but TextToImage model requires CUDA.")
        self.pipeline = FluxPipeline.from_pretrained(TEXT_TO_IMAGE_MODEL, torch_dtype=torch.bfloat16)
        self.pipeline.enable_model_cpu_offload()

    def generate_image(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
    ) -> Image.Image:
        image = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512,
        ).images[0]
        return image

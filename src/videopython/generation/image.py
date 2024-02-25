import io
import os

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image
from stability_sdk import client


class TextToImage:
    def __init__(
        self,
        stability_key: str | None = None,
        engine: str = "stable-diffusion-xl-1024-v1-0",
        verbose: bool = True,
    ):
        stability_key = stability_key or os.getenv("STABILITY_KEY")
        if stability_key is None:
            raise ValueError(
                "API Key for stability is required. Please provide it as an argument"
                " or set it as an environment variable `STABILITY_KEY`. "
            )

        self.client = client.StabilityInference(stability_key, verbose=verbose, engine=engine)

    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        cfg_scale: float = 8.0,
        seed: int = 1,
    ) -> Image.Image:
        answers = self.client.generate(
            prompt=prompt,
            seed=seed,
            steps=steps,  # Amount of inference steps performed on image generation.
            cfg_scale=cfg_scale,  # Influences how strongly your generation is guided to match your prompt.
            # Setting this value higher increases the strength in which it tries to match your prompt.
            # Defaults to 7.0 if not specified.
            width=width,
            height=height,
            safety=False,
            samples=1,
            sampler=generation.SAMPLER_K_DPMPP_2M,  # Choose which sampler we want to denoise our generation with.
            # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
            # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
        )
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.finish_reason == generation.FILTER:
                    raise RuntimeError(
                        "Your request activated the API's safety filters and could not be processed."
                        "Please modify the prompt and try again."
                    )
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                else:
                    raise ValueError(f"Unknown artifact type: {artifact.type}")
        return img

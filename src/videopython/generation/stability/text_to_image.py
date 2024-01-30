import io
import os

import numpy as np
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image
from stability_sdk import client

API_KEY = os.getenv("STABILITY_KEY")
if not API_KEY:
    raise KeyError(
        "Stability API key was not found in the environment! Please set in as `STABILITY_KEY` in your environment."
    )


def text_to_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    num_samples: int = 1,
    steps: int = 30,
    cfg_scale: float = 8.0,
    engine: str = "stable-diffusion-xl-1024-v1-0",
    verbose: bool = True,
    seed: int = 1,
) -> np.ndarray:
    """Generates image from prompt using the stability.ai API."""
    # Generate image
    stability_api = client.StabilityInference(
        key=API_KEY,
        verbose=verbose,
        engine=engine,  # Set the engine to use for generation.
        # Check out the following link for a list of available engines: https://platform.stability.ai/docs/features/api-parameters#engine
    )
    answers = stability_api.generate(
        prompt=prompt,
        seed=seed,
        steps=steps,  # Amount of inference steps performed on image generation.
        cfg_scale=cfg_scale,  # Influences how strongly your generation is guided to match your prompt.
        # Setting this value higher increases the strength in which it tries to match your prompt.
        # Defaults to 7.0 if not specified.
        width=width,
        height=height,
        samples=num_samples,
        sampler=generation.SAMPLER_K_DPMPP_2M  # Choose which sampler we want to denoise our generation with.
        # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
        # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )
    # Parse API response
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

    return np.array(img)

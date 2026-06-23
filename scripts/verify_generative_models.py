"""Manual GPU smoke test for the permissive generative-model swaps.

Exercises the real pipelines end-to-end (downloads weights + runs inference), so it
is NOT a CI test -- run it on a GPU machine before merging the generative swaps:

  - TextToImage  -> Qwen/Qwen-Image-2512        (~20B, Apache-2.0)
  - TextToVideo  -> Wan-AI/Wan2.2-T2V-A14B       (~28B MoE, Apache-2.0)
  - ImageToVideo -> Wan-AI/Wan2.2-I2V-A14B       (~28B MoE, Apache-2.0)

Each model is large; first run downloads many GB and needs a CUDA GPU (the classes
fall back to CPU/MPS but that is impractically slow for these sizes). Outputs are
written to --out-dir and each task is checked for non-black output (guards the
diffusers ``output_type`` frame-conversion bug).

Usage (with the [ai] extra installed):

    uv sync --extra ai
    uv run python scripts/verify_generative_models.py --task all
    uv run python scripts/verify_generative_models.py --task image --steps 50
    uv run python scripts/verify_generative_models.py --task i2v --image path/to.png

Exit code is non-zero if any selected task fails or produces an all-black result.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

_PROMPT = "A serene mountain lake at sunrise, mist over the water, photorealistic"


def _log(msg: str) -> None:
    print(f"[verify] {msg}", flush=True)


def _device_banner() -> None:
    import diffusers
    import torch

    cuda = torch.cuda.is_available()
    _log(f"torch={torch.__version__} diffusers={diffusers.__version__} cuda_available={cuda}")
    if cuda:
        _log(f"gpu={torch.cuda.get_device_name(0)}")
    else:
        _log("WARNING: no CUDA GPU detected -- these ~20-28B models will be impractically slow.")


def verify_image(out_dir: Path, steps: int, device: str | None) -> tuple[bool, str]:
    from videopython.ai.generation import TextToImage

    t0 = time.perf_counter()
    with TextToImage(device=device) as t2i:
        image = t2i.generate_image(_PROMPT, num_inference_steps=steps)
    elapsed = time.perf_counter() - t0

    path = out_dir / "qwen_image.png"
    image.save(path)
    non_black = int(np.asarray(image).max()) > 0
    msg = f"{image.size} -> {path} in {elapsed:.1f}s (non_black={non_black})"
    return non_black, msg


def verify_t2v(out_dir: Path, steps: int, frames: int, device: str | None) -> tuple[bool, str]:
    from videopython.ai.generation import TextToVideo

    t0 = time.perf_counter()
    with TextToVideo(device=device) as t2v:
        video = t2v.generate_video(_PROMPT, num_steps=steps, num_frames=frames)
    elapsed = time.perf_counter() - t0

    path = video.save(out_dir / "wan_t2v.mp4")
    non_black = int(video.frames.max()) > 0
    msg = f"frames={video.frames.shape} -> {path} in {elapsed:.1f}s (non_black={non_black})"
    return non_black, msg


def verify_i2v(out_dir: Path, steps: int, frames: int, device: str | None, image_path: str | None) -> tuple[bool, str]:
    from videopython.ai.generation import ImageToVideo

    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        # Simple synthetic landscape so i2v can run without a prior image-gen step.
        arr = np.zeros((480, 832, 3), dtype=np.uint8)
        arr[:240, :, 2] = 200  # sky (blue)
        arr[240:, :, 1] = 150  # ground (green)
        image = Image.fromarray(arr)

    t0 = time.perf_counter()
    with ImageToVideo(device=device) as i2v:
        video = i2v.generate_video(image, prompt="gentle camera pan", num_steps=steps, num_frames=frames)
    elapsed = time.perf_counter() - t0

    path = video.save(out_dir / "wan_i2v.mp4")
    non_black = int(video.frames.max()) > 0
    msg = f"frames={video.frames.shape} -> {path} in {elapsed:.1f}s (non_black={non_black})"
    return non_black, msg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["image", "t2v", "i2v", "all"], default="all")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps (lower = faster check).")
    parser.add_argument("--frames", type=int, default=49, help="Video frames (Wan expects 4n+1).")
    parser.add_argument("--device", default=None, help="Force device: cuda / cpu / mps. Default: auto.")
    parser.add_argument("--image", default=None, help="Input image for --task i2v (else a synthetic one).")
    parser.add_argument("--out-dir", default="generative_verify_out", help="Where to write outputs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _device_banner()

    tasks = ["image", "t2v", "i2v"] if args.task == "all" else [args.task]
    results: dict[str, tuple[bool, str]] = {}

    for task in tasks:
        _log(f"=== {task} ===")
        try:
            if task == "image":
                results[task] = verify_image(out_dir, args.steps, args.device)
            elif task == "t2v":
                results[task] = verify_t2v(out_dir, args.steps, args.frames, args.device)
            else:
                results[task] = verify_i2v(out_dir, args.steps, args.frames, args.device, args.image)
            _log(f"{task}: {results[task][1]}")
        except Exception:
            results[task] = (False, "EXCEPTION:\n" + traceback.format_exc())
            _log(f"{task} FAILED:\n{results[task][1]}")

    _log("==== SUMMARY ====")
    ok = True
    for task in tasks:
        passed, msg = results[task]
        ok = ok and passed
        _log(f"{'PASS' if passed else 'FAIL'}  {task}: {msg.splitlines()[0]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

from typing import Any

from PIL import Image

class PipelineOutput:
    images: list[Image.Image]

class FluxPipeline:
    @classmethod
    def from_pretrained(cls, model_id: str, torch_dtype: Any = None, **kwargs: Any) -> FluxPipeline: ...
    def enable_model_cpu_offload(self) -> None: ...
    def to(self, device: str) -> FluxPipeline: ...
    def __call__(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        max_sequence_length: int = 512,
        **kwargs: Any,
    ) -> PipelineOutput: ...

class DiffusionPipeline:
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> DiffusionPipeline: ...
    def to(self, device: str) -> DiffusionPipeline: ...
    def __call__(self, **kwargs: Any) -> PipelineOutput: ...

class WanPipeline:
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any) -> WanPipeline: ...
    def to(self, device: str) -> WanPipeline: ...
    def __call__(self, **kwargs: Any) -> Any: ...

class AutoencoderKLWan:
    @classmethod
    def from_pretrained(
        cls, model_id: str, subfolder: str | None = None, torch_dtype: Any = None, **kwargs: Any
    ) -> AutoencoderKLWan: ...

class DPMSolverMultistepScheduler:
    @classmethod
    def from_config(cls, config: Any) -> DPMSolverMultistepScheduler: ...

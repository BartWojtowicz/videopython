from typing import Any

import torch
from PIL import Image

class VitsModelOutput:
    waveform: torch.FloatTensor

class VitsConfig:
    sampling_rate: int

class VitsModel:
    config: VitsConfig
    @classmethod
    def from_pretrained(cls, model_type: str) -> VitsModel: ...
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> VitsModelOutput: ...

class BatchEncoding(dict[str, torch.Tensor]):
    def __getitem__(self, key: str) -> torch.Tensor: ...

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_type: str) -> AutoTokenizer: ...
    def __call__(self, text: str, return_tensors: str) -> BatchEncoding: ...

class PretrainedConfig:
    def __init__(self, **kwargs): ...

class EncodecConfig(PretrainedConfig):
    sampling_rate: int

class MusicgenDecoderConfig(PretrainedConfig): ...

class MusicgenConfig:
    @classmethod
    def from_sub_models_config(
        cls,
        text_encoder_config: PretrainedConfig,
        audio_encoder_config: PretrainedConfig,
        decoder_config: MusicgenDecoderConfig,
        **kwargs,
    ) -> MusicgenConfig: ...
    audio_encoder: EncodecConfig

class MusicgenForConditionalGeneration:
    @classmethod
    def from_pretrained(cls, model_type: str) -> MusicgenForConditionalGeneration: ...
    def generate(self, **inputs) -> torch.Tensor: ...
    config: MusicgenConfig

class AutoProcessor:
    @classmethod
    def from_pretrained(cls, model_type: str) -> AutoProcessor: ...
    def __call__(
        self,
        text: list | None = None,
        padding: bool | None = None,
        return_tensors: str | None = None,
        voice_preset: str | None = None,
    ) -> dict: ...

class GenerationConfig:
    sample_rate: int

class AutoModel:
    generation_config: GenerationConfig
    @classmethod
    def from_pretrained(cls, model_type: str) -> AutoModel: ...
    def to(self, device: str) -> AutoModel: ...
    def generate(self, **kwargs: Any) -> torch.Tensor: ...

class BlipProcessorOutput:
    def to(self, device: str) -> BlipProcessorOutput: ...
    def __getitem__(self, key: str) -> Any: ...
    def keys(self) -> list[str]: ...

class BlipProcessor:
    @classmethod
    def from_pretrained(cls, model_type: str) -> BlipProcessor: ...
    def __call__(
        self, images: Image.Image | None = None, text: str | None = None, return_tensors: str | None = None
    ) -> BlipProcessorOutput: ...
    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = False) -> str: ...

class BlipForConditionalGeneration:
    @classmethod
    def from_pretrained(cls, model_type: str) -> BlipForConditionalGeneration: ...
    def to(self, device: str) -> BlipForConditionalGeneration: ...
    def generate(self, **kwargs: Any) -> torch.Tensor: ...

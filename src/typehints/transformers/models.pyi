import torch

class VitsModelOutput:
    waveform: torch.FloatTensor

class VitsModel:
    @classmethod
    def from_pretrained(cls, model_type: str) -> VitsModel: ...
    def __call__(self, input_ids: torch.Tensor) -> VitsModelOutput: ...

class AutoTokenizer:
    @classmethod
    def from_pretrained(cls, model_type: str) -> AutoTokenizer: ...
    def __call__(self, text: str, return_tensors: str) -> torch.Tensor: ...

class PretrainedConfig:
    def __init__(self, **kwargs): ...

class EncodecConfig(PretrainedConfig):
    sampling_rate: int

class MusicgenDecoderConfig(PretrainedConfig):
    pass

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
    def __call__(self, text: list, padding: bool, return_tensors: str) -> dict: ...

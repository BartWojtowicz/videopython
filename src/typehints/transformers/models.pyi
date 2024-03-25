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

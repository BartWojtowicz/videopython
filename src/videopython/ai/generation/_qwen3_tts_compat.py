"""Compatibility shim for qwen-tts with transformers 5.x.

qwen-tts 0.1.x was built against transformers 4.57.x and uses APIs
that changed in transformers 5.x:

1. ``transformers.utils.generic.check_model_inputs`` -- removed; replaced
   with a no-op decorator.

2. ``ROPE_INIT_FUNCTIONS["default"]`` -- removed from the registry; we
   register a basic RoPE init (no scaling) under the "default" key.

3. ``PretrainedConfig`` attribute access -- in 5.x undefined attributes raise
   AttributeError instead of returning None. We ensure ``pad_token_id``
   defaults to None on all configs.

Call ``apply_patches()`` before importing ``qwen_tts``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig

_applied = False


def _compute_default_rope_parameters(
    config: PreTrainedConfig | None = None,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    """Basic RoPE init without any scaling -- the old 'default' type."""
    import torch

    base = config.rope_theta  # type: ignore[union-attr]
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0  # type: ignore[union-attr]
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)  # type: ignore[union-attr]
    dim = int(head_dim * partial_rotary_factor)

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, 1.0


def apply_patches() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    import transformers.utils.generic as generic_utils
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    # Patch 1: no-op decorator for check_model_inputs
    if not hasattr(generic_utils, "check_model_inputs"):

        def check_model_inputs():
            def decorator(func):
                return func

            return decorator

        generic_utils.check_model_inputs = check_model_inputs  # type: ignore[attr-defined]

    # Patch 2: register 'default' RoPE type (basic, no scaling)
    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    # Patch 3: ensure pad_token_id defaults to None on all configs
    from transformers.configuration_utils import PretrainedConfig

    _original_init = PretrainedConfig.__init__

    def _patched_init(self, **kwargs):  # type: ignore[no-untyped-def]
        _original_init(self, **kwargs)
        if not hasattr(self, "pad_token_id"):
            object.__setattr__(self, "pad_token_id", None)

    PretrainedConfig.__init__ = _patched_init  # type: ignore[method-assign]

    # Patch 4: qwen-tts passes fix_mistral_regex=True to AutoProcessor but
    # transformers 5.x already handles this internally, causing a duplicate kwarg
    # error in Qwen2Tokenizer. We wrap AutoProcessor.from_pretrained to strip it.
    from transformers import AutoProcessor

    _original_proc_from_pretrained = AutoProcessor.from_pretrained.__func__  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    def _patched_proc_from_pretrained(cls, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.pop("fix_mistral_regex", None)
        return _original_proc_from_pretrained(cls, *args, **kwargs)

    AutoProcessor.from_pretrained = _patched_proc_from_pretrained  # type: ignore[assignment]

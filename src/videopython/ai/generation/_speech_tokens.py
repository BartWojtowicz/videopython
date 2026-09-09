from functools import wraps
from typing import Any, Callable


class InvalidSpeechTokens(RuntimeError):
    """A generation cannot safely be decoded by the speech vocoder."""


def restrict_speech_vocabulary(head: Any, vocab_size: int, eos_id: int) -> None:
    """Keep sampling inside the vocoder vocabulary, allowing EOS to end speech."""
    if not 0 < vocab_size <= eos_id < head.out_features:
        raise ValueError("Unexpected Chatterbox speech vocabulary configuration")

    def restrict(_module: Any, _inputs: Any, output: Any) -> Any:
        scores = output
        # Classifier-free guidance combines two heads later. A finite mask
        # avoids inf - inf => NaN while underflowing to zero probability.
        scores[..., vocab_size:eos_id] = -(2**15)
        scores[..., eos_id + 1 :] = -(2**15)
        return scores

    head.register_forward_hook(restrict)


def guard_speech_tokens(inference: Callable[..., Any], vocab_size: int) -> Callable[..., Any]:
    """Reject invalid generated tokens before an embedding lookup poisons CUDA."""

    @wraps(inference)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        tokens = kwargs.get("speech_tokens", args[0] if args else None)
        if tokens is None or not tokens.numel() or not bool(((tokens >= 0) & (tokens < vocab_size)).all()):
            raise InvalidSpeechTokens("Chatterbox returned empty or out-of-range speech tokens")
        return inference(*args, **kwargs)

    return guarded

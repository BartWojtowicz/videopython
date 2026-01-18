"""Backend utilities for videopython.ai module."""

from __future__ import annotations

import os
from typing import Literal

from videopython.ai.exceptions import (
    API_KEY_ENV_VARS,
    BackendError,
    MissingAPIKeyError,
    UnsupportedBackendError,
)

# Backend type definitions per task
TextToVideoBackend = Literal["local", "luma"]
ImageToVideoBackend = Literal["local", "luma", "runway"]
TextToSpeechBackend = Literal["local", "openai", "elevenlabs"]
TextToMusicBackend = Literal["local"]
TextToImageBackend = Literal["local", "openai"]
ImageToTextBackend = Literal["local", "openai", "gemini"]
AudioToTextBackend = Literal["local", "openai", "gemini"]
AudioClassifierBackend = Literal["local"]
LLMBackend = Literal["local", "openai", "gemini"]
TextTranslatorBackend = Literal["openai", "gemini", "local"]
AudioSeparatorBackend = Literal["local"]
VideoDubberBackend = Literal["elevenlabs", "local"]
ObjectSwapperBackend = Literal["local", "replicate"]

# Re-export for backward compatibility
__all__ = [
    "TextToVideoBackend",
    "ImageToVideoBackend",
    "TextToSpeechBackend",
    "TextToMusicBackend",
    "TextToImageBackend",
    "ImageToTextBackend",
    "AudioToTextBackend",
    "AudioClassifierBackend",
    "LLMBackend",
    "TextTranslatorBackend",
    "AudioSeparatorBackend",
    "VideoDubberBackend",
    "ObjectSwapperBackend",
    "BackendError",
    "MissingAPIKeyError",
    "UnsupportedBackendError",
    "get_api_key",
]


def get_api_key(provider: str, api_key: str | None = None) -> str:
    """Get API key for a provider.

    Args:
        provider: Provider name (e.g., 'openai', 'runway', 'luma')
        api_key: Optional explicit API key. If provided, returns this directly.

    Returns:
        The API key string.

    Raises:
        MissingAPIKeyError: If no API key is found.
    """
    if api_key:
        return api_key

    env_var = API_KEY_ENV_VARS.get(provider)
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key

    raise MissingAPIKeyError(provider)

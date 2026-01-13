"""Backend utilities for videopython.ai module."""

from __future__ import annotations

import os
from typing import Literal

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

# Environment variable names per provider
API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "runway": "RUNWAYML_API_KEY",
    "luma": "LUMAAI_API_KEY",
}


class BackendError(Exception):
    """Base exception for backend-related errors."""

    pass


class MissingAPIKeyError(BackendError):
    """Raised when a required API key is not found."""

    def __init__(self, provider: str):
        env_var = API_KEY_ENV_VARS.get(provider, f"{provider.upper()}_API_KEY")
        super().__init__(
            f"API key for '{provider}' not found. Set the {env_var} environment variable or pass api_key parameter."
        )
        self.provider = provider


class UnsupportedBackendError(BackendError):
    """Raised when an unsupported backend is requested."""

    def __init__(self, backend: str, supported: list[str]):
        super().__init__(f"Backend '{backend}' is not supported. Supported backends: {', '.join(supported)}")
        self.backend = backend
        self.supported = supported


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

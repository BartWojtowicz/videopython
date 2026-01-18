"""Exception hierarchy for videopython.ai module."""

# Environment variable names per provider
API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "elevenlabs": "ELEVENLABS_API_KEY",
    "runway": "RUNWAYML_API_KEY",
    "luma": "LUMAAI_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
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


class GenerationError(BackendError):
    """Base exception for generation failures."""

    pass


class LumaGenerationError(GenerationError):
    """Raised when Luma AI video generation fails."""

    pass


class RunwayGenerationError(GenerationError):
    """Raised when Runway video generation fails."""

    pass


class ConfigError(BackendError):
    """Raised when there's an error loading or parsing configuration."""

    pass

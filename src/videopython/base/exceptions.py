"""Exception hierarchy for videopython.base module."""


class VideoPythonError(Exception):
    """Base exception for all videopython errors."""

    pass


class VideoError(VideoPythonError):
    """Base exception for video-related errors."""

    pass


class VideoLoadError(VideoError):
    """Raised when there's an error loading a video file."""

    pass


class VideoMetadataError(VideoError):
    """Raised when there's an error getting video metadata."""

    pass


class AudioError(VideoPythonError):
    """Base exception for audio-related errors."""

    pass


class AudioLoadError(AudioError):
    """Raised when there's an error loading audio."""

    pass


class TransformError(VideoPythonError):
    """Base exception for transformation errors."""

    pass


class InsufficientDurationError(TransformError):
    """Raised when a video doesn't have enough duration for an operation."""

    pass


class IncompatibleVideoError(TransformError):
    """Raised when videos have incompatible properties for merging."""

    pass


class TextRenderError(VideoPythonError):
    """Base exception for text rendering errors."""

    pass


class OutOfBoundsError(TextRenderError):
    """Raised when text would be rendered outside image bounds."""

    pass

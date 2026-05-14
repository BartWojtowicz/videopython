"""Exception hierarchy for videopython.base module."""


class VideoPythonError(Exception):
    """Base exception for all videopython errors."""

    pass


class FFmpegError(VideoPythonError):
    """Base exception for ffmpeg/ffprobe subprocess failures."""

    pass


class FFmpegProbeError(FFmpegError):
    """Raised when an ffprobe invocation or its JSON output fails."""

    pass


class FFmpegRunError(FFmpegError):
    """Raised when a blocking ffmpeg run returns a non-zero exit code."""

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


class TextRenderError(VideoPythonError):
    """Base exception for text rendering errors."""

    pass


class OutOfBoundsError(TextRenderError):
    """Raised when text would be rendered outside image bounds."""

    pass

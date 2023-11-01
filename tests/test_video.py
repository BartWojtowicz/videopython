import pytest
from dataclasses import dataclass

from videopython.base import Video
from videopython.base.video import VideoMetadata
from videopython.project_config import LocationConfig


@dataclass
class _Configuration:
    # Paths
    SHORT_VIDEO_PATH = str(LocationConfig.test_videos_dir /
                           "fast_benchmark.mp4")
    LONG_VIDEO_PATH = str(LocationConfig.test_videos_dir /
                          "slow_benchmark.mp4")
    # Original metadata
    ORIGINAL_SHORT_METADATA = VideoMetadata(height=500,
                                            width=800,
                                            fps=24,
                                            frame_count=288,
                                            total_seconds=12)
    ORIGINAL_LONG_METADATA = VideoMetadata(height=1920,
                                           width=1080,
                                           fps=30,
                                           frame_count=401,
                                           total_seconds=13.37)
    # Target metadata
    TARGET_SHORT_METADATA = VideoMetadata(
        height=400,
        width=600,
        fps=24,
        frame_count=5 * 24,
        total_seconds=5,
    )
    TARGET_LONG_METADATA = VideoMetadata(
        height=1800,
        width=1000,
        fps=30,
        frame_count=10 * 30,
        total_seconds=10,
    )


@pytest.mark.parametrize(
    "video_path, target_metadata",
    [
        (
            _Configuration.SHORT_VIDEO_PATH,
            _Configuration.ORIGINAL_SHORT_METADATA,
        ),
        (_Configuration.LONG_VIDEO_PATH,
         _Configuration.ORIGINAL_LONG_METADATA),
    ],
)
def test_from_video(video_path: str, target_metadata: VideoMetadata):
    """Tests VideoMetadata.from_video."""
    metadata = VideoMetadata.from_video(video_path)
    assert metadata == target_metadata


@pytest.mark.parametrize(
    "video_path, target_metadata, expected",
    [
        (
            _Configuration.SHORT_VIDEO_PATH,
            _Configuration.TARGET_SHORT_METADATA,
            True,
        ),
        (
            _Configuration.LONG_VIDEO_PATH,
            _Configuration.TARGET_LONG_METADATA,
            True,
        ),
        (
            _Configuration.SHORT_VIDEO_PATH,
            _Configuration.TARGET_LONG_METADATA,
            False,
            # Cannot be downsampled, because target video is longer.
        ),
        (
            _Configuration.LONG_VIDEO_PATH,
            _Configuration.TARGET_SHORT_METADATA,
            True,
            # FPS differs, but can be downsampled
        ),
    ],
)
def test_can_be_downsampled_to(
    video_path: str,
    target_metadata: VideoMetadata,
    expected: bool,
):
    """Tests VideoMetadata.can_be_downsampled_to."""
    metadata = VideoMetadata.from_video(video_path)
    assert metadata.can_be_downsampled_to(target_metadata) == expected


@pytest.mark.parametrize(
    "video_path, target_metadata",
    [
        (
            _Configuration.SHORT_VIDEO_PATH,
            _Configuration.ORIGINAL_SHORT_METADATA,
        ),
        (
            _Configuration.LONG_VIDEO_PATH,
            _Configuration.ORIGINAL_LONG_METADATA,
        ),
    ],
)
def test_load_video(video_path: str, target_metadata: VideoMetadata):
    """Tests Video.load_video."""
    video = Video.from_path(video_path)
    assert (video.frames.shape == target_metadata.get_video_shape()).all()

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from videopython.base.video import Video, VideoMetadata
from videopython.project_config import LocationConfig


@dataclass
class _Configuration:
    # Paths

    SMALL_IMG_PATH = str(LocationConfig.test_videos_dir / "small_image.png")
    SMALL_VIDEO_PATH = str(LocationConfig.test_videos_dir / "fast_benchmark.mp4")
    BIG_VIDEO_PATH = str(LocationConfig.test_videos_dir / "slow_benchmark.mp4")
    AUDIO_PATH = str(LocationConfig.test_videos_dir / "test_audio.mp3")

    # Original metadata
    ORIGINAL_SMALL_METADATA = VideoMetadata(height=500, width=800, fps=24, frame_count=288, total_seconds=12)
    ORIGINAL_BIG_METADATA = VideoMetadata(height=1920, width=1080, fps=30, frame_count=401, total_seconds=13.37)

    # Target metadata
    TARGET_SMALL_METADATA = VideoMetadata(
        height=400,
        width=600,
        fps=24,
        frame_count=5 * 24,
        total_seconds=5,
    )
    TARGET_BIG_METADATA = VideoMetadata(
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
            _Configuration.SMALL_VIDEO_PATH,
            _Configuration.ORIGINAL_SMALL_METADATA,
        ),
        (_Configuration.BIG_VIDEO_PATH, _Configuration.ORIGINAL_BIG_METADATA),
    ],
)
def test_from_video(video_path: str, target_metadata: VideoMetadata):
    """Tests VideoMetadata.from_video."""
    metadata = VideoMetadata.from_path(video_path)
    assert metadata == target_metadata


@pytest.mark.parametrize(
    "video_path, target_metadata, expected",
    [
        (
            _Configuration.SMALL_VIDEO_PATH,
            _Configuration.TARGET_SMALL_METADATA,
            True,
        ),
        (
            _Configuration.BIG_VIDEO_PATH,
            _Configuration.TARGET_BIG_METADATA,
            True,
        ),
        (
            _Configuration.SMALL_VIDEO_PATH,
            _Configuration.TARGET_BIG_METADATA,
            False,
            # Cannot be downsampled, because target video is longer.
        ),
        (
            _Configuration.BIG_VIDEO_PATH,
            _Configuration.TARGET_SMALL_METADATA,
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
    metadata = VideoMetadata.from_path(video_path)
    assert metadata.can_be_downsampled_to(target_metadata) == expected


def test_video_from_image():
    with Image.open(_Configuration.SMALL_IMG_PATH) as img:
        img = np.array(img.convert("RGB"))  # Otherwise we get a 4 channel image for .png

    video = Video.from_image(img, fps=30, length_seconds=0.5)

    assert video.frames.shape == (15, *img.shape)
    assert video.fps == 30
    assert np.array_equal(video.frames[0], img)
    assert np.array_equal(video.frames[-1], img)


@pytest.mark.parametrize(
    "video_path, target_metadata",
    [
        (
            _Configuration.SMALL_VIDEO_PATH,
            _Configuration.ORIGINAL_SMALL_METADATA,
        ),
        (
            _Configuration.BIG_VIDEO_PATH,
            _Configuration.ORIGINAL_BIG_METADATA,
        ),
    ],
)
def test_load_video(video_path: str, target_metadata: VideoMetadata):
    """Tests Video.load_video."""
    video = Video.from_path(video_path)
    assert (video.frames.shape == target_metadata.get_video_shape()).all()


def test_save_with_audio(black_frames_video):
    black_frames_video.add_audio_from_file(_Configuration.AUDIO_PATH)
    with tempfile.TemporaryDirectory() as temp_dir:
        saved_path = black_frames_video.save(Path(temp_dir) / "test_add_audio_from_file.mp4")
        black_frames_video.save(saved_path)

        assert np.all(Video.from_path(saved_path).frames == black_frames_video.frames)
        assert Path(saved_path).exists()

    assert black_frames_video.audio is not None

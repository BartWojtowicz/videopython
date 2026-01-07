import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.test_config import (
    BIG_VIDEO_METADATA,
    BIG_VIDEO_PATH,
    SMALL_VIDEO_METADATA,
    SMALL_VIDEO_PATH,
    TEST_AUDIO_PATH,
    TEST_IMAGE_PATH,
)
from videopython.base.video import Video, VideoMetadata


@pytest.mark.parametrize(
    "video_path, original_metadata",
    [
        (
            SMALL_VIDEO_PATH,
            SMALL_VIDEO_METADATA,
        ),
        (BIG_VIDEO_PATH, BIG_VIDEO_METADATA),
    ],
)
def test_from_video(video_path: str, original_metadata: VideoMetadata):
    """Tests VideoMetadata.from_video."""
    metadata = VideoMetadata.from_path(video_path)
    assert metadata == original_metadata


_TARGET_SMALL_METADATA = VideoMetadata(
    height=400,
    width=600,
    fps=24,
    frame_count=5 * 24,
    total_seconds=5,
)
_TARGET_BIG_METADATA = VideoMetadata(
    height=1800,
    width=1000,
    fps=30,
    frame_count=10 * 30,
    total_seconds=10,
)


@pytest.mark.parametrize(
    "video_path, target_metadata, expected",
    [
        (
            SMALL_VIDEO_PATH,
            _TARGET_SMALL_METADATA,
            True,
        ),
        (
            BIG_VIDEO_PATH,
            _TARGET_BIG_METADATA,
            True,
        ),
        (
            SMALL_VIDEO_PATH,
            _TARGET_BIG_METADATA,
            False,
            # Cannot be downsampled, because target video is longer.
        ),
        (
            BIG_VIDEO_PATH,
            _TARGET_SMALL_METADATA,
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
    with Image.open(TEST_IMAGE_PATH) as img:
        img = np.array(img.convert("RGB"))  # Otherwise we get a 4 channel image for .png

    video = Video.from_image(img, fps=30, length_seconds=0.5)

    assert video.frames.shape == (15, *img.shape)
    assert video.fps == 30
    assert np.array_equal(video.frames[0], img)
    assert np.array_equal(video.frames[-1], img)


@pytest.mark.parametrize(
    "video_path, original_metadata",
    [
        (
            SMALL_VIDEO_PATH,
            SMALL_VIDEO_METADATA,
        ),
        (
            BIG_VIDEO_PATH,
            BIG_VIDEO_METADATA,
        ),
    ],
)
def test_load_video_shape(video_path: str, original_metadata: VideoMetadata):
    """Tests Video.load_video."""
    video = Video.from_path(video_path)
    assert (video.frames.shape == original_metadata.get_video_shape()).all()


def test_save_and_load():
    test_video = Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
    with tempfile.TemporaryDirectory() as temp_dir:
        saved_path = test_video.save(Path(temp_dir) / "test_save_and_load.mp4")
        test_video.save(saved_path)
        assert np.all(Video.from_path(saved_path).frames == test_video.frames)
        assert Path(saved_path).exists()


def test_save_with_audio():
    test_video = Video.from_image(np.zeros((100, 100, 3), dtype=np.uint8), fps=24, length_seconds=5.0)
    test_video.add_audio_from_file(TEST_AUDIO_PATH, overlay=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        saved_path = test_video.save(Path(temp_dir) / "test_add_audio_from_file.mp4")
        test_video.save(saved_path)

        assert np.all(Video.from_path(saved_path).frames == test_video.frames)
        assert Path(saved_path).exists()

    assert test_video.audio is not None

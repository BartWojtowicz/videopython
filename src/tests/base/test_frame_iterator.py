"""Tests for memory-efficient frame iteration and extraction."""

import numpy as np
import pytest

from tests.test_config import SMALL_VIDEO_PATH
from videopython import _ffmpeg
from videopython.base import Video, VideoLoadError
from videopython.base.video import (
    FrameIterator,
    VideoMetadata,
    extract_frames_at_indices,
    extract_frames_at_times,
)


@pytest.fixture
def truncated_video(tmp_path):
    complete = tmp_path / "complete.mp4"
    _ffmpeg.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            SMALL_VIDEO_PATH,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(complete),
        ]
    )
    damaged = tmp_path / "truncated.mp4"
    data = complete.read_bytes()
    damaged.write_bytes(data[: int(len(data) * 0.9)])
    VideoMetadata.from_path(damaged)
    return damaged


class TestFrameIterator:
    """Tests for FrameIterator class."""

    def test_iteration_yields_frames(self):
        """Test that iterator yields frame tuples with correct shapes."""
        metadata = VideoMetadata.from_path(SMALL_VIDEO_PATH)

        with FrameIterator(SMALL_VIDEO_PATH) as frames:
            idx, frame = next(iter(frames))
            assert idx == 0
            assert frame.shape == (metadata.height, metadata.width, 3)
            assert frame.dtype == np.uint8

    def test_all_frames_iterable(self):
        """Test that all frames can be iterated."""
        metadata = VideoMetadata.from_path(SMALL_VIDEO_PATH)

        frame_count = 0
        with FrameIterator(SMALL_VIDEO_PATH) as frames:
            for idx, frame in frames:
                frame_count += 1
                assert frame.shape == (metadata.height, metadata.width, 3)

        assert abs(frame_count - metadata.frame_count) <= 2

    def test_frame_indices_sequential(self):
        """Test that frame indices are sequential."""
        with FrameIterator(SMALL_VIDEO_PATH) as frames:
            prev_idx = -1
            for idx, _ in frames:
                assert idx == prev_idx + 1
                prev_idx = idx
                if idx > 10:  # Only check first few frames
                    break

    def test_context_manager_cleanup(self):
        """Test that resources are cleaned up after context exit."""
        iterator = FrameIterator(SMALL_VIDEO_PATH)
        with iterator as frames:
            next(iter(frames))
        assert iterator._iter is None

    def test_start_second_offset(self):
        """Test that start_second skips frames correctly."""
        metadata = VideoMetadata.from_path(SMALL_VIDEO_PATH)
        start = 1.0

        with FrameIterator(SMALL_VIDEO_PATH, start_second=start) as frames:
            idx, _ = next(iter(frames))
            expected_start_frame = int(start * metadata.fps)
            # Allow some tolerance for seek accuracy
            assert abs(idx - expected_start_frame) <= 2

    def test_end_second_stops_iteration(self):
        """Test that end_second limits the frames returned."""
        metadata = VideoMetadata.from_path(SMALL_VIDEO_PATH)
        end = 2.0  # Stop at 2 seconds

        frame_count = 0
        with FrameIterator(SMALL_VIDEO_PATH, end_second=end) as frames:
            for idx, _ in frames:
                frame_count += 1

        expected_frames = int(end * metadata.fps)
        # Allow some tolerance for timing
        assert abs(frame_count - expected_frames) <= 5

    def test_file_not_found_raises(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            FrameIterator("/nonexistent/path/video.mp4")

    def test_frames_are_writable(self):
        """Test that yielded frames are writable (not read-only buffers)."""
        with FrameIterator(SMALL_VIDEO_PATH) as frames:
            _, frame = next(iter(frames))
            # Should be able to modify without error
            frame[0, 0, 0] = 255

    def test_corrupt_source_raises_instead_of_returning_partial_frames(self, truncated_video):
        with pytest.raises(VideoLoadError, match="FFmpeg failed") as exc_info:
            list(FrameIterator(truncated_video))
        assert "ffmpeg version" not in str(exc_info.value)

        with pytest.raises(VideoLoadError, match="FFmpeg failed"):
            Video.from_path(str(truncated_video))

    def test_audio_longer_than_video_does_not_fail_decode(self, tmp_path):
        source_meta = VideoMetadata.from_path(SMALL_VIDEO_PATH)
        mixed_duration = source_meta.total_seconds + 2
        path = tmp_path / "longer-audio.mkv"
        _ffmpeg.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                SMALL_VIDEO_PATH,
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=1000:duration={mixed_duration}",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "pcm_s16le",
                str(path),
            ]
        )

        assert VideoMetadata.from_path(path).frame_count > source_meta.frame_count
        assert len(Video.from_path(str(path)).frames) == source_meta.frame_count
        assert len(list(FrameIterator(path))) == source_meta.frame_count


class TestExtractFramesAtIndices:
    """Tests for extract_frames_at_indices function."""

    def test_extract_specific_frames(self):
        """Test extracting specific frame indices."""
        metadata = VideoMetadata.from_path(SMALL_VIDEO_PATH)
        indices = [0, 10, 20]

        frames = extract_frames_at_indices(SMALL_VIDEO_PATH, indices)

        assert frames.shape[0] == 3
        assert frames.shape[1:] == (metadata.height, metadata.width, 3)
        assert frames.dtype == np.uint8

    def test_extract_single_frame(self):
        """Test extracting a single frame."""
        metadata = VideoMetadata.from_path(SMALL_VIDEO_PATH)

        frames = extract_frames_at_indices(SMALL_VIDEO_PATH, [0])

        assert frames.shape[0] == 1
        assert frames.shape[1:] == (metadata.height, metadata.width, 3)

    def test_extract_empty_list(self):
        """Test extracting with empty frame list."""
        metadata = VideoMetadata.from_path(SMALL_VIDEO_PATH)

        frames = extract_frames_at_indices(SMALL_VIDEO_PATH, [])

        assert frames.shape[0] == 0
        assert frames.shape[1:] == (metadata.height, metadata.width, 3)

    def test_extract_duplicate_indices(self):
        """Test that duplicate indices return duplicated frames."""
        indices = [0, 0, 10]

        frames = extract_frames_at_indices(SMALL_VIDEO_PATH, indices)

        # Should return 3 frames (with first two being the same)
        assert frames.shape[0] == 3
        # First two frames should be identical
        assert np.array_equal(frames[0], frames[1])

    def test_extract_unsorted_indices(self):
        """Test that unsorted indices are handled correctly."""
        indices = [20, 0, 10]

        frames = extract_frames_at_indices(SMALL_VIDEO_PATH, indices)

        assert frames.shape[0] == 3
        # Frames should be in the order requested, not sorted
        # (This is verified by the implementation reordering logic)

    def test_file_not_found_raises(self):
        """Test that non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            extract_frames_at_indices("/nonexistent/path/video.mp4", [0])

    def test_frames_are_writable(self):
        """Test that extracted frames are writable."""
        frames = extract_frames_at_indices(SMALL_VIDEO_PATH, [0])
        # Should be able to modify without error
        frames[0, 0, 0, 0] = 255


class TestExtractFramesAtTimes:
    """Tests for extract_frames_at_times function."""

    def test_extract_at_times(self):
        """Test extracting frames at specific timestamps."""
        metadata = VideoMetadata.from_path(SMALL_VIDEO_PATH)
        timestamps = [0.0, 1.0, 2.0]

        frames = extract_frames_at_times(SMALL_VIDEO_PATH, timestamps)

        assert frames.shape[0] == 3
        assert frames.shape[1:] == (metadata.height, metadata.width, 3)

    def test_extract_at_single_time(self):
        """Test extracting frame at single timestamp."""
        metadata = VideoMetadata.from_path(SMALL_VIDEO_PATH)

        frames = extract_frames_at_times(SMALL_VIDEO_PATH, [0.5])

        assert frames.shape[0] == 1
        assert frames.shape[1:] == (metadata.height, metadata.width, 3)

    def test_extract_at_empty_times(self):
        """Test extracting with empty timestamp list."""
        frames = extract_frames_at_times(SMALL_VIDEO_PATH, [])

        assert frames.shape[0] == 0


def test_batched_extraction_matches_decoded_frames(tmp_path, monkeypatch):
    import subprocess

    path = tmp_path / "colors.mp4"
    frames = np.zeros((30, 48, 64, 3), dtype=np.uint8)
    frames[:10, :, :, 0] = 255
    frames[10:20, :, :, 1] = 255
    frames[20:, :, :, 2] = 255
    Video(frames, fps=10).save(path)
    decoded = Video.from_path(path).frames
    VideoMetadata.clear_cache()
    calls = []
    original = subprocess.Popen

    def record(args, *a, **kwargs):
        calls.append(args)
        return original(args, *a, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", record)
    selected = extract_frames_at_times(path, [2.1, 0.2, 1.1, 0.2])
    np.testing.assert_array_equal(selected, decoded[[21, 2, 11, 2]])
    assert len([args for args in calls if args[0] == "ffprobe"]) == 1
    [command] = [args for args in calls if args[0] == "ffmpeg"]
    assert command[command.index("-frames:v") + 1] == "3"


@pytest.mark.parametrize("indices", [[0, 10, 100000], [100000, 0], [-1, 0]])
def test_selected_frames_reject_short_decode(indices):
    with pytest.raises(VideoLoadError, match="incomplete decode"):
        extract_frames_at_indices(SMALL_VIDEO_PATH, indices)

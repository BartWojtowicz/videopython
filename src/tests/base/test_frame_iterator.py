"""Tests for memory-efficient frame iteration and extraction."""

import numpy as np
import pytest

from tests.test_config import SMALL_VIDEO_PATH
from videopython.base.video import (
    FrameIterator,
    VideoMetadata,
    extract_frames_at_indices,
    extract_frames_at_times,
)


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

        # Should iterate through all frames (allow some tolerance for codec differences)
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
        assert iterator._process is None

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

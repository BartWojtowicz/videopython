import cv2
import numpy as np
import pytest

from videopython.base.transforms import (
    CutFrames,
    CutSeconds,
    PictureInPicture,
    Resize,
    SpeedChange,
)
from videopython.base.video import Video


@pytest.mark.parametrize("start, end", [(0, 100), (100, 101), (100, 120)])
def test_cut_frames(start, end, small_video):
    cut_frames = CutFrames(start=start, end=end)
    start_frame = small_video.frames[start].copy()
    transformed = cut_frames.apply(small_video)
    assert len(transformed.frames) == (end - start)
    assert np.all(transformed.frames[0] == start_frame)


@pytest.mark.parametrize("start, end", [(0, 0.5), (0, 1), (0.5, 1.5)])
def test_cut_seconds(start, end, small_video):
    cut_seconds = CutSeconds(start=start, end=end)
    start_frame = small_video.frames[round(start * small_video.fps)].copy()
    transformed = cut_seconds.apply(small_video)
    assert len(transformed.frames) == round((end - start) * small_video.fps)
    assert np.all(transformed.frames[0] == start_frame)


@pytest.mark.parametrize(
    "height,width",
    [
        (
            40,
            60,
        ),
        (
            500,
            700,
        ),
    ],
)
def test_video_resize(height, width, small_video):
    """Tests Video.resize."""

    resample = Resize(height=height, width=width)
    video = resample.apply(small_video)

    assert video.frames.shape[1:3] == (height, width)
    assert np.all(
        video.frames[0]
        == cv2.resize(
            small_video.frames[0],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    )
    assert np.all(
        video.frames[-1]
        == cv2.resize(
            small_video.frames[-1],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    )
    assert np.all(
        video.frames[len(video.frames) // 2]
        == cv2.resize(
            small_video.frames[len(small_video.frames) // 2],
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    )


class TestSpeedChange:
    """Tests for SpeedChange transformation."""

    def test_speed_up_2x(self, small_video):
        """Test 2x speed results in half the frames."""
        original_frames = len(small_video.frames)
        transform = SpeedChange(speed=2.0)
        result = transform.apply(small_video)

        # 2x speed should give approximately half the frames
        assert len(result.frames) == original_frames // 2

    def test_slow_down_half(self, small_video):
        """Test 0.5x speed results in double the frames."""
        original_frames = len(small_video.frames)
        transform = SpeedChange(speed=0.5)
        result = transform.apply(small_video)

        # 0.5x speed should give approximately double the frames
        assert len(result.frames) == original_frames * 2

    def test_speed_1x_no_change(self, small_video):
        """Test 1x speed keeps same frame count."""
        original_frames = len(small_video.frames)
        transform = SpeedChange(speed=1.0)
        result = transform.apply(small_video)

        assert len(result.frames) == original_frames

    def test_speed_ramp(self, small_video):
        """Test speed ramping from 1x to 2x."""
        original_frames = len(small_video.frames)
        transform = SpeedChange(speed=1.0, end_speed=2.0)
        result = transform.apply(small_video)

        # Average speed is 1.5x, so should have ~2/3 the frames
        expected_frames = int(original_frames / 1.5)
        assert abs(len(result.frames) - expected_frames) <= 1

    def test_invalid_speed_raises(self):
        """Test that invalid speed values raise errors."""
        with pytest.raises(ValueError):
            SpeedChange(speed=0)

        with pytest.raises(ValueError):
            SpeedChange(speed=-1.0)

        with pytest.raises(ValueError):
            SpeedChange(speed=1.0, end_speed=0)

    def test_preserves_frame_shape(self, small_video):
        """Test that speed change preserves frame dimensions."""
        original_shape = small_video.frame_shape
        transform = SpeedChange(speed=2.0)
        result = transform.apply(small_video)

        assert result.frame_shape == original_shape


class TestPictureInPicture:
    """Tests for PictureInPicture transformation."""

    @pytest.fixture
    def main_video(self):
        """Create a main video for testing."""
        frames = np.full((30, 200, 300, 3), 100, dtype=np.uint8)  # Gray video
        return Video.from_frames(frames, fps=30)

    @pytest.fixture
    def overlay_video(self):
        """Create an overlay video for testing."""
        frames = np.full((30, 100, 150, 3), 200, dtype=np.uint8)  # Lighter gray video
        return Video.from_frames(frames, fps=30)

    def test_basic_overlay(self, main_video, overlay_video):
        """Test basic PIP overlay."""
        transform = PictureInPicture(overlay=overlay_video, position=(0.8, 0.8), scale=0.25)
        result = transform.apply(main_video)

        # Should preserve main video dimensions
        assert result.frame_shape == main_video.frame_shape
        assert len(result.frames) == len(main_video.frames)

    def test_overlay_position_corners(self, main_video, overlay_video):
        """Test overlay at different corner positions."""
        positions = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]

        for pos in positions:
            transform = PictureInPicture(overlay=overlay_video, position=pos, scale=0.2)
            result = transform.apply(main_video.copy())
            assert result.frame_shape == main_video.frame_shape

    def test_overlay_scale(self, main_video, overlay_video):
        """Test different overlay scales."""
        for scale in [0.1, 0.25, 0.5]:
            transform = PictureInPicture(overlay=overlay_video, position=(0.5, 0.5), scale=scale)
            result = transform.apply(main_video.copy())
            assert result.frame_shape == main_video.frame_shape

    def test_overlay_with_border(self, main_video, overlay_video):
        """Test overlay with border."""
        transform = PictureInPicture(
            overlay=overlay_video,
            position=(0.7, 0.7),
            scale=0.25,
            border_width=5,
            border_color=(255, 0, 0),
        )
        result = transform.apply(main_video)
        assert result.frame_shape == main_video.frame_shape

    def test_overlay_with_rounded_corners(self, main_video, overlay_video):
        """Test overlay with rounded corners."""
        transform = PictureInPicture(
            overlay=overlay_video,
            position=(0.7, 0.7),
            scale=0.25,
            corner_radius=10,
        )
        result = transform.apply(main_video)
        assert result.frame_shape == main_video.frame_shape

    def test_overlay_with_opacity(self, main_video, overlay_video):
        """Test overlay with reduced opacity."""
        transform = PictureInPicture(
            overlay=overlay_video,
            position=(0.5, 0.5),
            scale=0.25,
            opacity=0.5,
        )
        result = transform.apply(main_video)
        assert result.frame_shape == main_video.frame_shape

    def test_overlay_loops_when_shorter(self, main_video):
        """Test that shorter overlay loops."""
        # Create shorter overlay (10 frames vs 30 in main)
        short_overlay = Video.from_frames(
            np.full((10, 100, 150, 3), 200, dtype=np.uint8),
            fps=30,
        )
        transform = PictureInPicture(overlay=short_overlay, position=(0.5, 0.5), scale=0.25)
        result = transform.apply(main_video)

        # Result should have same length as main video
        assert len(result.frames) == len(main_video.frames)

    def test_invalid_position_raises(self, main_video, overlay_video):
        """Test that invalid position raises error."""
        with pytest.raises(ValueError):
            PictureInPicture(overlay=overlay_video, position=(1.5, 0.5))

        with pytest.raises(ValueError):
            PictureInPicture(overlay=overlay_video, position=(-0.1, 0.5))

    def test_invalid_scale_raises(self, main_video, overlay_video):
        """Test that invalid scale raises error."""
        with pytest.raises(ValueError):
            PictureInPicture(overlay=overlay_video, scale=0)

        with pytest.raises(ValueError):
            PictureInPicture(overlay=overlay_video, scale=1.5)

    def test_invalid_opacity_raises(self, main_video, overlay_video):
        """Test that invalid opacity raises error."""
        with pytest.raises(ValueError):
            PictureInPicture(overlay=overlay_video, opacity=-0.1)

        with pytest.raises(ValueError):
            PictureInPicture(overlay=overlay_video, opacity=1.5)

    def test_all_options_combined(self, main_video, overlay_video):
        """Test with all options enabled."""
        transform = PictureInPicture(
            overlay=overlay_video,
            position=(0.75, 0.75),
            scale=0.3,
            border_width=3,
            border_color=(0, 255, 0),
            corner_radius=8,
            opacity=0.9,
        )
        result = transform.apply(main_video)
        assert result.frame_shape == main_video.frame_shape

    def test_audio_mode_main(self, main_video, overlay_video):
        """Test audio_mode='main' keeps only main audio."""
        transform = PictureInPicture(
            overlay=overlay_video,
            position=(0.5, 0.5),
            scale=0.25,
            audio_mode="main",
        )
        result = transform.apply(main_video)

        # Should preserve main video audio duration
        assert result.audio is not None
        assert abs(result.audio.metadata.duration_seconds - len(result.frames) / result.fps) < 0.1

    def test_audio_mode_overlay(self, main_video, overlay_video):
        """Test audio_mode='overlay' uses only overlay audio."""
        transform = PictureInPicture(
            overlay=overlay_video,
            position=(0.5, 0.5),
            scale=0.25,
            audio_mode="overlay",
        )
        result = transform.apply(main_video)

        # Result should have audio with correct duration
        assert result.audio is not None
        assert abs(result.audio.metadata.duration_seconds - len(result.frames) / result.fps) < 0.1

    def test_audio_mode_mix(self, main_video, overlay_video):
        """Test audio_mode='mix' combines both audio tracks."""
        transform = PictureInPicture(
            overlay=overlay_video,
            position=(0.5, 0.5),
            scale=0.25,
            audio_mode="mix",
            audio_mix=(1.0, 1.0),
        )
        result = transform.apply(main_video)

        assert result.audio is not None
        assert abs(result.audio.metadata.duration_seconds - len(result.frames) / result.fps) < 0.1

    def test_audio_mix_factors(self, main_video, overlay_video):
        """Test custom audio_mix volume factors."""
        transform = PictureInPicture(
            overlay=overlay_video,
            position=(0.5, 0.5),
            scale=0.25,
            audio_mode="mix",
            audio_mix=(0.5, 0.8),
        )
        result = transform.apply(main_video)

        assert result.audio is not None

    def test_invalid_audio_mode_raises(self, main_video, overlay_video):
        """Test that invalid audio_mode raises ValueError."""
        with pytest.raises(ValueError, match="audio_mode"):
            PictureInPicture(overlay=overlay_video, audio_mode="invalid")

    def test_negative_audio_mix_raises(self, main_video, overlay_video):
        """Test that negative audio_mix factors raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            PictureInPicture(overlay=overlay_video, audio_mix=(-1.0, 1.0))

        with pytest.raises(ValueError, match="non-negative"):
            PictureInPicture(overlay=overlay_video, audio_mix=(1.0, -0.5))


class TestSpeedChangeAudio:
    """Tests for SpeedChange audio adjustment."""

    @pytest.fixture
    def video_with_audio(self):
        """Create a video with non-silent audio for testing."""
        from videopython.base.audio import Audio, AudioMetadata

        # Create video frames (1 second at 30fps)
        frames = np.full((30, 100, 150, 3), 128, dtype=np.uint8)
        video = Video.from_frames(frames, fps=30)

        # Create non-silent audio (sine wave)
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio_data = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        video.audio = Audio(
            data=audio_data,
            metadata=AudioMetadata(
                sample_rate=sample_rate,
                channels=1,
                sample_width=2,
                duration_seconds=duration,
                frame_count=len(audio_data),
            ),
        )
        return video

    def test_speed_up_2x_audio_duration(self, video_with_audio):
        """Test 2x speed results in adjusted audio duration."""
        original_video_duration = len(video_with_audio.frames) / video_with_audio.fps

        transform = SpeedChange(speed=2.0, adjust_audio=True)
        result = transform.apply(video_with_audio)

        # Video duration should be halved
        new_video_duration = len(result.frames) / result.fps
        assert abs(new_video_duration - original_video_duration / 2) < 0.1

        # Audio duration should match video duration
        assert abs(result.audio.metadata.duration_seconds - new_video_duration) < 0.2

    def test_slow_down_half_audio_duration(self, video_with_audio):
        """Test 0.5x speed results in adjusted audio duration."""
        original_video_duration = len(video_with_audio.frames) / video_with_audio.fps

        transform = SpeedChange(speed=0.5, adjust_audio=True)
        result = transform.apply(video_with_audio)

        # Video duration should be doubled
        new_video_duration = len(result.frames) / result.fps
        assert abs(new_video_duration - original_video_duration * 2) < 0.1

        # Audio duration should match video duration
        assert abs(result.audio.metadata.duration_seconds - new_video_duration) < 0.5

    def test_speed_change_adjust_audio_false(self, video_with_audio):
        """Test adjust_audio=False slices without time-stretching."""
        transform = SpeedChange(speed=2.0, adjust_audio=False)
        result = transform.apply(video_with_audio)

        # Audio should still match video duration (sliced, not time-stretched)
        new_video_duration = len(result.frames) / result.fps
        assert abs(result.audio.metadata.duration_seconds - new_video_duration) < 0.2

    def test_speed_change_silent_audio(self, small_video):
        """Test speed change with silent audio."""
        # small_video fixture has silent audio by default
        transform = SpeedChange(speed=2.0, adjust_audio=True)
        result = transform.apply(small_video)

        # Should complete without error
        assert result.audio is not None

    def test_speed_ramp_audio(self, video_with_audio):
        """Test audio adjustment with speed ramp."""
        transform = SpeedChange(speed=1.0, end_speed=2.0, adjust_audio=True)
        result = transform.apply(video_with_audio)

        # Audio duration should match video duration
        new_video_duration = len(result.frames) / result.fps
        assert abs(result.audio.metadata.duration_seconds - new_video_duration) < 0.3

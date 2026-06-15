import shutil
import tempfile
import threading
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
from videopython.audio.audio import Audio, AudioMetadata
from videopython.base import video as _video_mod
from videopython.base.video import Video, VideoMetadata


def _tone(duration_seconds: float, sample_rate: int, freq: float = 440.0) -> Audio:
    """Build a non-silent mono tone Audio at the given sample rate."""
    frame_count = int(duration_seconds * sample_rate)
    t = np.arange(frame_count, dtype=np.float32) / sample_rate
    data = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    metadata = AudioMetadata(
        sample_rate=sample_rate,
        channels=1,
        sample_width=2,
        duration_seconds=frame_count / sample_rate,
        frame_count=frame_count,
    )
    return Audio(data, metadata)


@pytest.fixture(autouse=True)
def _clear_metadata_cache():
    """Keep the module-global probe cache from leaking across tests."""
    VideoMetadata.clear_cache()
    yield
    VideoMetadata.clear_cache()


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
    assert video.frames.shape == tuple(original_metadata.get_video_shape())


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


def test_save_rejects_odd_dimensions_with_clear_error():
    odd_video = Video.from_image(np.zeros((101, 100, 3), dtype=np.uint8), fps=24, length_seconds=1.0)
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="requires even frame dimensions.*100x101"):
            odd_video.save(Path(temp_dir) / "odd_dims.mp4")


def _count_probes(monkeypatch) -> dict[str, int]:
    """Patch VideoMetadata._run_ffprobe with a passthrough counter; returns the counter dict."""
    calls = {"n": 0}
    original = VideoMetadata._run_ffprobe

    def counted(video_path):
        calls["n"] += 1
        return original(video_path)

    monkeypatch.setattr(VideoMetadata, "_run_ffprobe", staticmethod(counted))
    return calls


def test_from_path_caches_probe(monkeypatch):
    """A repeated probe of the same unchanged file hits the cache and skips ffprobe."""
    calls = _count_probes(monkeypatch)
    first = VideoMetadata.from_path(SMALL_VIDEO_PATH)
    second = VideoMetadata.from_path(SMALL_VIDEO_PATH)
    assert calls["n"] == 1
    assert first == second == SMALL_VIDEO_METADATA


def test_from_path_reprobes_when_file_changes(monkeypatch, tmp_path):
    """Overwriting a file in place invalidates its cache entry (mtime_ns/size change)."""
    calls = _count_probes(monkeypatch)
    target = tmp_path / "clip.mp4"
    shutil.copy(SMALL_VIDEO_PATH, target)
    first = VideoMetadata.from_path(target)
    assert first == SMALL_VIDEO_METADATA

    shutil.copy(BIG_VIDEO_PATH, target)  # different content -> different size and mtime
    second = VideoMetadata.from_path(target)
    assert calls["n"] == 2
    assert second != first


def test_clear_cache_forces_reprobe(monkeypatch):
    calls = _count_probes(monkeypatch)
    VideoMetadata.from_path(SMALL_VIDEO_PATH)
    VideoMetadata.clear_cache()
    VideoMetadata.from_path(SMALL_VIDEO_PATH)
    assert calls["n"] == 2


def test_missing_file_raises_and_is_not_cached():
    with pytest.raises(FileNotFoundError):
        VideoMetadata.from_path("/no/such/file.mp4")
    assert len(_video_mod._METADATA_CACHE) == 0


def test_cache_is_lru_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(_video_mod, "_METADATA_CACHE_MAXSIZE", 2)
    for i in range(3):
        path = tmp_path / f"clip_{i}.mp4"
        shutil.copy(SMALL_VIDEO_PATH, path)
        VideoMetadata.from_path(path)
    assert len(_video_mod._METADATA_CACHE) <= 2


def test_from_path_is_threadsafe():
    """Concurrent probes of the same and different files must not race the LRU dict."""
    paths = [SMALL_VIDEO_PATH, BIG_VIDEO_PATH]
    results: list[VideoMetadata] = []
    errors: list[Exception] = []

    def worker(path: str):
        try:
            results.append(VideoMetadata.from_path(path))
        except Exception as exc:  # noqa: BLE001 - surfaced via the errors list
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(paths[i % 2],)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert all(result in (SMALL_VIDEO_METADATA, BIG_VIDEO_METADATA) for result in results)


def test_add_audio_overlay_resamples_to_existing_rate_without_drift():
    """Overlaying audio at a mismatched sample rate must resample to the existing
    track's rate and produce no duration/length drift."""
    video = Video.from_image(np.zeros((64, 64, 3), dtype=np.uint8), fps=24, length_seconds=3.0)
    # Existing track at the canonical 44100 Hz (non-silent so it triggers overlay).
    video.audio = _tone(video.total_seconds, sample_rate=44100)
    assert not video.audio.is_silent

    # Incoming audio at a mismatched rate; without resampling overlay() would raise.
    incoming = _tone(video.total_seconds, sample_rate=48000)

    result = video.add_audio(incoming, overlay=True)

    # Canonical target is the existing track's sample rate.
    assert result.audio.metadata.sample_rate == 44100
    # No A/V drift: audio duration matches the video duration.
    assert result.audio.metadata.duration_seconds == pytest.approx(video.total_seconds, abs=0.01)
    expected_frames = round(video.total_seconds * 44100)
    assert result.audio.metadata.frame_count == pytest.approx(expected_frames, abs=2)
    assert len(result.audio.data) == pytest.approx(expected_frames, abs=2)


def test_add_audio_overlay_longer_mismatched_rate_no_drift():
    """A longer incoming clip at a mismatched rate must be resampled then trimmed
    to the video duration without drift."""
    video = Video.from_image(np.zeros((64, 64, 3), dtype=np.uint8), fps=24, length_seconds=2.0)
    video.audio = _tone(video.total_seconds, sample_rate=44100)

    # Incoming clip is longer than the video and at a different rate.
    incoming = _tone(5.0, sample_rate=22050)

    result = video.add_audio(incoming, overlay=True)

    assert result.audio.metadata.sample_rate == 44100
    assert result.audio.metadata.duration_seconds == pytest.approx(video.total_seconds, abs=0.01)
    expected_frames = round(video.total_seconds * 44100)
    assert len(result.audio.data) == pytest.approx(expected_frames, abs=2)


def test_add_audio_replace_keeps_incoming_rate():
    """Pure replace (overlay=False) must keep the incoming sample rate as-is."""
    video = Video.from_image(np.zeros((64, 64, 3), dtype=np.uint8), fps=24, length_seconds=2.0)
    video.audio = _tone(video.total_seconds, sample_rate=44100)

    incoming = _tone(video.total_seconds, sample_rate=48000)
    result = video.add_audio(incoming, overlay=False)

    assert result.audio.metadata.sample_rate == 48000
    assert result.audio.metadata.duration_seconds == pytest.approx(video.total_seconds, abs=0.01)


def test_add_audio_attach_to_silent_keeps_incoming_rate():
    """Attaching to a video with only the default silent track keeps the incoming rate."""
    video = Video.from_image(np.zeros((64, 64, 3), dtype=np.uint8), fps=24, length_seconds=2.0)
    assert video.audio.is_silent

    incoming = _tone(video.total_seconds, sample_rate=48000)
    result = video.add_audio(incoming, overlay=True)

    assert result.audio.metadata.sample_rate == 48000
    assert result.audio.metadata.duration_seconds == pytest.approx(video.total_seconds, abs=0.01)

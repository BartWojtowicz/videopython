from pathlib import Path

from videopython.base.video import VideoMetadata

# Repo paths
TEST_ROOT_DIR: Path = Path(__file__).parent
TEST_DATA_DIR: Path = TEST_ROOT_DIR / "test_data"

# Videos
SMALL_VIDEO_PATH = str(TEST_DATA_DIR / "small_video.mp4")
SMALL_VIDEO_METADATA = VideoMetadata(height=500, width=800, fps=24, frame_count=288, total_seconds=12)

BIG_VIDEO_PATH = str(TEST_DATA_DIR / "big_video.mp4")
BIG_VIDEO_METADATA = VideoMetadata(height=1920, width=1080, fps=30, frame_count=401, total_seconds=13.37)

# Other
TEST_AUDIO_PATH = str(TEST_DATA_DIR / "test_audio.mp3")
TEST_IMAGE_PATH = str(TEST_DATA_DIR / "small_image.png")
TEST_FONT_PATH = str(TEST_DATA_DIR / "test_font.ttf")

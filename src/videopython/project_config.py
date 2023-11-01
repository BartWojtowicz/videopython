from dataclasses import dataclass
from pathlib import Path


@dataclass
class LocationConfig:
    project_root: Path = Path(__file__).parent
    data_dir: Path = project_root.parent.parent / "data"
    test_dir: Path = project_root.parent.parent / "tests"
    test_videos_dir: Path = test_dir / "test_data"
    downloaded_videos_dir: Path = data_dir / "downloaded"
    exported_videos_dir: Path = data_dir / "exported"

    def __post_init__(self):
        self.data_dir.mkdir(exist_ok=True)
        self.downloaded_videos_dir.mkdir(exist_ok=True)
        self.exported_videos_dir.mkdir(exist_ok=True)

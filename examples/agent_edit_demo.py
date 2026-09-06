from __future__ import annotations

import argparse
from pathlib import Path

from videopython.editing import VideoEdit

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "src/tests/test_data/big_video.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the agent-authored edit plan used in the documentation.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/assets/agent-edit-demo.mp4")
    args = parser.parse_args()

    edit = VideoEdit.from_dict(
        {
            "segments": [
                {
                    "source": str(args.source),
                    "start": 2.0,
                    "end": 10.0,
                    "operations": [
                        {"op": "crop", "width": 400, "height": 500},
                        {"op": "resize", "width": 512, "height": 640},
                        {"op": "color_adjust", "contrast": 1.08, "saturation": 1.15},
                        {"op": "punch_in", "zoom_factor": 1.04, "attack_frames": 12, "release_frames": 12},
                        {"op": "fade", "mode": "in_out", "duration": 0.35},
                    ],
                }
            ]
        }
    )
    edit.validate()
    edit.run_to_file(args.output)


if __name__ == "__main__":
    main()

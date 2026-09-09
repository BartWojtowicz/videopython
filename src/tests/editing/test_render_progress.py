import subprocess

import numpy as np
import pytest

from tests.test_config import SMALL_VIDEO_PATH
from videopython import _ffmpeg
from videopython.base.video import Video
from videopython.editing import VideoEdit


@pytest.mark.parametrize("framewise", [False, True])
def test_render_progress_preserves_output(tmp_path, framewise):
    operations = [{"op": "film_grain", "intensity": 0.05}] if framewise else []
    edit = VideoEdit.from_dict(
        {"segments": [{"source": SMALL_VIDEO_PATH, "start": 0, "end": 1, "operations": operations}]}
    )
    events = []
    result = edit.run_to_file(tmp_path / "progress.mp4", on_progress=events.append)
    plain = edit.run_to_file(tmp_path / "plain.mp4")
    np.testing.assert_array_equal(Video.from_path(result).frames, Video.from_path(plain).frames)
    assert events[0].stage == "compilation"
    assert events[-1].stage == "complete" and events[-1].finished
    frames = [e for e in events if e.stage == "segment"]
    assert frames[0].completed == 0 and not frames[0].finished
    assert frames[-1].finished and frames[-1].completed > 0
    assert [e.completed for e in frames] == sorted(e.completed for e in frames)
    assert all(e.segment_index == 0 and e.unit == "frames" for e in frames)


def test_multistage_progress(tmp_path):
    bed = tmp_path / "bed.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", str(bed)], check=True
    )
    edit = VideoEdit.from_dict(
        {
            "segments": [
                {"source": SMALL_VIDEO_PATH, "start": 0, "end": 1},
                {"source": SMALL_VIDEO_PATH, "start": 1, "end": 2, "transition_in": {"type": "fade", "duration": 0.2}},
            ],
            "post_operations": [{"op": "color_adjust", "brightness": 0.05}],
        }
    )
    events = []
    edit.run_to_file(tmp_path / "out.mp4", on_progress=events.append)
    assert [e.stage for e in events if e.finished] == [
        "compilation",
        "segment",
        "segment",
        "assembly",
        "post_operations",
        "complete",
    ]
    single = VideoEdit.from_dict(
        {
            "segments": [{"source": SMALL_VIDEO_PATH, "start": 0, "end": 1}],
            "music_bed": {"source": str(bed), "gain": 0.1},
        }
    )
    events.clear()
    single.run_to_file(tmp_path / "bed.mp4", on_progress=events.append)
    assert [e.stage for e in events if e.finished] == ["compilation", "segment", "audio_mix", "complete"]


def test_ffmpeg_failure_does_not_report_success(tmp_path, monkeypatch):
    original = _ffmpeg.run_with_progress

    def fail(cmd, callback):
        original([*cmd[:-1], "-c:v", "missing_test_encoder", cmd[-1]], callback)

    monkeypatch.setattr(_ffmpeg, "run_with_progress", fail)
    edit = VideoEdit.from_dict({"segments": [{"source": SMALL_VIDEO_PATH, "start": 0, "end": 1}]})
    events = []
    target = tmp_path / "failed.mp4"
    with pytest.raises(_ffmpeg.FFmpegRunError):
        edit.run_to_file(target, on_progress=events.append)
    assert not target.exists()
    assert not any(e.stage == "complete" or (e.stage == "segment" and e.finished) for e in events)


def test_callback_failure_reaps_ffmpeg(tmp_path, monkeypatch):
    processes = []
    original = subprocess.Popen

    def capture(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    def fail(_frame):
        raise RuntimeError("Callback failed")

    monkeypatch.setattr(subprocess, "Popen", capture)
    with pytest.raises(RuntimeError, match="Callback failed"):
        _ffmpeg.run_with_progress(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=640x360:duration=30",
                str(tmp_path / "interrupted.mp4"),
            ],
            fail,
        )
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdout.closed

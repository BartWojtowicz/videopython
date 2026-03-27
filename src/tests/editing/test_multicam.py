from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from videopython.base.transitions import FadeTransition, InstantTransition
from videopython.editing.multicam import CutPoint, MultiCamEdit

# ---------------------------------------------------------------------------
# Fixtures: generate short test videos with ffmpeg
# ---------------------------------------------------------------------------

FPS = 25
DURATION = 10  # seconds
WIDTH, HEIGHT = 320, 240


@pytest.fixture(scope="module")
def test_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_color_video(path: Path, color: str, duration: int = DURATION) -> Path:
    """Generate a solid-color video with silent audio using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={WIDTH}x{HEIGHT}:r={FPS}:d={duration}",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=44100:cl=stereo",
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _make_audio(path: Path, duration: int = DURATION) -> Path:
    """Generate a sine wave audio file."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def cam_paths(test_dir):
    """Create 3 camera source videos (red, green, blue)."""
    return {
        "cam1": _make_color_video(test_dir / "cam1.mp4", "red"),
        "cam2": _make_color_video(test_dir / "cam2.mp4", "green"),
        "cam3": _make_color_video(test_dir / "cam3.mp4", "blue"),
    }


@pytest.fixture(scope="module")
def audio_path(test_dir):
    return _make_audio(test_dir / "audio.aac")


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_sources(self, cam_paths):
        with pytest.raises(ValueError, match="at least one source"):
            MultiCamEdit(sources={}, cuts=[CutPoint(time=0.0, camera="cam1")])

    def test_empty_cuts(self, cam_paths):
        with pytest.raises(ValueError, match="at least one cut"):
            MultiCamEdit(sources=cam_paths, cuts=[])

    def test_first_cut_not_at_zero(self, cam_paths):
        with pytest.raises(ValueError, match="First cut must start at time 0.0"):
            MultiCamEdit(sources=cam_paths, cuts=[CutPoint(time=5.0, camera="cam1")])

    def test_cuts_not_ascending(self, cam_paths):
        with pytest.raises(ValueError, match="strictly ascending"):
            MultiCamEdit(
                sources=cam_paths,
                cuts=[
                    CutPoint(time=0.0, camera="cam1"),
                    CutPoint(time=5.0, camera="cam2"),
                    CutPoint(time=3.0, camera="cam3"),
                ],
            )

    def test_unknown_camera(self, cam_paths):
        with pytest.raises(ValueError, match="unknown camera 'nonexistent'"):
            MultiCamEdit(
                sources=cam_paths,
                cuts=[CutPoint(time=0.0, camera="nonexistent")],
            )

    def test_missing_source_file(self, test_dir):
        with pytest.raises(FileNotFoundError, match="not found"):
            MultiCamEdit(
                sources={"cam1": test_dir / "does_not_exist.mp4"},
                cuts=[CutPoint(time=0.0, camera="cam1")],
            )

    def test_missing_audio_file(self, cam_paths, test_dir):
        with pytest.raises(FileNotFoundError, match="Audio source not found"):
            MultiCamEdit(
                sources=cam_paths,
                cuts=[CutPoint(time=0.0, camera="cam1")],
                audio_source=test_dir / "missing_audio.aac",
            )

    def test_cut_exceeds_duration(self, cam_paths):
        with pytest.raises(ValueError, match="exceeds source duration"):
            MultiCamEdit(
                sources=cam_paths,
                cuts=[
                    CutPoint(time=0.0, camera="cam1"),
                    CutPoint(time=999.0, camera="cam2"),
                ],
            )

    def test_unknown_offset_key(self, cam_paths):
        with pytest.raises(ValueError, match="source_offsets references unknown source 'nonexistent'"):
            MultiCamEdit(
                sources=cam_paths,
                cuts=[CutPoint(time=0.0, camera="cam1")],
                source_offsets={"nonexistent": 1.0},
            )

    def test_negative_adjusted_seek(self, cam_paths):
        """Offset larger than cut time results in negative seek."""
        with pytest.raises(ValueError, match="negative seek position"):
            MultiCamEdit(
                sources=cam_paths,
                cuts=[
                    CutPoint(time=0.0, camera="cam1"),
                    CutPoint(time=2.0, camera="cam2"),
                ],
                source_offsets={"cam2": 5.0},
            )

    def test_adjusted_end_exceeds_duration(self, cam_paths):
        """Negative offset pushes adjusted end past source duration."""
        with pytest.raises(ValueError, match="exceeds source duration"):
            MultiCamEdit(
                sources=cam_paths,
                cuts=[CutPoint(time=0.0, camera="cam1")],
                source_offsets={"cam1": -3.0},
            )


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_single_cut_full_video(self, cam_paths):
        """Single cut at 0 = use one camera for the whole duration."""
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        result = edit.run()
        assert result.fps == FPS
        assert result.frames.shape[1:3] == (HEIGHT, WIDTH)
        # Should be roughly DURATION seconds
        assert abs(result.total_seconds - DURATION) < 0.5

    def test_hard_cuts(self, cam_paths):
        """Multiple hard cuts between cameras."""
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=3.0, camera="cam2"),
                CutPoint(time=7.0, camera="cam3"),
            ],
        )
        result = edit.run()
        assert abs(result.total_seconds - DURATION) < 0.5

    def test_fade_transitions(self, cam_paths):
        """Fade transitions between cameras."""
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=3.0, camera="cam2", transition=FadeTransition(0.5)),
                CutPoint(time=7.0, camera="cam3", transition=FadeTransition(0.5)),
            ],
        )
        result = edit.run()
        # Fade transitions consume some frames, so result is slightly shorter
        assert result.total_seconds < DURATION
        assert result.total_seconds > DURATION - 2  # at most 1s lost from 2 fades

    def test_default_transition(self, cam_paths):
        """Default transition applies when cut has no explicit transition."""
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=5.0, camera="cam2"),
            ],
            default_transition=FadeTransition(0.5),
        )
        result = edit.run()
        assert result.total_seconds < DURATION

    def test_audio_replacement(self, cam_paths, audio_path):
        """External audio replaces camera audio."""
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
            audio_source=audio_path,
        )
        result = edit.run()
        assert not result.audio.is_silent

    def test_silence_when_no_audio(self, cam_paths):
        """No audio source means silent output."""
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        result = edit.run()
        assert result.audio.is_silent

    def test_source_offsets(self, cam_paths):
        """Source offsets adjust seek positions correctly."""
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=5.0, camera="cam2"),
            ],
            source_offsets={"cam2": 3.0},
        )
        result = edit.run()
        # cam1: 0-5s from file, cam2: timeline 5-10 -> file 2-7s
        # Total output should still be ~10s
        assert abs(result.total_seconds - DURATION) < 0.5

    def test_source_offsets_no_offset_is_default(self, cam_paths):
        """Omitting source_offsets gives same result as empty dict."""
        cuts = [
            CutPoint(time=0.0, camera="cam1"),
            CutPoint(time=5.0, camera="cam2"),
        ]
        edit_no_offsets = MultiCamEdit(sources=cam_paths, cuts=cuts)
        edit_empty = MultiCamEdit(sources=cam_paths, cuts=cuts, source_offsets={})
        r1 = edit_no_offsets.run()
        r2 = edit_empty.run()
        assert abs(r1.total_seconds - r2.total_seconds) < 0.1


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip(self, cam_paths, audio_path):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=3.0, camera="cam2", transition=FadeTransition(0.5)),
                CutPoint(time=7.0, camera="cam3"),
            ],
            audio_source=audio_path,
            default_transition=InstantTransition(),
        )
        data = edit.to_dict()
        restored = MultiCamEdit.from_dict(data)

        assert restored.to_dict() == data

    def test_roundtrip_with_offsets(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=5.0, camera="cam2"),
            ],
            source_offsets={"cam2": 3.0},
        )
        data = edit.to_dict()
        assert data["source_offsets"] == {"cam2": 3.0}
        restored = MultiCamEdit.from_dict(data)
        assert restored.source_offsets == {"cam2": 3.0}
        assert restored.to_dict() == data

    def test_roundtrip_without_offsets(self, cam_paths):
        """source_offsets is omitted from dict when empty."""
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        data = edit.to_dict()
        assert "source_offsets" not in data

    def test_from_json(self, cam_paths):
        data = {
            "sources": {k: str(v) for k, v in cam_paths.items()},
            "cuts": [
                {"time": 0.0, "camera": "cam1"},
                {"time": 5.0, "camera": "cam2"},
            ],
        }
        edit = MultiCamEdit.from_json(json.dumps(data))
        assert len(edit.cuts) == 2
        assert isinstance(edit.default_transition, InstantTransition)

    def test_from_json_invalid(self):
        with pytest.raises(ValueError, match="Invalid MultiCamEdit JSON"):
            MultiCamEdit.from_json("not json")

    def test_from_dict_missing_sources(self):
        with pytest.raises(ValueError, match="non-empty 'sources'"):
            MultiCamEdit.from_dict({"cuts": [{"time": 0, "camera": "x"}]})

    def test_from_dict_missing_cuts(self, cam_paths):
        with pytest.raises(ValueError, match="non-empty 'cuts'"):
            MultiCamEdit.from_dict({"sources": {k: str(v) for k, v in cam_paths.items()}})


# ---------------------------------------------------------------------------
# Validate tests
# ---------------------------------------------------------------------------


class TestValidate:
    def test_validate_single_cut(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        meta = edit.validate()
        assert meta.fps == FPS
        assert meta.width == WIDTH
        assert meta.height == HEIGHT
        assert abs(meta.total_seconds - DURATION) < 0.5

    def test_validate_hard_cuts_preserves_duration(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=3.0, camera="cam2"),
                CutPoint(time=7.0, camera="cam3"),
            ],
        )
        meta = edit.validate()
        assert abs(meta.total_seconds - DURATION) < 0.5

    def test_validate_fade_subtracts_overlap(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=3.0, camera="cam2", transition=FadeTransition(0.5)),
                CutPoint(time=7.0, camera="cam3", transition=FadeTransition(0.5)),
            ],
        )
        meta = edit.validate()
        # 2 fade transitions at 0.5s each = 1.0s subtracted
        assert meta.total_seconds < DURATION
        assert meta.total_seconds > DURATION - 2

    def test_validate_matches_run(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=3.0, camera="cam2", transition=FadeTransition(0.5)),
                CutPoint(time=7.0, camera="cam3"),
            ],
        )
        predicted = edit.validate()
        actual = edit.run()
        assert predicted.fps == actual.fps
        assert predicted.width == actual.metadata.width
        assert predicted.height == actual.metadata.height
        assert abs(predicted.total_seconds - actual.total_seconds) < 0.5

    def test_validate_default_fade_transition(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=5.0, camera="cam2"),
            ],
            default_transition=FadeTransition(0.5),
        )
        meta = edit.validate()
        # 1 fade at 0.5s
        assert meta.total_seconds < DURATION


# ---------------------------------------------------------------------------
# JSON Schema tests
# ---------------------------------------------------------------------------


class TestJsonSchema:
    def test_schema_structure(self):
        schema = MultiCamEdit.json_schema()
        assert schema["type"] == "object"
        assert "sources" in schema["properties"]
        assert "cuts" in schema["properties"]
        assert schema["required"] == ["sources", "cuts"]

    def test_schema_has_source_offsets(self):
        schema = MultiCamEdit.json_schema()
        offsets_schema = schema["properties"]["source_offsets"]
        assert offsets_schema["type"] == "object"
        assert offsets_schema["additionalProperties"] == {"type": "number"}

    def test_schema_has_transition_options(self):
        schema = MultiCamEdit.json_schema()
        cut_props = schema["properties"]["cuts"]["items"]["properties"]
        transition_types = {s["properties"]["type"]["const"] for s in cut_props["transition"]["oneOf"]}
        assert transition_types == {"instant", "fade", "blur"}

    def test_schema_plan_roundtrip(self, cam_paths):
        plan = {
            "sources": {k: str(v) for k, v in cam_paths.items()},
            "cuts": [
                {"time": 0.0, "camera": "cam1"},
                {"time": 5.0, "camera": "cam2", "transition": {"type": "fade", "effect_time_seconds": 0.5}},
            ],
            "default_transition": {"type": "instant"},
        }
        edit = MultiCamEdit.from_dict(plan)
        assert len(edit.cuts) == 2
        assert isinstance(edit.cuts[1].transition, FadeTransition)

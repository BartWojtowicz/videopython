from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from xml.etree.ElementTree import fromstring

import pytest

from videopython.base.transitions import FadeTransition, InstantTransition
from videopython.editing.multicam import CutPoint, MultiCamEdit
from videopython.editing.premiere_xml import (
    _fps_to_rate_info as fps_to_rate_info,
)
from videopython.editing.premiere_xml import (
    _seconds_to_frames as seconds_to_frames,
)
from videopython.editing.premiere_xml import (
    to_premiere_xml,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FPS = 25
DURATION = 10
WIDTH, HEIGHT = 320, 240


@pytest.fixture(scope="module")
def test_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_color_video(path: Path, color: str, duration: int = DURATION) -> Path:
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
    return {
        "cam1": _make_color_video(test_dir / "cam1.mp4", "red"),
        "cam2": _make_color_video(test_dir / "cam2.mp4", "green"),
        "cam3": _make_color_video(test_dir / "cam3.mp4", "blue"),
    }


@pytest.fixture(scope="module")
def audio_path(test_dir):
    return _make_audio(test_dir / "audio.aac")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse(xml_str: str):
    return fromstring(xml_str)


# ---------------------------------------------------------------------------
# Rate and frame conversion
# ---------------------------------------------------------------------------


class TestRateConversion:
    def test_25fps(self):
        assert fps_to_rate_info(25) == (25, False)

    def test_24fps(self):
        assert fps_to_rate_info(24) == (24, False)

    def test_30fps(self):
        assert fps_to_rate_info(30) == (30, False)

    def test_2997fps(self):
        assert fps_to_rate_info(29.97) == (30, True)

    def test_2397fps(self):
        assert fps_to_rate_info(23.976) == (24, True)

    def test_5994fps(self):
        assert fps_to_rate_info(59.94) == (60, True)


class TestFrameConversion:
    def test_zero(self):
        assert seconds_to_frames(0.0, 25) == 0

    def test_whole_seconds(self):
        assert seconds_to_frames(5.0, 25) == 125

    def test_fractional(self):
        assert seconds_to_frames(0.5, 25) == 12 or seconds_to_frames(0.5, 25) == 13

    def test_ntsc(self):
        frames = seconds_to_frames(1.0, 29.97)
        assert frames == 30


# ---------------------------------------------------------------------------
# XML structure
# ---------------------------------------------------------------------------


class TestXmlStructure:
    def test_root_element(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        root = _parse(to_premiere_xml(edit))
        assert root.tag == "xmeml"
        assert root.get("version") == "5"

    def test_has_sequence(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        root = _parse(to_premiere_xml(edit))
        seq = root.find("sequence")
        assert seq is not None
        assert seq.find("name").text == "MultiCamEdit"

    def test_has_video_track(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        root = _parse(to_premiere_xml(edit))
        track = root.find("sequence/media/video/track")
        assert track is not None

    def test_has_format_samplecharacteristics(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        root = _parse(to_premiere_xml(edit))
        sc = root.find("sequence/media/video/format/samplecharacteristics")
        assert sc is not None
        assert sc.find("width").text == str(WIDTH)
        assert sc.find("height").text == str(HEIGHT)

    def test_has_doctype(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        xml_str = to_premiere_xml(edit)
        assert "<!DOCTYPE xmeml>" in xml_str

    def test_starts_with_xml_declaration(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        xml_str = to_premiere_xml(edit)
        assert xml_str.startswith("<?xml")


# ---------------------------------------------------------------------------
# Flat clipitems (no multiclip)
# ---------------------------------------------------------------------------


class TestFlatClipitems:
    def test_no_multiclip_elements(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1"), CutPoint(time=5.0, camera="cam2")],
        )
        root = _parse(to_premiere_xml(edit))
        assert root.find(".//multiclip") is None

    def test_clipitem_references_file_directly(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        root = _parse(to_premiere_xml(edit))
        ci = root.find("sequence/media/video/track/clipitem")
        f = ci.find("file")
        assert f is not None
        assert f.get("id") == "file-cam1"

    def test_clipitem_named_after_camera(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam2")],
        )
        root = _parse(to_premiere_xml(edit))
        ci = root.find("sequence/media/video/track/clipitem")
        assert ci.find("name").text == "cam2"


# ---------------------------------------------------------------------------
# Clipitem timing
# ---------------------------------------------------------------------------


class TestClipitemTiming:
    def test_single_cut_full_duration(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        root = _parse(to_premiere_xml(edit))
        ci = root.find("sequence/media/video/track/clipitem")
        total_frames = seconds_to_frames(DURATION, FPS)
        assert ci.find("in").text == "0"
        assert ci.find("out").text == str(total_frames)
        assert ci.find("start").text == "0"
        assert ci.find("end").text == str(total_frames)

    def test_two_cuts_timing(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=3.0, camera="cam2"),
            ],
        )
        root = _parse(to_premiere_xml(edit))
        clipitems = root.findall("sequence/media/video/track/clipitem")

        assert clipitems[0].find("start").text == "0"
        assert clipitems[0].find("end").text == str(seconds_to_frames(3.0, FPS))

        assert clipitems[1].find("start").text == str(seconds_to_frames(3.0, FPS))
        assert clipitems[1].find("end").text == str(seconds_to_frames(DURATION, FPS))


# ---------------------------------------------------------------------------
# File deduplication
# ---------------------------------------------------------------------------


class TestFileDedup:
    def test_file_defined_once(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=5.0, camera="cam2"),
            ],
        )
        root = _parse(to_premiere_xml(edit))
        all_files = list(root.iter("file"))
        file_ids_with_pathurl = set()
        for f in all_files:
            if f.find("pathurl") is not None:
                assert f.get("id") not in file_ids_with_pathurl, f"File {f.get('id')} defined more than once"
                file_ids_with_pathurl.add(f.get("id"))

    def test_file_referenced_after_first(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=5.0, camera="cam1"),
            ],
        )
        root = _parse(to_premiere_xml(edit))
        file_cam1_elements = [f for f in root.iter("file") if f.get("id") == "file-cam1"]
        defined = [f for f in file_cam1_elements if f.find("pathurl") is not None]
        refs = [f for f in file_cam1_elements if f.find("pathurl") is None]
        assert len(defined) == 1
        assert len(refs) >= 1


# ---------------------------------------------------------------------------
# Source offsets
# ---------------------------------------------------------------------------


class TestSourceOffsets:
    def test_offset_adjusts_in_out(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam2"),
                CutPoint(time=5.0, camera="cam1"),
            ],
            source_offsets={"cam2": -2.0},
        )
        root = _parse(to_premiere_xml(edit))
        ci = root.find("sequence/media/video/track/clipitem")
        # cam2 offset -2.0: timeline 0-5s, source (0-(-2))-(5-(-2)) = 2-7s
        assert ci.find("in").text == str(seconds_to_frames(2.0, FPS))
        assert ci.find("out").text == str(seconds_to_frames(7.0, FPS))


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


class TestAudio:
    def test_audio_tracks_present(self, cam_paths, audio_path):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
            audio_source=audio_path,
        )
        root = _parse(to_premiere_xml(edit))
        audio_tracks = root.findall("sequence/media/audio/track")
        assert len(audio_tracks) == 2

    def test_audio_single_clipitem_per_track(self, cam_paths, audio_path):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=5.0, camera="cam2"),
            ],
            audio_source=audio_path,
        )
        root = _parse(to_premiere_xml(edit))
        for track in root.findall("sequence/media/audio/track"):
            clipitems = track.findall("clipitem")
            assert len(clipitems) == 1

    def test_audio_clipitems_reference_audio_file(self, cam_paths, audio_path):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
            audio_source=audio_path,
        )
        root = _parse(to_premiere_xml(edit))
        for track in root.findall("sequence/media/audio/track"):
            for ci in track.findall("clipitem"):
                f = ci.find("file")
                assert f.get("id") == "file-audio"

    def test_no_audio_tracks_without_audio_source(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[CutPoint(time=0.0, camera="cam1")],
        )
        root = _parse(to_premiere_xml(edit))
        assert root.find("sequence/media/audio") is None


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


class TestTransitions:
    def test_fade_creates_video_transition(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=5.0, camera="cam2", transition=FadeTransition(0.5)),
            ],
        )
        root = _parse(to_premiere_xml(edit))
        transitions = root.findall("sequence/media/video/track/transitionitem")
        assert len(transitions) == 1
        assert transitions[0].find("effect/name").text == "Cross Dissolve"

    def test_instant_no_transition(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=5.0, camera="cam2", transition=InstantTransition()),
            ],
        )
        root = _parse(to_premiere_xml(edit))
        transitions = root.findall("sequence/media/video/track/transitionitem")
        assert len(transitions) == 0

    def test_default_fade_transition(self, cam_paths):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=3.0, camera="cam2"),
                CutPoint(time=7.0, camera="cam3"),
            ],
            default_transition=FadeTransition(0.5),
        )
        root = _parse(to_premiere_xml(edit))
        transitions = root.findall("sequence/media/video/track/transitionitem")
        assert len(transitions) == 2


# ---------------------------------------------------------------------------
# Well-formed XML round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_output_is_valid_xml(self, cam_paths, audio_path):
        edit = MultiCamEdit(
            sources=cam_paths,
            cuts=[
                CutPoint(time=0.0, camera="cam1"),
                CutPoint(time=3.0, camera="cam2", transition=FadeTransition(0.5)),
                CutPoint(time=7.0, camera="cam3"),
            ],
            audio_source=audio_path,
        )
        xml_str = to_premiere_xml(edit)
        root = fromstring(xml_str)
        assert root.tag == "xmeml"

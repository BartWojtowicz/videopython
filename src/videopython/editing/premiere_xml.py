"""Export MultiCamEdit plans to FCP7 XML (xmeml) for Adobe Premiere Pro.

Generates xmeml version 5 with a flat sequence of clipitems on one video
track and stereo audio tracks. Each cut becomes a clipitem that directly
references its source file.

Known limitations:
- BlurTransition has no xmeml equivalent and is exported as a hard cut.
- Source file paths are absolute file://localhost/ URLs; the xmeml is not
  portable across machines without relinking media in Premiere.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote
from xml.dom.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

if TYPE_CHECKING:
    from videopython.editing.multicam import MultiCamEdit

__all__ = ["to_premiere_xml"]

_XMEML_VERSION = "5"

# Integer timebase for common NTSC rates (xmeml uses <timebase>/<ntsc>).
_NTSC_TIMEBASES: dict[float, int] = {
    23.976: 24,
    23.98: 24,
    29.97: 30,
    59.94: 60,
}


# ---------------------------------------------------------------------------
# Time and path helpers
# ---------------------------------------------------------------------------


def _fps_to_rate_info(fps: float) -> tuple[int, bool]:
    """Return ``(timebase, ntsc)`` for a frame rate."""
    rounded = round(fps, 3)
    if rounded in _NTSC_TIMEBASES:
        return _NTSC_TIMEBASES[rounded], True
    return round(fps), False


def _seconds_to_frames(seconds: float, fps: float) -> int:
    """Convert seconds to an integer frame count."""
    timebase, ntsc = _fps_to_rate_info(fps)
    actual_fps = timebase / 1.001 if ntsc else timebase
    return round(seconds * actual_fps)


def _path_to_url(path: Path) -> str:
    """Convert a local path to a ``file://localhost/...`` URL."""
    absolute = str(path.resolve())
    if not absolute.startswith("/"):
        absolute = "/" + absolute
    return "file://localhost" + quote(absolute)


def _serialize(root: Element, doctype: str) -> str:
    """Pretty-print an ElementTree *root* with XML declaration and DOCTYPE."""
    rough = tostring(root, encoding="unicode", xml_declaration=False)
    dom = parseString(rough)
    lines = dom.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8").splitlines(True)
    return lines[0] + doctype + "\n" + "".join(lines[1:])


# ---------------------------------------------------------------------------
# Element builders
# ---------------------------------------------------------------------------


def _add_rate(parent: Element, fps: float) -> Element:
    timebase, ntsc = _fps_to_rate_info(fps)
    rate = SubElement(parent, "rate")
    SubElement(rate, "timebase").text = str(timebase)
    SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"
    return rate


def _add_text(parent: Element, tag: str, text: str) -> Element:
    el = SubElement(parent, tag)
    el.text = text
    return el


def _build_file_element(
    parent: Element,
    file_id: str,
    name: str,
    path: Path,
    fps: float,
    duration_frames: int,
    width: int | None = None,
    height: int | None = None,
    has_video: bool = True,
    has_audio: bool = True,
) -> Element:
    """Build a full ``<file>`` definition (first occurrence)."""
    f = SubElement(parent, "file", id=file_id)
    _add_text(f, "name", name)
    _add_text(f, "pathurl", _path_to_url(path))
    _add_rate(f, fps)
    _add_text(f, "duration", str(duration_frames))

    tc = SubElement(f, "timecode")
    _add_text(tc, "string", "00:00:00:00")
    _add_text(tc, "frame", "0")
    _add_text(tc, "displayformat", "NDF")
    _add_rate(tc, fps)

    media = SubElement(f, "media")
    if has_video:
        vid = SubElement(media, "video")
        _add_text(vid, "duration", str(duration_frames))
        sc = SubElement(vid, "samplecharacteristics")
        _add_text(sc, "width", str(width))
        _add_text(sc, "height", str(height))
    if has_audio:
        aud = SubElement(media, "audio")
        asc = SubElement(aud, "samplecharacteristics")
        _add_text(asc, "depth", "16")
        _add_text(asc, "samplerate", "48000")
        _add_text(aud, "channelcount", "2")

    return f


def _build_video_transition(parent: Element, fps: float, start_frame: int, end_frame: int) -> Element:
    ti = SubElement(parent, "transitionitem")
    _add_rate(ti, fps)
    _add_text(ti, "start", str(start_frame))
    _add_text(ti, "end", str(end_frame))
    _add_text(ti, "alignment", "center")

    effect = SubElement(ti, "effect")
    _add_text(effect, "name", "Cross Dissolve")
    _add_text(effect, "effectid", "Cross Dissolve")
    _add_text(effect, "effectcategory", "Dissolve")
    _add_text(effect, "effecttype", "transition")
    _add_text(effect, "mediatype", "video")
    _add_text(effect, "wipecode", "0")
    _add_text(effect, "wipeaccuracy", "100")
    _add_text(effect, "startratio", "0")
    _add_text(effect, "endratio", "1")
    _add_text(effect, "reverse", "FALSE")
    return ti


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def to_premiere_xml(edit: MultiCamEdit) -> str:
    """Export a *MultiCamEdit* plan to FCP7 XML (xmeml) for Premiere Pro.

    Generates a flat sequence with one video track containing clipitems
    that directly reference source files, plus stereo audio tracks for
    the external audio source.

    Args:
        edit: A validated ``MultiCamEdit`` instance.

    Returns:
        A UTF-8 XML string with ``<?xml ...?>`` declaration and
        ``<!DOCTYPE xmeml>``.
    """
    from videopython.base.transitions import FadeTransition

    meta = edit._source_meta
    fps = meta.fps
    source_duration = edit._source_duration
    total_frames = _seconds_to_frames(source_duration, fps)

    def frames(s: float) -> int:
        return _seconds_to_frames(s, fps)

    root = Element("xmeml", version=_XMEML_VERSION)

    # -- sequence --------------------------------------------------------------
    seq = SubElement(root, "sequence", id="MultiCamEdit")
    _add_text(seq, "name", "MultiCamEdit")
    _add_text(seq, "duration", str(total_frames))
    _add_rate(seq, fps)

    tc = SubElement(seq, "timecode")
    _add_text(tc, "string", "00:00:00:00")
    _add_text(tc, "frame", "0")
    _add_text(tc, "displayformat", "NDF")
    _add_rate(tc, fps)

    media = SubElement(seq, "media")

    # -- video -----------------------------------------------------------------
    video = SubElement(media, "video")
    fmt = SubElement(video, "format")
    sc = SubElement(fmt, "samplecharacteristics")
    _add_text(sc, "width", str(meta.width))
    _add_text(sc, "height", str(meta.height))
    _add_text(sc, "anamorphic", "FALSE")
    _add_text(sc, "pixelaspectratio", "square")
    _add_text(sc, "fielddominance", "none")
    _add_rate(sc, fps)
    _add_text(sc, "colordepth", "24")

    v_track = SubElement(video, "track")
    defined_file_ids: set[str] = set()
    cuts = edit.cuts

    for i, cut in enumerate(cuts):
        start_time = cut.time
        end_time = cuts[i + 1].time if i + 1 < len(cuts) else source_duration
        start_frame = frames(start_time)
        end_frame = frames(end_time)

        camera = cut.camera
        offset = edit.source_offsets.get(camera, 0.0)
        in_frame = frames(start_time - offset)
        out_frame = frames(end_time - offset)

        # Transition
        if i > 0:
            transition = cut.transition or edit.default_transition
            effect_time = getattr(transition, "effect_time_seconds", 0.0)
            if isinstance(transition, FadeTransition) and effect_time > 0:
                t_half = effect_time / 2
                _build_video_transition(v_track, fps, frames(start_time - t_half), frames(start_time + t_half))

        ci = SubElement(v_track, "clipitem", id=f"clipitem-v-{i + 1}")
        _add_text(ci, "name", camera)
        _add_text(ci, "duration", str(total_frames))
        _add_rate(ci, fps)
        _add_text(ci, "in", str(in_frame))
        _add_text(ci, "out", str(out_frame))
        _add_text(ci, "start", str(start_frame))
        _add_text(ci, "end", str(end_frame))
        _add_text(ci, "enabled", "TRUE")

        file_id = f"file-{camera}"
        if file_id not in defined_file_ids:
            src_meta = edit._source_metas[camera]
            src_dur_frames = _seconds_to_frames(src_meta.total_seconds, fps)
            _build_file_element(
                ci,
                file_id,
                edit.sources[camera].name,
                edit.sources[camera],
                fps,
                src_dur_frames,
                width=src_meta.width,
                height=src_meta.height,
            )
            defined_file_ids.add(file_id)
        else:
            SubElement(ci, "file", id=file_id)

    # -- audio -----------------------------------------------------------------
    if edit.audio_source:
        audio = SubElement(media, "audio")
        afmt = SubElement(audio, "format")
        asc = SubElement(afmt, "samplecharacteristics")
        _add_text(asc, "depth", "16")
        _add_text(asc, "samplerate", "48000")

        outputs = SubElement(audio, "outputs")
        group = SubElement(outputs, "group")
        _add_text(group, "index", "1")
        _add_text(group, "numchannels", "2")
        _add_text(group, "downmix", "0")
        for ch in (1, 2):
            channel = SubElement(group, "channel")
            _add_text(channel, "index", str(ch))

        audio_file_id = "file-audio"

        for track_idx in (1, 2):
            a_track = SubElement(audio, "track")

            aci = SubElement(a_track, "clipitem", id=f"clipitem-a{track_idx}")
            _add_text(aci, "name", "Audio")
            _add_text(aci, "duration", str(total_frames))
            _add_rate(aci, fps)
            _add_text(aci, "in", "0")
            _add_text(aci, "out", str(total_frames))
            _add_text(aci, "start", "0")
            _add_text(aci, "end", str(total_frames))
            _add_text(aci, "enabled", "TRUE")

            if audio_file_id not in defined_file_ids:
                _build_file_element(
                    aci,
                    audio_file_id,
                    edit.audio_source.name,
                    edit.audio_source,
                    fps,
                    total_frames,
                    has_video=False,
                )
                defined_file_ids.add(audio_file_id)
            else:
                SubElement(aci, "file", id=audio_file_id)

            st = SubElement(aci, "sourcetrack")
            _add_text(st, "mediatype", "audio")
            _add_text(st, "trackindex", str(track_idx))

    return _serialize(root, "<!DOCTYPE xmeml>")

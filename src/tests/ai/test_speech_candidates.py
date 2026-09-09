import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from mcp.types import TextContent

from tests.ai.test_auto_edit import _analysis, _scene
from tests.test_config import SMALL_VIDEO_PATH
from videopython.ai.auto_edit import EditPlan, SpeechCandidateConfig, UnknownSceneIdsError, build_catalog, resolve_plan
from videopython.base import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import VideoMetadata
from videopython.mcp import server


def _transcription(items):
    return Transcription(words=[TranscriptionWord(word=text, start=start, end=end) for text, start, end in items])


def test_speech_candidates_ignore_visual_cuts_and_invalidate_changed_ids():
    transcription = _transcription([("First", 0.5, 1), ("sentence.", 1, 3), ("Second.", 3.5, 6), ("Third.", 6.5, 9)])
    config = SpeechCandidateConfig(min_duration=2, max_duration=5)
    single = _analysis("/one/talk.mp4", [_scene(0, 0, 10)], transcription=transcription)
    shots = _analysis("/one/talk.mp4", [_scene(0, 0, 1.5), _scene(1, 1.5, 10)], transcription=transcription)
    bundle = build_catalog([single], mode="speech", speech=config, keyframes=False)
    assert bundle.catalog == build_catalog([shots], mode="speech", speech=config, keyframes=False).catalog
    assert [(s.start, s.end) for s in bundle.catalog.scenes] == [(0.5, 3), (3.5, 6), (6.5, 9)]
    assert bundle.catalog == build_catalog([single], mode="speech", speech=config, keyframes=False).catalog
    other = single.model_copy(update={"source": single.source.model_copy(update={"path": "/two/talk.mp4"})})
    combined = build_catalog([single, other], mode="speech", speech=config, keyframes=False)
    assert len(combined.catalog.by_id()) == 6
    plan = EditPlan.model_validate({"segments": [{"scene_id": bundle.catalog.scenes[0].id}]})
    changed = build_catalog(
        [single], mode="speech", speech=SpeechCandidateConfig(min_duration=3, max_duration=6), keyframes=False
    )
    with pytest.raises(UnknownSceneIdsError):
        resolve_plan(plan, changed.catalog)
    with pytest.raises(UnknownSceneIdsError):
        resolve_plan(plan, build_catalog([single], keyframes=False).catalog)


def test_overlap_pause_and_unfinished_tail():
    transcription = _transcription(
        [("End.", 0, 1), ("interruption", 0.5, 2), ("ends.", 2, 3), ("Paused", 4, 5), ("unfinished", 7, 8)]
    )
    bundle = build_catalog(
        [_analysis("talk.mp4", [], transcription=transcription)],
        mode="speech",
        speech=SpeechCandidateConfig(min_duration=1, max_duration=3),
        keyframes=False,
    )
    assert [(s.start, s.end) for s in bundle.catalog.scenes] == [(0, 3), (4, 5)]
    for scene in bundle.catalog.scenes:
        assert all(
            not (w.start < boundary < w.end) for w in transcription.words for boundary in (scene.start, scene.end)
        )


def test_combines_short_sentences_and_skips_oversize_passages():
    transcription = _transcription([("One.", 0, 1), ("Two.", 1.1, 2.1), ("Too long.", 3, 10), ("Fits.", 11, 14)])
    bundle = build_catalog(
        [_analysis("talk.mp4", [], transcription=transcription)],
        mode="speech",
        speech=SpeechCandidateConfig(min_duration=2, max_duration=4),
        keyframes=False,
    )
    assert [(s.start, s.end) for s in bundle.catalog.scenes] == [(0, 2.1), (11, 14)]


def test_missing_alignment_and_impossible_duration_return_no_candidates():
    config = SpeechCandidateConfig(min_duration=2, max_duration=4)
    for transcription in [
        None,
        Transcription(segments=[TranscriptionSegment(start=0, end=3, text="No alignment.", words=[])]),
        _transcription([("Long.", 0, 10)]),
    ]:
        bundle = build_catalog(
            [_analysis("talk.mp4", [], transcription=transcription)], mode="speech", speech=config, keyframes=False
        )
        assert bundle.catalog.scenes == []


def test_speech_settings_are_explicit():
    with pytest.raises(ValueError, match="max_duration"):
        SpeechCandidateConfig(min_duration=5, max_duration=2)
    with pytest.raises(ValueError, match="speech settings"):
        build_catalog([], mode="speech")
    with pytest.raises(ValueError, match="speech settings"):
        build_catalog([], speech=SpeechCandidateConfig(min_duration=1, max_duration=2))


def test_mcp_speech_catalog_transcripts_and_render(tmp_path, monkeypatch):
    text = "A long sentence with readable source words " * 10 + "."
    tokens = text.split()
    transcription = _transcription(
        [(token, 0.5 + i * 2.5 / len(tokens), 0.5 + (i + 1) * 2.5 / len(tokens)) for i, token in enumerate(tokens)]
        + [("Another", 3.5, 4), ("sentence.", 4, 6)]
    )
    analysis = _analysis(SMALL_VIDEO_PATH, [_scene(0, 0, 12)], transcription=transcription)
    monkeypatch.setattr(server, "_analyses", {SMALL_VIDEO_PATH: analysis})
    monkeypatch.setattr(server, "_bundle", None)
    blocks = server.build_catalog(mode="speech", speech=SpeechCandidateConfig(min_duration=2, max_duration=4))
    catalog = json.loads(blocks[0].text)
    assert len(catalog["scenes"]) == 2
    first = catalog["scenes"][0]
    assert len(first["transcript"]) <= 280
    [full] = server.scene_transcripts([first["id"], first["id"]])
    assert json.loads(full.text) == {first["id"]: text}
    [error] = server.scene_transcripts(["unknown"])
    assert json.loads(error.text)["code"] == "unknown_scene_ids"
    plan = {"segments": [{"scene_id": first["id"]}]}
    assert server.validate_edit(plan).valid
    result = asyncio.run(server.run_edit(plan, str(tmp_path / "speech.mp4"), AsyncMock()))
    assert result.output_path is not None
    assert VideoMetadata.from_path(result.output_path).total_seconds == pytest.approx(2.5, abs=0.05)
    empty = server.build_catalog(mode="speech", speech=SpeechCandidateConfig(min_duration=20, max_duration=30))
    assert isinstance(empty[-1], TextContent)
    assert "No complete" in empty[-1].text


def test_zero_duration_words_keep_one_owner():
    transcription = _transcription([("First", 0, 1), ("sentence.", 1, 1), ("Second.", 1, 2)])
    bundle = build_catalog(
        [_analysis("talk.mp4", [], transcription=transcription)],
        mode="speech",
        speech=SpeechCandidateConfig(min_duration=1, max_duration=2),
        keyframes=False,
    )
    assert list(bundle.transcripts.values()) == ["First sentence.", "Second."]
    assert [(s.start, s.end) for s in bundle.catalog.scenes] == [(0, 1), (1, 2)]

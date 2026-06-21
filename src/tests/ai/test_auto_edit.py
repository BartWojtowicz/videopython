"""Tests for the auto_edit core primitives (catalog, resolve, schema, editor).

All deterministic and CI-safe: no model downloads. The planner is a stub, and
the only real I/O is keyframe extraction / metadata from the small test video.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.test_config import BIG_VIDEO_PATH, SMALL_VIDEO_PATH
from videopython.ai.auto_edit import (
    AutoEditError,
    AutoEditor,
    CatalogScene,
    EditCatalog,
    EditPlan,
    PlannerError,
    PlanSegment,
    UnknownSceneIdsError,
    build_catalog,
    resolve_plan,
)
from videopython.ai.video_analysis.models import (
    AnalysisRunInfo,
    AudioAnalysisSection,
    SceneAnalysisSample,
    SceneAnalysisSection,
    VideoAnalysis,
    VideoAnalysisConfig,
    VideoAnalysisSource,
)
from videopython.base.description import SceneDescription
from videopython.base.transcription import Transcription, TranscriptionSegment, TranscriptionWord
from videopython.base.video import VideoMetadata

# --------------------------------------------------------------------------- builders


def _scene(
    index: int,
    start: float,
    end: float,
    *,
    caption: str = "",
    shot_type: str | None = None,
    faces: list[Any] | None = None,
) -> SceneAnalysisSample:
    desc = SceneDescription(caption=caption, subjects=[], shot_type=shot_type) if (caption or shot_type) else None
    return SceneAnalysisSample(
        scene_index=index,
        start_second=start,
        end_second=end,
        scene_description=desc,
        faces=faces,
    )


def _analysis(
    path: str | None,
    scenes: list[SceneAnalysisSample],
    *,
    transcription: Transcription | None = None,
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    frame_count: int = 300,
    duration: float = 10.0,
) -> VideoAnalysis:
    return VideoAnalysis(
        source=VideoAnalysisSource(
            path=path, fps=fps, width=width, height=height, frame_count=frame_count, duration=duration
        ),
        config=VideoAnalysisConfig(),
        run_info=AnalysisRunInfo(created_at="2026-01-01T00:00:00Z", mode="path"),
        scenes=SceneAnalysisSection(samples=scenes),
        audio=AudioAnalysisSection(transcription=transcription) if transcription is not None else None,
    )


def _real_video_analysis() -> tuple[VideoAnalysis, str]:
    """An analysis over the real small test video, with two scenes covering it."""
    meta = VideoMetadata.from_path(SMALL_VIDEO_PATH)
    dur = meta.total_seconds
    scenes = [
        _scene(0, 0.0, dur / 2, caption="opening shot", shot_type="wide"),
        _scene(1, dur / 2, dur, caption="closing shot", shot_type="close-up"),
    ]
    analysis = _analysis(
        str(SMALL_VIDEO_PATH),
        scenes,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        frame_count=meta.frame_count,
        duration=dur,
    )
    return analysis, Path(SMALL_VIDEO_PATH).stem


class _StubPlanner:
    """Returns a canned plan dict and records every call."""

    def __init__(self, plan_dict: dict[str, Any]) -> None:
        self.plan_dict = plan_dict
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, *, system: str, parts: list[Any], schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"system": system, "parts": parts, "schema": schema})
        return json.loads(json.dumps(self.plan_dict))  # deep copy


# --------------------------------------------------------------------------- catalog


def test_build_catalog_projection() -> None:
    analysis = _analysis(
        "/clips/vidA.mp4",
        [
            _scene(0, 0.0, 2.5, caption="a person waves", shot_type="medium"),
            _scene(1, 2.5, 6.0, caption="a wide vista"),
        ],
    )
    bundle = build_catalog([analysis], keyframes=False)

    assert [s.id for s in bundle.catalog.scenes] == ["vidA#0", "vidA#1"]
    first = bundle.catalog.scenes[0]
    assert first.source == Path("/clips/vidA.mp4")
    assert (first.start, first.end, first.duration) == (0.0, 2.5, 2.5)
    assert first.shot_type == "medium"
    assert first.caption == "a person waves"
    assert first.has_faces is False
    assert bundle.keyframes == {}


def test_catalog_unique_ids_across_sources_with_same_stem() -> None:
    a = _analysis("/a/clip.mp4", [_scene(0, 0.0, 1.0, caption="x")])
    b = _analysis("/b/clip.mp4", [_scene(0, 0.0, 1.0, caption="y")])
    bundle = build_catalog([a, b], keyframes=False)
    ids = [s.id for s in bundle.catalog.scenes]
    assert ids == ["clip#0", "clip#0-2"]
    assert len(set(ids)) == 2


def test_catalog_transcript_excerpt() -> None:
    tx = Transcription(
        segments=[
            TranscriptionSegment(
                start=0.5,
                end=1.5,
                text="hello world",
                words=[
                    TranscriptionWord(start=0.5, end=1.0, word="hello"),
                    TranscriptionWord(start=1.0, end=1.5, word="world"),
                ],
            )
        ]
    )
    analysis = _analysis("vid.mp4", [_scene(0, 0.0, 5.0, caption="c")], transcription=tx)
    scene = build_catalog([analysis], keyframes=False).catalog.scenes[0]
    assert "hello" in scene.transcript
    assert scene.has_speech is True


def test_build_catalog_extracts_keyframes() -> None:
    analysis, stem = _real_video_analysis()
    bundle = build_catalog([analysis], keyframes=True)
    assert set(bundle.keyframes) == {f"{stem}#0", f"{stem}#1"}
    for frame in bundle.keyframes.values():
        assert frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] == 3


def test_build_catalog_requires_path_for_keyframes() -> None:
    analysis = _analysis(None, [_scene(0, 0.0, 2.0, caption="x")])
    with pytest.raises(ValueError, match="keyframe"):
        build_catalog([analysis], keyframes=True)


# --------------------------------------------------------------------------- resolve


def test_resolve_plan_maps_ids_to_exact_bounds() -> None:
    catalog = EditCatalog(
        scenes=[
            CatalogScene(id="s0", source=Path("a.mp4"), start=0.0, end=2.0, duration=2.0),
            CatalogScene(id="s1", source=Path("b.mp4"), start=1.0, end=3.0, duration=2.0),
        ]
    )
    plan = EditPlan(
        segments=[
            PlanSegment(scene_id="s1"),
            PlanSegment(scene_id="s0", operations=[{"op": "resize", "width": 640, "height": 480}]),
        ]
    )
    edit = resolve_plan(plan, catalog)

    assert [s.source for s in edit.segments] == [Path("b.mp4"), Path("a.mp4")]
    assert (edit.segments[0].start, edit.segments[0].end) == (1.0, 3.0)
    assert edit.segments[1].operations[0].op == "resize"


def test_resolve_plan_unknown_ids_raise() -> None:
    catalog = EditCatalog(scenes=[CatalogScene(id="s0", source=Path("a.mp4"), start=0.0, end=2.0, duration=2.0)])
    plan = EditPlan(segments=[PlanSegment(scene_id="missing")])
    with pytest.raises(UnknownSceneIdsError) as exc:
        resolve_plan(plan, catalog)
    assert exc.value.ids == ["missing"]


# --------------------------------------------------------------------------- schema


def test_editplan_schema_nonstrict() -> None:
    schema = EditPlan.json_schema()
    assert schema["type"] == "object"
    assert "segments" in schema["properties"]
    seg_items = schema["properties"]["segments"]["items"]
    assert seg_items["properties"]["scene_id"]["type"] == "string"
    assert "operations" in seg_items["properties"]


def test_editplan_schema_strict_is_closed_grammar() -> None:
    strict = EditPlan.json_schema(strict=True)
    assert strict["additionalProperties"] is False
    assert "$defs" in strict
    # The full operation union is embedded (e.g. resize is an LLM-exposed op).
    assert "resize" in json.dumps(strict)


# --------------------------------------------------------------------------- editor


def test_autoeditor_full_loop_returns_valid_edit() -> None:
    analysis, stem = _real_video_analysis()
    planner = _StubPlanner({"segments": [{"scene_id": f"{stem}#0"}, {"scene_id": f"{stem}#1"}]})
    editor = AutoEditor(planner)

    edit = editor.edit_from_analyses([analysis], "make a short clip")

    assert len(edit.segments) == 2
    assert edit.segments[0].start == 0.0
    edit.validate()  # raises if the produced plan is not actually runnable
    assert planner.calls, "planner should have been invoked"
    parts = planner.calls[0]["parts"]
    assert any(getattr(p, "image", None) is not None for p in parts), "planner should receive keyframes"


def test_autoeditor_unknown_id_exhausts_rounds() -> None:
    analysis, _ = _real_video_analysis()
    planner = _StubPlanner({"segments": [{"scene_id": "does-not-exist"}]})
    editor = AutoEditor(planner, max_rounds=2)

    with pytest.raises(AutoEditError):
        editor.edit_from_analyses([analysis], "anything")
    assert len(planner.calls) == 2


class _SequencePlanner:
    """Returns successive canned plans on successive calls (last one repeats)."""

    def __init__(self, plans: list[dict[str, Any]]) -> None:
        self.plans = list(plans)
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, *, system: str, parts: list[Any], schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"system": system, "parts": parts, "schema": schema})
        idx = min(len(self.calls) - 1, len(self.plans) - 1)
        return json.loads(json.dumps(self.plans[idx]))


def test_autoeditor_validation_error_then_succeeds() -> None:
    analysis, stem = _real_video_analysis()
    planner = _SequencePlanner(
        [
            {"segments": []},  # min_length violation -> ValidationError, retried
            {"segments": [{"scene_id": f"{stem}#0"}]},  # valid on the second round
        ]
    )
    editor = AutoEditor(planner, max_rounds=3)

    edit = editor.edit_from_analyses([analysis], "x")
    assert len(edit.segments) == 1
    assert len(planner.calls) == 2


def test_autoeditor_multi_source_normalizes_and_validates() -> None:
    small_meta = VideoMetadata.from_path(SMALL_VIDEO_PATH)
    big_meta = VideoMetadata.from_path(BIG_VIDEO_PATH)
    a_small = _analysis(
        str(SMALL_VIDEO_PATH),
        [_scene(0, 0.0, min(1.0, small_meta.total_seconds), caption="small clip")],
        fps=small_meta.fps,
        width=small_meta.width,
        height=small_meta.height,
        frame_count=small_meta.frame_count,
        duration=small_meta.total_seconds,
    )
    a_big = _analysis(
        str(BIG_VIDEO_PATH),
        [_scene(0, 0.0, min(1.0, big_meta.total_seconds), caption="big clip")],
        fps=big_meta.fps,
        width=big_meta.width,
        height=big_meta.height,
        frame_count=big_meta.frame_count,
        duration=big_meta.total_seconds,
    )
    small_stem = Path(SMALL_VIDEO_PATH).stem
    big_stem = Path(BIG_VIDEO_PATH).stem
    planner = _StubPlanner({"segments": [{"scene_id": f"{small_stem}#0"}, {"scene_id": f"{big_stem}#0"}]})

    edit = AutoEditor(planner).edit_from_analyses([a_small, a_big], "stitch them")

    assert len(edit.segments) == 2
    assert {str(s.source) for s in edit.segments} == {str(SMALL_VIDEO_PATH), str(BIG_VIDEO_PATH)}
    # Runnable end to end across two differently-sized sources: validate() chains
    # per-source metadata (keyed by source path) and confirms concat compatibility
    # (handled by VideoEdit's match_to_lowest_resolution default).
    edit.validate()


def test_resolve_plan_passes_post_operations_and_transitions() -> None:
    catalog = EditCatalog(
        scenes=[
            CatalogScene(id="s0", source=Path("a.mp4"), start=0.0, end=2.0, duration=2.0),
            CatalogScene(id="s1", source=Path("a.mp4"), start=2.0, end=4.0, duration=2.0),
        ]
    )
    plan = EditPlan(
        segments=[
            PlanSegment(scene_id="s0"),
            PlanSegment(scene_id="s1", transition_in={"type": "fade", "duration": 0.5}),
        ],
        post_operations=[{"op": "resize", "width": 100, "height": 100}],
    )
    edit = resolve_plan(plan, catalog)

    assert edit.segments[1].transition_in is not None
    assert edit.segments[1].transition_in.type == "fade"
    assert edit.post_operations[0].op == "resize"


def test_catalog_transcript_truncation() -> None:
    words = [TranscriptionWord(start=i * 0.1, end=i * 0.1 + 0.1, word=f"word{i}") for i in range(50)]
    tx = Transcription(words=words)
    analysis = _analysis("vid.mp4", [_scene(0, 0.0, 10.0, caption="c")], transcription=tx)
    scene = build_catalog([analysis], keyframes=False, max_transcript_chars=20).catalog.scenes[0]
    assert len(scene.transcript) <= 20
    assert scene.transcript.endswith("...")


def test_catalog_transcript_no_overlap() -> None:
    tx = Transcription(
        segments=[
            TranscriptionSegment(
                start=0.0, end=1.0, text="hi", words=[TranscriptionWord(start=0.0, end=1.0, word="hi")]
            )
        ]
    )
    analysis = _analysis("vid.mp4", [_scene(0, 5.0, 8.0, caption="c")], transcription=tx)
    scene = build_catalog([analysis], keyframes=False).catalog.scenes[0]
    assert scene.transcript == ""
    assert scene.has_speech is False


def test_autoeditor_planner_error_is_retried() -> None:
    analysis, stem = _real_video_analysis()

    class _FlakyPlanner:
        def __init__(self) -> None:
            self.calls = 0

        def generate_json(self, *, system: str, parts: list[Any], schema: dict[str, Any]) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise PlannerError("unparseable output")
            return {"segments": [{"scene_id": f"{stem}#0"}]}

    planner = _FlakyPlanner()
    edit = AutoEditor(planner, max_rounds=3).edit_from_analyses([analysis], "x")
    assert len(edit.segments) == 1
    assert planner.calls == 2


def test_autoeditor_infra_error_propagates() -> None:
    analysis, _ = _real_video_analysis()

    class _BrokenPlanner:
        def generate_json(self, *, system: str, parts: list[Any], schema: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("connection refused")

    with pytest.raises(RuntimeError, match="connection refused"):
        AutoEditor(_BrokenPlanner()).edit_from_analyses([analysis], "x")


def test_editplan_strict_schema_round_trips() -> None:
    strict = EditPlan.json_schema(strict=True)
    seg = strict["properties"]["segments"]["items"]
    # Strict mode lists every property as required (provider-grammar contract);
    # a document that conforms (all keys present) parses back into EditPlan.
    assert set(seg["required"]) == {"scene_id", "operations", "transition_in"}
    full: dict[str, Any] = {
        "segments": [{"scene_id": "x", "operations": [], "transition_in": None}],
        "post_operations": [],
    }
    EditPlan.model_validate(full)

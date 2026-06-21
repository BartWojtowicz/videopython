"""Build the LLM-facing edit catalog from VideoAnalysis results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from videopython.base.video import extract_frames_at_times

from .models import CatalogBundle, CatalogScene, EditCatalog

if TYPE_CHECKING:
    from videopython.ai.video_analysis import SceneAnalysisSample, VideoAnalysis
    from videopython.base.transcription import Transcription

DEFAULT_TRANSCRIPT_CHARS = 280


def build_catalog(
    analyses: Sequence[VideoAnalysis],
    *,
    keyframes: bool = True,
    max_transcript_chars: int = DEFAULT_TRANSCRIPT_CHARS,
) -> CatalogBundle:
    """Project VideoAnalysis results into a flat scene catalog with one midpoint keyframe per scene."""
    scenes: list[CatalogScene] = []
    frames: dict[str, np.ndarray] = {}
    used_ids: set[str] = set()

    for analysis in analyses:
        source_path = analysis.source.path
        samples = analysis.scenes.samples if analysis.scenes else []
        transcription = analysis.audio.transcription if analysis.audio else None
        stem = Path(source_path).stem if source_path else "clip"

        for sample in samples:
            scene_id = _unique_id(stem, sample.scene_index, used_ids)
            caption, shot_type = _description(sample)
            transcript = _transcript_excerpt(
                transcription, sample.start_second, sample.end_second, max_transcript_chars
            )
            scenes.append(
                CatalogScene(
                    id=scene_id,
                    source=Path(source_path) if source_path else Path(stem),
                    start=sample.start_second,
                    end=sample.end_second,
                    duration=max(0.0, sample.end_second - sample.start_second),
                    shot_type=shot_type,
                    caption=caption,
                    transcript=transcript,
                    has_speech=bool(transcript),
                    has_faces=bool(sample.faces),
                )
            )
            if keyframes:
                if source_path is None:
                    raise ValueError(f"Scene {scene_id!r} has no source path to extract a keyframe from.")
                midpoint = (sample.start_second + sample.end_second) / 2.0
                frames[scene_id] = extract_frames_at_times(source_path, [midpoint])[0]

    return CatalogBundle(catalog=EditCatalog(scenes=scenes), keyframes=frames)


def _unique_id(stem: str, scene_index: int, used: set[str]) -> str:
    base = f"{stem}#{scene_index}"
    candidate, suffix = base, 2
    while candidate in used:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _description(sample: SceneAnalysisSample) -> tuple[str, str | None]:
    desc = sample.scene_description
    return (desc.caption or "", desc.shot_type) if desc else ("", None)


def _transcript_excerpt(transcription: Transcription | None, start: float, end: float, max_chars: int) -> str:
    sliced = transcription.slice(start, end) if transcription else None
    if sliced is None:
        return ""
    text = " ".join(" ".join(segment.text.split()) for segment in sliced.segments).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text

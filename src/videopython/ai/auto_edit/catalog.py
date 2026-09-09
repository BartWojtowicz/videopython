"""Build the LLM-facing edit catalog from VideoAnalysis results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from videopython.base.video import extract_frames_at_times

from ._speech import speech_passages
from .models import CatalogBundle, CatalogScene, EditCatalog, SpeechCandidateConfig

if TYPE_CHECKING:
    from videopython.ai.video_analysis import SceneAnalysisSample, VideoAnalysis
    from videopython.base.transcription import Transcription

DEFAULT_TRANSCRIPT_CHARS = 280


def build_catalog(
    analyses: Sequence[VideoAnalysis],
    *,
    keyframes: bool = True,
    max_transcript_chars: int = DEFAULT_TRANSCRIPT_CHARS,
    mode: Literal["visual", "speech"] = "visual",
    speech: SpeechCandidateConfig | None = None,
) -> CatalogBundle:
    """Build visual scenes or timed speech passages; speech mode requires speech settings."""
    if mode not in ("visual", "speech") or (mode == "speech") != (speech is not None):
        raise ValueError("Supply speech settings exactly when mode='speech'")
    scenes: list[CatalogScene] = []
    transcripts: dict[str, str] = {}
    used_ids: set[str] = set()

    for analysis in analyses:
        source_path = analysis.source.path
        samples = analysis.scenes.samples if analysis.scenes else []
        transcription = analysis.audio.transcription if analysis.audio else None
        stem = Path(source_path).stem if source_path else "clip"

        if speech is not None:
            identity = json.dumps(
                [
                    str(Path(source_path).resolve()) if source_path else None,
                    transcription.model_dump() if transcription else None,
                    speech.model_dump(),
                ],
                sort_keys=True,
            )
            digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
            for index, passage in enumerate(speech_passages(transcription, speech)):
                start, end = passage[0].start, max(word.end for word in passage)
                scene_id = _unique_id(f"{stem}#speech-{digest}", index, used_ids)
                text = " ".join(" ".join(word.word.split()) for word in passage).strip()
                scenes.append(
                    CatalogScene(
                        id=scene_id,
                        source=Path(source_path) if source_path else Path(stem),
                        start=start,
                        end=end,
                        duration=end - start,
                        transcript=_shorten(text, max_transcript_chars),
                        has_speech=True,
                    )
                )
                transcripts[scene_id] = text
                if keyframes and source_path is None:
                    raise ValueError(f"Scene {scene_id!r} has no source path to extract a keyframe from.")
            continue

        for sample in samples:
            scene_id = _unique_id(stem, sample.scene_index, used_ids)
            caption, shot_type = _description(sample)
            text = _transcript_text(transcription, sample.start_second, sample.end_second)
            transcripts[scene_id] = text
            transcript = _shorten(text, max_transcript_chars)
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
            if keyframes and source_path is None:
                raise ValueError(f"Scene {scene_id!r} has no source path to extract a keyframe from.")

    frames = extract_catalog_keyframes(scenes) if keyframes else {}
    return CatalogBundle(catalog=EditCatalog(scenes=scenes), keyframes=frames, transcripts=transcripts)


def extract_catalog_keyframes(scenes: Sequence[CatalogScene]) -> dict[str, np.ndarray]:
    """Extract scene midpoints in one decode per source, preserving scene IDs."""
    by_source: dict[Path, list[CatalogScene]] = {}
    for scene in scenes:
        by_source.setdefault(scene.source, []).append(scene)
    frames: dict[str, np.ndarray] = {}
    for source, source_scenes in by_source.items():
        timestamps = [(scene.start + scene.end) / 2.0 for scene in source_scenes]
        extracted = extract_frames_at_times(source, timestamps)
        for index, scene in enumerate(source_scenes):
            frames[scene.id] = extracted[index]
    return frames


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


def _transcript_text(transcription: Transcription | None, start: float, end: float) -> str:
    sliced = transcription.slice(start, end) if transcription else None
    if sliced is None:
        return ""
    return " ".join(" ".join(segment.text.split()) for segment in sliced.segments).strip()


def _shorten(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    return text

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from videopython.ai.dubbing.models import TranslatedSegment
from videopython.audio import Audio
from videopython.base.transcription import Transcription


class DubbingCheckpoint:
    """Persist restartable artifacts for one file-based dubbing run."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._speech_dir = self.root / "speech"
        self._speech_dir.mkdir(exist_ok=True)
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError(f"Checkpoint does not match this dubbing run: {self.root}")
        else:
            self._write_json(manifest_path, manifest)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        temporary.replace(path)

    def load_transcription(self) -> Transcription | None:
        path = self.root / "transcription.json"
        return Transcription.model_validate_json(path.read_text()) if path.exists() else None

    def save_transcription(self, transcription: Transcription) -> None:
        self._write_json(self.root / "transcription.json", transcription.model_dump(mode="json"))

    def load_translation(self) -> tuple[list[TranslatedSegment], list[int]] | None:
        path = self.root / "translation.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        segments = [TranslatedSegment.model_validate(value) for value in payload["segments"]]
        return segments, payload["failures"]

    def save_translation(self, segments: list[TranslatedSegment], failures: list[int]) -> None:
        self._write_json(
            self.root / "translation.json",
            {"segments": [segment.model_dump(mode="json") for segment in segments], "failures": failures},
        )

    def speech_path(self, indices: list[int]) -> Path:
        name = "-".join(str(index) for index in indices)
        return self._speech_dir / f"{name}.wav"

    def load_speech(self, indices: list[int]) -> Audio | None:
        path = self.speech_path(indices)
        return Audio.from_path(path) if path.exists() else None

    def save_speech(self, indices: list[int], audio: Audio) -> Path:
        path = self.speech_path(indices)
        temporary = path.with_name(f".{path.stem}.tmp.wav")
        audio.save(temporary)
        temporary.replace(path)
        return path

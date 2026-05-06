"""Filesystem-backed cache for resumable dubbing runs.

A long dub crashes at TTS segment 312/400 today and re-runs Whisper,
Demucs, translation, and the first 311 TTS segments from scratch.
:class:`DubCache` stores three artifacts so subsequent runs skip stages
whose inputs match:

- ``transcription.json`` — output of ``AudioToText.transcribe``.
- ``translation_<key>.json`` — output of ``TranslationBackend.translate_segments``.
- ``tts/<key>.wav`` — per-segment TTS WAV.

Cache directories are opt-in via ``VideoDubber(cache_dir=...)`` / ``LocalDubbingPipeline(cache_dir=...)``.
``cache_dir=None`` (default) is a no-op pass-through.

Hash inputs are conservative — false misses (re-run a stage) are cheap;
false hits (deliver a stale dub) are bugs. Source-audio identity uses a
sha256 of the raw float32 bytes, not file path, so re-encoding the same
content invalidates correctly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from videopython.base.audio import Audio
    from videopython.base.text.transcription import Transcription

logger = logging.getLogger(__name__)


# Cache schema version. Bump on incompatible changes to any artifact's
# on-disk format (e.g. TranscriptionSegment field changes that break
# from_dict). Mismatched cache entries are treated as a miss.
SCHEMA_VERSION = 1

# Reserved for M4.3 per-speaker voice library. M3.2 does not write here;
# documented so future code knows the path is taken.
_VOICE_CLONES_SUBDIR = "voice_clones"


@dataclass(frozen=True)
class _ArtifactPaths:
    """Resolved paths for a single source's cache directory."""

    src_dir: Path
    metadata: Path
    transcription: Path
    tts_dir: Path

    def translation_path(self, lang_key: str) -> Path:
        return self.src_dir / f"translation_{lang_key}.json"

    def tts_path(self, seg_key: str) -> Path:
        return self.tts_dir / f"{seg_key}.wav"


def _stable_hash(*parts: str | int | float | bool | None) -> str:
    """Short hex digest over a tuple of primitive values.

    Stable across runs — uses ``str(part)`` so int/float/bool/None all
    serialize deterministically. 16 hex chars (64 bits) is plenty of
    space for the small cardinality we're hashing into.
    """
    h = hashlib.sha256()
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _audio_bytes_hash(audio: Audio) -> str:
    """sha256 over the raw audio data buffer.

    Used as the per-source cache directory name. Bytes-level so re-encoded
    sources (different container, same content) collide intentionally only
    when the decoded float32 buffer matches.
    """
    h = hashlib.sha256()
    h.update(audio.data.tobytes())
    return h.hexdigest()[:16]


class DubCache:
    """Filesystem cache for transcription, translation, and TTS artifacts.

    Layout under ``root``::

        <root>/<src_hash>/
            metadata.json              # schema version + hash inputs
            transcription.json         # populated on transcription cache miss
            translation_<lang_key>.json
            tts/<seg_key>.wav
            voice_clones/              # reserved for M4.3, not written here

    All getters return ``None`` on miss. Putters are idempotent
    (overwrite). Schema-version mismatch is treated as a miss for every
    artifact under that source.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ----- key derivation --------------------------------------------------

    @staticmethod
    def source_key(audio: Audio) -> str:
        """Per-source identifier — sha256 of the raw audio buffer.

        This is the directory name under ``root``; one dir per distinct
        source, regardless of which stage's kwargs vary.
        """
        return _audio_bytes_hash(audio)

    @staticmethod
    def transcription_kwargs_hash(
        *,
        whisper_model: str,
        enable_diarization: bool,
        condition_on_previous_text: bool,
        no_speech_threshold: float,
        logprob_threshold: float | None,
    ) -> str:
        return _stable_hash(
            whisper_model,
            enable_diarization,
            condition_on_previous_text,
            no_speech_threshold,
            logprob_threshold,
        )

    @staticmethod
    def translation_key(
        *,
        source_lang: str,
        target_lang: str,
        translator_class: str,
    ) -> str:
        """Hash captures the source/target pair + the resolved backend class.

        ``translator_class`` is the *resolved* class name (e.g. ``"MarianTranslator"``),
        not the user-supplied ``"auto"`` — a CPU run that resolves to Marian
        must not collide with a GPU run that resolves to Qwen on the same
        language pair.
        """
        return _stable_hash(source_lang, target_lang, translator_class)

    @staticmethod
    def tts_key(
        *,
        translated_text: str,
        voice_sample_bytes: bytes | None,
        language: str,
    ) -> str:
        """Per-segment key over text + voice sample + language."""
        h = hashlib.sha256()
        h.update(translated_text.encode("utf-8"))
        h.update(b"\x00")
        h.update(voice_sample_bytes or b"")
        h.update(b"\x00")
        h.update(language.encode("utf-8"))
        return h.hexdigest()[:16]

    # ----- path resolution -------------------------------------------------

    def _paths_for(self, src_hash: str) -> _ArtifactPaths:
        src_dir = self.root / src_hash
        return _ArtifactPaths(
            src_dir=src_dir,
            metadata=src_dir / "metadata.json",
            transcription=src_dir / "transcription.json",
            tts_dir=src_dir / "tts",
        )

    def _ensure_metadata(self, paths: _ArtifactPaths, hash_inputs: dict[str, Any]) -> None:
        """Create the source dir + metadata.json if missing.

        ``hash_inputs`` records the kwargs we hashed against so a future
        schema change can audit cache entries. The schema field is
        load-bearing: mismatched versions invalidate the entire source dir.
        """
        paths.src_dir.mkdir(parents=True, exist_ok=True)
        paths.tts_dir.mkdir(parents=True, exist_ok=True)
        if not paths.metadata.exists():
            paths.metadata.write_text(
                json.dumps(
                    {"schema": SCHEMA_VERSION, "hash_inputs": hash_inputs},
                    indent=2,
                ),
                encoding="utf-8",
            )

    def _schema_ok(self, paths: _ArtifactPaths) -> bool:
        if not paths.metadata.exists():
            return True  # fresh dir; we'll write metadata on first put.
        try:
            data = json.loads(paths.metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return data.get("schema") == SCHEMA_VERSION

    # ----- transcription ---------------------------------------------------

    def get_transcription(self, src_hash: str, kwargs_hash: str) -> Transcription | None:
        from videopython.base.text.transcription import Transcription

        paths = self._paths_for(src_hash)
        if not paths.transcription.exists() or not self._schema_ok(paths):
            return None
        try:
            data = json.loads(paths.transcription.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if data.get("kwargs_hash") != kwargs_hash:
            return None
        logger.info("cache hit: transcription (%s)", src_hash)
        return Transcription.from_dict(data["transcription"])

    def put_transcription(
        self,
        src_hash: str,
        kwargs_hash: str,
        transcription: Transcription,
        hash_inputs: dict[str, Any],
    ) -> None:
        paths = self._paths_for(src_hash)
        self._ensure_metadata(paths, hash_inputs)
        paths.transcription.write_text(
            json.dumps(
                {"kwargs_hash": kwargs_hash, "transcription": transcription.to_dict()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # ----- translation -----------------------------------------------------

    def get_translation(self, src_hash: str, lang_key: str) -> list[dict[str, Any]] | None:
        paths = self._paths_for(src_hash)
        if not self._schema_ok(paths):
            return None
        path = paths.translation_path(lang_key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        logger.info("cache hit: translation (%s/%s)", src_hash, lang_key)
        return data["segments"]

    def put_translation(self, src_hash: str, lang_key: str, segments_dict: list[dict[str, Any]]) -> None:
        paths = self._paths_for(src_hash)
        self._ensure_metadata(paths, {})
        paths.translation_path(lang_key).write_text(
            json.dumps({"segments": segments_dict}, ensure_ascii=False),
            encoding="utf-8",
        )

    # ----- tts -------------------------------------------------------------

    def get_tts_path(self, src_hash: str, seg_key: str) -> Path | None:
        paths = self._paths_for(src_hash)
        if not self._schema_ok(paths):
            return None
        path = paths.tts_path(seg_key)
        return path if path.exists() else None

    def reserve_tts_path(self, src_hash: str, seg_key: str) -> Path:
        """Return the path TTS output should be written to. Caller is
        responsible for the actual write (Audio.save)."""
        paths = self._paths_for(src_hash)
        self._ensure_metadata(paths, {})
        return paths.tts_path(seg_key)


def dub_cache_clear(cache_dir: str | Path, src_hash: str | None = None) -> None:
    """Delete cache entries for a specific source or the whole cache root.

    No auto-eviction in M3.2 — call this to reclaim disk space when a
    cache directory has grown unwieldy. Safe no-op if ``cache_dir`` or
    ``cache_dir/<src_hash>`` does not exist.
    """
    import shutil

    root = Path(cache_dir)
    target = root / src_hash if src_hash else root
    if target.exists():
        shutil.rmtree(target)
        logger.info("dub_cache_clear: removed %s", target)

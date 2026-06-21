"""Packaging + import-topology tests for ``videopython.ai``.

They enforce, mechanically:

* The ``[dependency-groups].ai`` dev group stays consistent with the ``[ai]``
  optional-dependencies extra, and every dep declared in ``[ai]`` is imported
  somewhere under ``ai/`` (no vestigial declarations).
* No heavy ML dependency is imported at module top level anywhere under ``ai/``
  (so leaf modules stay importable on a slim install; deps load lazily).
* Importing a leaf symbol (or ``videopython.ai`` itself) does NOT eagerly import
  sibling AI leaf modules — i.e. the PEP 562 lazy ``__getattr__`` chains work.
* ``ai._optional.require`` raises an actionable, extra-pointing ``ImportError``.
* The local ``TextToSpeech`` satisfies the ``SpeechBackend`` protocol and the
  dubbing pipeline accepts an injected backend without importing chatterbox.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path
from typing import cast

import pytest

from tests.conftest import _flatten_extra, _toplevel_imports

# Parsed-TOML shape is recursive; an Any-valued mapping is the honest type here.
Pyproject = dict[str, object]

_REPO_ROOT = Path(__file__).parent.parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_AI_ROOT = Path(__file__).parent.parent / "videopython" / "ai"

# Top-level distributions whose import names differ from the PyPI package name.
# Used by the "every declared dep is actually imported somewhere" check.
_DIST_TO_IMPORT_NAMES: dict[str, set[str]] = {
    "openai-whisper": {"whisper"},
    "pyannote-audio": {"pyannote"},
    "silero-vad": {"silero_vad"},
    "transnetv2-pytorch": {"transnetv2_pytorch"},
    "qwen-vl-utils": {"qwen_vl_utils"},
    "chatterbox-tts": {"chatterbox"},
    "llama-cpp-python": {"llama_cpp"},
    "scikit-learn": {"sklearn"},
}

# Heavy ML import-names that must never appear at module top level under ai/.
_HEAVY_IMPORT_NAMES: set[str] = {
    "torch",
    "torchaudio",
    "transformers",
    "diffusers",
    "whisper",
    "pyannote",
    "silero_vad",
    "ultralytics",
    "transnetv2_pytorch",
    "chatterbox",
    "demucs",
    "llama_cpp",
    "sentencepiece",
    "pyloudnorm",
    "qwen_vl_utils",
    "imagehash",
    "accelerate",
    "numba",
    "scipy",
    "sklearn",
    "ollama",
    "huggingface_hub",
}


@pytest.fixture(scope="module")
def pyproject() -> Pyproject:
    with open(_PYPROJECT, "rb") as f:
        return tomllib.load(f)


def _req_name(requirement: str) -> str:
    """Lowercased distribution name from a PEP 508 requirement string."""
    name = requirement.split(";")[0].strip()
    for sep in ("[", ">", "<", "=", "!", "~", " ", "@"):
        name = name.split(sep)[0]
    return name.strip().lower()


def _optional_deps(pyproject: Pyproject) -> dict[str, list[str]]:
    """Typed view of ``[project.optional-dependencies]``."""
    project = cast("dict[str, object]", pyproject["project"])
    return cast("dict[str, list[str]]", project["optional-dependencies"])


def _dependency_groups(pyproject: Pyproject) -> dict[str, list[str]]:
    """Typed view of ``[dependency-groups]`` (empty when absent)."""
    return cast("dict[str, list[str]]", pyproject.get("dependency-groups", {}))


# --------------------------------------------------------------------------- #
# Drift guard
# --------------------------------------------------------------------------- #


def test_dependency_group_ai_matches_optional_ai(pyproject: Pyproject) -> None:
    """[dependency-groups].ai resolves the same set as [project.optional-dependencies].ai."""
    opt = _optional_deps(pyproject)
    groups = _dependency_groups(pyproject)
    assert "ai" in groups, "Expected a thin [dependency-groups].ai for uv dev workflows"

    aggregate = _flatten_extra(opt, "ai")

    # The dev group references the package's own [ai] extra; flatten it the same
    # way (a `videopython[ai]` entry).
    group_deps: set[str] = set()
    for req in groups["ai"]:
        if _req_name(req) == "videopython":
            inner = req.split("[", 1)[1].split("]", 1)[0]
            for ref in inner.split(","):
                group_deps |= _flatten_extra(opt, ref.strip())
        else:
            group_deps.add(req.strip())

    assert group_deps == aggregate, (
        "dependency-groups.ai drifted from optional-dependencies.ai:\n"
        f"  only in group: {sorted(group_deps - aggregate)}\n"
        f"  only in extra: {sorted(aggregate - group_deps)}"
    )


# Deps that are declared as resolver co-pins/floors but have NO direct import
# under ai/ — they're pulled transitively by a sibling dep that we DO import:
#   torchaudio  -> co-pin for the torch stack (whisper/chatterbox/demucs)
#   sentencepiece -> MarianTokenizer needs it transitively (no `import sentencepiece`)
#   accelerate  -> diffusers/transformers device-map plumbing (transitive)
_TRANSITIVE_ONLY_DEPS = {"torchaudio", "sentencepiece", "accelerate"}


def test_every_declared_dep_is_imported_somewhere(pyproject: Pyproject) -> None:
    """Each dep declared in [ai] is actually imported under ai/ (lazily, anywhere)
    — or is a known transitive-only co-pin. Guards against vestigial declarations."""
    extras = _optional_deps(pyproject)
    declared = {_req_name(r) for r in _flatten_extra(extras, "ai")}

    source = "\n".join(p.read_text() for p in _AI_ROOT.rglob("*.py"))

    def _is_referenced(import_name: str) -> bool:
        # Matches `import X`, `from X import`, `from X.sub import`, AND the
        # lazy `require("X...")` / `require("X.sub", ...)` literal form.
        return (
            f"import {import_name}" in source
            or f"from {import_name} " in source
            or f"from {import_name}." in source
            or f'"{import_name}"' in source
            or f'"{import_name}.' in source
        )

    missing: list[str] = []
    for dist in declared:
        if dist in _TRANSITIVE_ONLY_DEPS:
            continue
        import_names = _DIST_TO_IMPORT_NAMES.get(dist, {dist.replace("-", "_")})
        if not any(_is_referenced(imp) for imp in import_names):
            missing.append(dist)

    assert not missing, f"Declared but never imported under ai/: {sorted(missing)}"


# --------------------------------------------------------------------------- #
# Import topology
# --------------------------------------------------------------------------- #


def test_no_heavy_toplevel_imports_in_ai() -> None:
    """No heavy ML dep is imported at module top level anywhere under ai/."""
    violations: list[tuple[str, list[str]]] = []
    for file_path in _AI_ROOT.rglob("*.py"):
        heavy = [imp for imp in _toplevel_imports(file_path) if imp.split(".")[0] in _HEAVY_IMPORT_NAMES]
        if heavy:
            violations.append((str(file_path.relative_to(_REPO_ROOT)), heavy))

    assert not violations, "Heavy deps imported at top level (must be lazy):\n" + "\n".join(
        f"  {path}: {imps}" for path, imps in violations
    )


def _ai_leaf_modules_in_sys() -> set[str]:
    """AI leaf modules currently resident in sys.modules.

    Excludes the lightweight spine (``_optional``, ``_device``) and the package
    __init__ shells, which are allowed to load.
    """
    allowed = {
        "videopython.ai",
        "videopython.ai._optional",
        "videopython.ai._device",
        "videopython.ai.generation",
        "videopython.ai.understanding",
        "videopython.ai.dubbing",
        "videopython.ai.video_analysis",
        "videopython.ai.generation._tts_backend",
    }
    return {m for m in sys.modules if m.startswith("videopython.ai.") and m not in allowed}


def _import_in_clean_ai(import_stmt: str) -> set[str]:
    """Drop all videopython.ai modules, run ``import_stmt``, return the set of AI
    leaf modules that got pulled in. Restores sys.modules afterwards."""
    ai_modules = [k for k in sys.modules if k.startswith("videopython.ai")]
    cached = {k: sys.modules.pop(k) for k in ai_modules}
    try:
        exec(import_stmt, {})  # noqa: S102 - test-controlled literal
        return _ai_leaf_modules_in_sys()
    finally:
        for k in list(sys.modules):
            if k.startswith("videopython.ai"):
                del sys.modules[k]
        sys.modules.update(cached)


def test_import_ai_package_does_not_load_leaf_modules() -> None:
    """`import videopython.ai` must not eagerly load any leaf module."""
    pulled = _import_in_clean_ai("import videopython.ai")
    assert not pulled, f"import videopython.ai eagerly loaded leaf modules: {sorted(pulled)}"


def test_import_one_symbol_does_not_load_siblings() -> None:
    """Importing ObjectDetector ([vision]) must not pull audio/generation siblings."""
    pulled = _import_in_clean_ai("from videopython.ai.understanding import ObjectDetector")
    siblings = {
        m
        for m in pulled
        if m
        in {
            "videopython.ai.understanding.audio",
            "videopython.ai.understanding.separation",
            "videopython.ai.generation.audio",
            "videopython.ai.generation.video",
        }
    }
    assert not siblings, f"Importing ObjectDetector pulled in sibling leaf modules: {sorted(siblings)}"


# --------------------------------------------------------------------------- #
# _optional.require
# --------------------------------------------------------------------------- #


def test_require_raises_with_extra_and_pip_hint() -> None:
    from videopython.ai._optional import require

    with pytest.raises(ImportError) as exc_info:
        require("definitely_not_a_real_module_xyz", "ai", feature="TextToSpeech")

    msg = str(exc_info.value)
    assert "pip install" in msg
    assert "videopython[ai]" in msg


def test_require_returns_module_when_present() -> None:
    from videopython.ai._optional import require

    mod = require("json", "asr")
    import json

    assert mod is json


# --------------------------------------------------------------------------- #
# SpeechBackend protocol + injection
# --------------------------------------------------------------------------- #


def test_local_tts_satisfies_speech_backend() -> None:
    from videopython.ai.generation._tts_backend import SpeechBackend

    # Structural (runtime_checkable) match — TextToSpeech is never instantiated
    # here, so chatterbox is not imported.
    tts_cls = importlib.import_module("videopython.ai.generation.audio").TextToSpeech
    instance = tts_cls.__new__(tts_cls)  # no __init__ side effects needed
    assert isinstance(instance, SpeechBackend)


def test_pipeline_uses_injected_backend_without_chatterbox() -> None:
    """An injected SpeechBackend is used as-is; _init_tts never imports chatterbox."""
    from videopython.ai.dubbing.pipeline import LocalDubbingPipeline
    from videopython.ai.generation._tts_backend import SpeechBackend
    from videopython.audio import Audio, AudioMetadata

    class FakeBackend:
        def generate_audio(self, text, voice_sample=None, voice_sample_path=None, **kwargs):
            import numpy as np

            data = np.zeros(2400, dtype=np.float32)
            meta = AudioMetadata(sample_rate=24000, channels=1, sample_width=2, duration_seconds=0.1, frame_count=2400)
            return Audio(data, meta)

    backend = FakeBackend()
    assert isinstance(backend, SpeechBackend)

    chatterbox_before = "chatterbox" in sys.modules
    pipeline = LocalDubbingPipeline(tts_backend=backend)
    pipeline._init_tts(language="es")

    assert pipeline._tts is backend
    if not chatterbox_before:
        assert "chatterbox" not in sys.modules, "Injecting a backend must not import chatterbox"

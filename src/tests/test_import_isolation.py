"""Test that non-AI subpackages don't pull in videopython.ai.

videopython.ai brings in heavy ML dependencies (torch, diffusers, whisper,
demucs, ...). Anything outside it must stay importable on a vanilla
``pip install videopython`` (no ``[ai]`` extra).
"""

import importlib
import sys
from pathlib import Path

import pytest

from tests.conftest import _toplevel_imports

NON_AI_SUBPACKAGES = ["videopython.base", "videopython.audio", "videopython.editing"]


def _package_files(package: str) -> list[Path]:
    parts = package.split(".")
    pkg_path = Path(__file__).parent.parent / "/".join(parts)
    return list(pkg_path.rglob("*.py"))


@pytest.mark.parametrize("package", NON_AI_SUBPACKAGES)
def test_no_toplevel_ai_imports(package: str) -> None:
    files = _package_files(package)
    assert files, f"Expected to find Python files under {package}"

    violations: list[tuple[Path, list[str]]] = []
    for file_path in files:
        ai_imports = [imp for imp in _toplevel_imports(file_path) if imp.startswith("videopython.ai")]
        if ai_imports:
            violations.append((file_path, ai_imports))

    if violations:
        msg = [f"{package} must not import from videopython.ai at top level:"]
        for file_path, ai_imports in violations:
            relative_path = file_path.relative_to(file_path.parent.parent.parent.parent)
            msg.append(f"  {relative_path}: {ai_imports}")
        raise AssertionError("\n".join(msg))


@pytest.mark.parametrize("package", NON_AI_SUBPACKAGES)
def test_importable_without_ai(package: str) -> None:
    """Importing the subpackage must not pull in any videopython.ai module."""
    ai_modules = [key for key in sys.modules if key.startswith("videopython.ai")]
    cached = {key: sys.modules.pop(key) for key in ai_modules}

    try:
        module = importlib.import_module(package)
        importlib.reload(module)
        pulled_in = [key for key in sys.modules if key.startswith("videopython.ai")]
        assert not pulled_in, f"Importing {package} pulled in AI modules: {pulled_in}"
    finally:
        sys.modules.update(cached)

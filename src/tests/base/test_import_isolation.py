"""Test that videopython.base does not import from videopython.ai.

This ensures the base module can be used without AI dependencies installed.
"""

import ast
import sys
from pathlib import Path


def get_base_module_files() -> list[Path]:
    """Get all Python files in the base module."""
    base_path = Path(__file__).parent.parent.parent / "videopython" / "base"
    return list(base_path.rglob("*.py"))


def get_imports_from_file(file_path: Path) -> list[str]:
    """Extract all import statements from a Python file."""
    with open(file_path) as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return imports


def test_base_module_does_not_import_ai():
    """Verify that no file in videopython.base imports from videopython.ai."""
    base_files = get_base_module_files()
    assert len(base_files) > 0, "Should find base module files"

    violations = []
    for file_path in base_files:
        imports = get_imports_from_file(file_path)
        ai_imports = [imp for imp in imports if imp.startswith("videopython.ai")]
        if ai_imports:
            violations.append((file_path, ai_imports))

    if violations:
        msg_parts = ["Base module should not import from videopython.ai:"]
        for file_path, ai_imports in violations:
            relative_path = file_path.relative_to(file_path.parent.parent.parent.parent)
            msg_parts.append(f"  {relative_path}: {ai_imports}")
        raise AssertionError("\n".join(msg_parts))


def test_base_module_importable_without_ai():
    """Verify that videopython.base can be imported independently.

    This is a runtime check that the base module doesn't have hidden
    dependencies on AI modules through lazy imports or similar.
    """
    # Clear any cached imports of AI module
    ai_modules = [key for key in sys.modules if key.startswith("videopython.ai")]
    cached = {key: sys.modules.pop(key) for key in ai_modules}

    try:
        # Re-import base module - should work without AI
        import importlib

        import videopython.base

        importlib.reload(videopython.base)

        # Check that AI modules weren't pulled in
        newly_imported_ai = [key for key in sys.modules if key.startswith("videopython.ai")]
        assert not newly_imported_ai, f"Importing base pulled in AI modules: {newly_imported_ai}"
    finally:
        # Restore cached modules
        sys.modules.update(cached)

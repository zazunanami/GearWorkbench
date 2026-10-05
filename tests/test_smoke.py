from __future__ import annotations

import compileall
import os
from pathlib import Path

from makelpro.analysis import default_mesh_summary


PUBLIC_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PUBLIC_ROOT / "src" / "makelpro"
EXPECTED_MODULES = [
    SRC_ROOT / "__init__.py",
    SRC_ROOT / "analysis.py",
    SRC_ROOT / "gearbox_gui.py",
    SRC_ROOT / "technic_draw.py",
    SRC_ROOT / "fbd.py",
]


# Local tool and cache folders (all gitignored) are never part of the public release.
LOCAL_ONLY_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}


def _join(*parts: str) -> str:
    return "".join(parts)


def _release_files() -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(PUBLIC_ROOT):
        dirnames[:] = [name for name in dirnames if name not in LOCAL_ONLY_DIRS]
        files.extend(Path(dirpath) / name for name in filenames)
    return files


FORBIDDEN_PATTERNS = [
    _join("Licensed by ", "Fimaki", " Games"),
    _join("230", "601", "012051"),
    _join("230", "601"),
    _join("C:", "\\", "Users"),
    _join("/", "home", "/"),
    _join("Down", "loads"),
    _join("One", "Drive"),
    _join("gmail", ".com"),
    _join("Ah", "met"),
    _join("Gül", "tekin"),
    _join("Gul", "tekin"),
]


def test_expected_modules_exist() -> None:
    for path in EXPECTED_MODULES:
        assert path.exists(), f"Missing expected public file: {path}"


def test_license_and_readme_branding() -> None:
    license_path = PUBLIC_ROOT / "LICENSE"
    readme_path = PUBLIC_ROOT / "README.md"
    assert license_path.exists(), "Apache-2.0 license file is required"
    readme_text = readme_path.read_text(encoding="utf-8")
    assert "Zazu Nanami" in readme_text
    assert "Apache License 2.0" in readme_text


def test_sanitized_report_folder_exists() -> None:
    report_dir = PUBLIC_ROOT / "docs" / "report"
    assert report_dir.exists()
    assert any(report_dir.iterdir()), "Expected at least one sanitized report artifact"


def test_compileall_passes_for_public_release() -> None:
    for folder in ("src", "tests"):
        assert compileall.compile_dir(str(PUBLIC_ROOT / folder), quiet=1)


def test_no_forbidden_strings_in_public_release() -> None:
    for path in _release_files():
        data = path.read_bytes()
        lowered = data.lower()
        for pattern in FORBIDDEN_PATTERNS:
            assert pattern.lower().encode("utf-8") not in lowered, f"Forbidden string {pattern!r} found in {path}"


def test_no_disallowed_artifacts_in_public_release() -> None:
    banned_suffixes = {".exe", ".toc", ".pkg", ".pyz"}
    for path in _release_files():
        parts_lower = {part.lower() for part in path.relative_to(PUBLIC_ROOT).parts}
        assert "build" not in parts_lower
        assert path.suffix.lower() not in banned_suffixes


def test_default_mesh_regression_values() -> None:
    summary = default_mesh_summary()
    assert abs(summary["phi_t_deg"] - 22.7958772589) < 1e-6
    assert abs(summary["ft23_n"] - 2000.0) < 1e-9
    assert abs(summary["fr23_n"] - 840.6) < 0.1
    assert abs(summary["resultant23_n"] - 2169.5) < 0.1
    assert abs(summary["ft45_n"] - 400.0) < 1e-9
    assert abs(summary["fr45_n"] - 168.1) < 0.1
    assert abs(summary["resultant45_n"] - 433.9) < 0.1

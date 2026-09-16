import json
from pathlib import Path

import pytest

from oncolncai.deployment import (
    PublicPackageError, build_public_package, scan_public_tree,
)


def _spec(root: Path, files: list[str]) -> Path:
    path = root / "public.json"
    path.write_text(json.dumps({"files": files, "directories": []}), encoding="utf-8")
    return path


def test_builder_copies_only_allowlisted_files_and_sorts_inventory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "src/pkg").mkdir(parents=True)
    (source / "src/pkg/z.py").write_text("Z = 1\n", encoding="utf-8")
    (source / "src/pkg/a.py").write_text("A = 1\n", encoding="utf-8")
    (source / "AGENTS.md").write_text("internal\n", encoding="utf-8")
    spec = _spec(source, ["src/pkg/z.py", "src/pkg/a.py"])

    inventory = build_public_package(source, tmp_path / "dist", spec)

    assert [item.path for item in inventory.files] == ["src/pkg/a.py", "src/pkg/z.py"]
    assert not (tmp_path / "dist/AGENTS.md").exists()
    assert len(inventory.aggregate_sha256) == 64


def test_builder_fails_closed_when_allowlisted_source_is_missing(tmp_path: Path) -> None:
    spec = _spec(tmp_path, ["missing.py"])
    with pytest.raises(PublicPackageError, match="missing allowlisted"):
        build_public_package(tmp_path, tmp_path / "dist", spec)
    assert not (tmp_path / "dist").exists()


def test_scanner_redacts_secret_and_private_path_values(tmp_path: Path) -> None:
    secret = "hf_" + "abcdefghijklmnopqrstuvwxyz"
    private = "C:/" + "Users/private/data"
    (tmp_path / "bad.txt").write_text(f"token={secret}\npath={private}\n", encoding="utf-8")

    findings = scan_public_tree(tmp_path)

    assert {(item.category, item.path) for item in findings} == {
        ("secret-pattern", "bad.txt"), ("private-path", "bad.txt")
    }
    assert all(secret not in item.message and private not in item.message for item in findings)


def test_builder_rejects_internal_path_even_when_allowlisted(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("internal\n", encoding="utf-8")
    with pytest.raises(PublicPackageError, match="forbidden public path"):
        build_public_package(tmp_path, tmp_path / "dist", _spec(tmp_path, ["AGENTS.md"]))

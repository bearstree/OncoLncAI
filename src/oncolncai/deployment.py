"""Fail-closed public-package construction and deployment metadata primitives."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from tempfile import mkdtemp

from pydantic import BaseModel, ConfigDict, Field


FORBIDDEN_PARTS = frozenset({
    "AGENTS.md", "CODEX_IMPLEMENTATION_INSTRUCTIONS.md", ".env", ".git",
    ".cache", ".pytest_cache", "__pycache__", "notebooks", "training",
    "artifacts", "benchmarks", "runs", "outputs", "build",
})
TEXT_SUFFIXES = frozenset({"", ".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".csv", ".jinja", ".lock", ".example"})
SECRET_PATTERNS = (
    re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(?:api[_-]?key|password|token)\s*[:=]\s*['\"](?!example|replace|optional)[^'\"\s]{12,}['\"]"),
)
PRIVATE_PATH_PATTERNS = (
    re.compile(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\s'\"]+"),
    re.compile("/" + r"home/[^/\s'\"]+"),
    re.compile("/" + r"Users/[^/\s'\"]+"),
)


class PublicPackageError(RuntimeError):
    pass


class PublicPackageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    files: tuple[str, ...] = ()
    directories: tuple[str, ...] = ()


class ScanFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    category: str
    path: str
    message: str


class PublicFileHash(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0)


class PublicInventory(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    files: tuple[PublicFileHash, ...]
    aggregate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DeploymentManifest(BaseModel):
    """Auditable identity and operational boundaries for one public build."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    project_name: str = "oncolncai"
    project_version: str
    source_git_sha: str
    git_dirty: bool
    build_timestamp_utc: str
    python_version: str
    dependency_versions: dict[str, str]
    lockfile_sha256: str
    public_content_sha256: str
    public_inventory_path: str = "deployment/public_inventory.json"
    supported_modes: tuple[str, ...] = ("DEMO", "REAL")
    tested_tcga_projects: tuple[str, ...] = ("TCGA-LUAD", "TCGA-BRCA", "TCGA-KIRC")
    entry_points: dict[str, str] = {"web": "app.py", "cli": "examples/run_real_analysis.py"}
    run_manifest_location: str = "configured output directory per analysis run"
    github: dict[str, str] | None = None
    hugging_face: dict[str, str] | None = None
    hosted_limitations: tuple[str, ...] = (
        "runtime storage may be ephemeral",
        "full REAL analyses depend on network access, memory, and compute limits",
    )


def _portable(path: Path) -> str:
    return path.as_posix()


def _validate_relative(value: str) -> Path:
    pure = PurePosixPath(value.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise PublicPackageError(f"invalid public path: {value}")
    if any(part in FORBIDDEN_PARTS for part in pure.parts):
        raise PublicPackageError(f"forbidden public path: {value}")
    return Path(*pure.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_public_tree(root: Path, *, paths: tuple[str, ...] | None = None) -> PublicInventory:
    candidates = ((root / value for value in paths) if paths is not None else root.rglob("*"))
    files = tuple(PublicFileHash(path=_portable(path.relative_to(root)), sha256=_sha256(path), size=path.stat().st_size)
                  for path in sorted((item for item in candidates if item.is_file()),
                                     key=lambda item: _portable(item.relative_to(root))))
    canonical = "".join(f"{item.path}\0{item.sha256}\0{item.size}\n" for item in files)
    return PublicInventory(files=files, aggregate_sha256=hashlib.sha256(canonical.encode()).hexdigest())


def scan_public_tree(root: Path) -> tuple[ScanFinding, ...]:
    findings: list[ScanFinding] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = _portable(path.relative_to(root))
        if any(part in FORBIDDEN_PARTS for part in PurePosixPath(relative).parts):
            findings.append(ScanFinding(category="internal-path", path=relative, message="forbidden internal path"))
            continue
        if path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append(ScanFinding(category="secret-pattern", path=relative, message="possible credential pattern; value redacted"))
        if any(pattern.search(text) for pattern in PRIVATE_PATH_PATTERNS):
            findings.append(ScanFinding(category="private-path", path=relative, message="machine-specific private path; value redacted"))
    return tuple(findings)


def _git_value(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments), cwd=source_root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _deployment_manifest(source_root: Path, inventory: PublicInventory) -> DeploymentManifest:
    project = tomllib.loads((source_root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies: dict[str, str] = {}
    for line in (source_root / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==", 1)
            dependencies[name] = version
    targets_path = source_root / "deployment/public_targets.json"
    targets = json.loads(targets_path.read_text(encoding="utf-8")) if targets_path.is_file() else {}
    return DeploymentManifest(
        project_version=project["version"],
        source_git_sha=_git_value(source_root, "rev-parse", "HEAD"),
        git_dirty=bool(_git_value(source_root, "status", "--porcelain")),
        build_timestamp_utc=datetime.now(UTC).isoformat(),
        python_version=sys.version.split()[0],
        dependency_versions=dependencies,
        lockfile_sha256=_sha256(source_root / "requirements.lock"),
        public_content_sha256=inventory.aggregate_sha256,
        github=targets.get("github"),
        hugging_face=targets.get("hugging_face"),
    )


def verify_public_package(root: Path) -> PublicInventory:
    """Fail unless every inventoried file is present, unchanged, and scan-clean."""

    if scan_public_tree(root):
        raise PublicPackageError("public package scan failed")
    saved = PublicInventory.model_validate_json(
        (root / "deployment/public_inventory.json").read_text(encoding="utf-8")
    )
    current = inventory_public_tree(root, paths=tuple(item.path for item in saved.files))
    if current != saved:
        raise PublicPackageError("public package hash mismatch")
    return current


def build_public_package(source_root: Path, destination: Path, spec_path: Path) -> PublicInventory:
    source_root, destination = source_root.resolve(), destination.resolve()
    spec = PublicPackageSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    selected: set[Path] = set()
    for value in spec.files:
        relative = _validate_relative(value)
        source = source_root / relative
        if not source.is_file():
            raise PublicPackageError(f"missing allowlisted file: {value}")
        selected.add(relative)
    for value in spec.directories:
        relative = _validate_relative(value)
        source = source_root / relative
        if not source.is_dir():
            raise PublicPackageError(f"missing allowlisted directory: {value}")
        for path in source.rglob("*"):
            if path.is_file() and not any(part in FORBIDDEN_PARTS for part in path.relative_to(source_root).parts):
                selected.add(path.relative_to(source_root))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for relative in sorted(selected, key=_portable):
            source = (source_root / relative).resolve()
            if source_root not in source.parents:
                raise PublicPackageError(f"allowlisted path escapes source root: {_portable(relative)}")
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        findings = scan_public_tree(temporary)
        if findings:
            summary = ", ".join(f"{item.category}:{item.path}" for item in findings)
            raise PublicPackageError(f"public scan failed: {summary}")
        inventory = inventory_public_tree(temporary)
        inventory_path = temporary / "deployment/public_inventory.json"
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(inventory.model_dump_json(indent=2) + "\n", encoding="utf-8")
        if (source_root / "pyproject.toml").is_file() and (source_root / "requirements.lock").is_file():
            (temporary / "deployment/deployment_manifest.json").write_text(
                _deployment_manifest(source_root, inventory).model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return inventory
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

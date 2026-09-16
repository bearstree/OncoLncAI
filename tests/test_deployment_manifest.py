import json
from pathlib import Path

from oncolncai.deployment import (
    DeploymentManifest,
    build_public_package,
    verify_public_package,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_build_emits_valid_deployment_manifest(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "public"
    inventory = build_public_package(ROOT, target, ROOT / "deployment/public_files.json")
    manifest = DeploymentManifest.model_validate_json(
        (target / "deployment/deployment_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.project_version == "0.1.0"
    assert manifest.public_content_sha256 == inventory.aggregate_sha256
    assert manifest.supported_modes == ("DEMO", "REAL")
    assert manifest.tested_tcga_projects == ("TCGA-LUAD", "TCGA-BRCA", "TCGA-KIRC")
    assert manifest.source_git_sha
    assert manifest.github == {
        "repository": "bearstree/OncoLncAI",
        "url": "https://github.com/bearstree/OncoLncAI",
    }
    assert manifest.hugging_face == {
        "space_id": "weiyi/OncoLncAI",
        "url": "https://huggingface.co/spaces/weiyi/OncoLncAI",
        "sdk": "gradio",
    }
    assert verify_public_package(target).aggregate_sha256 == inventory.aggregate_sha256


def test_verification_detects_changed_public_file(tmp_path: Path) -> None:
    target = tmp_path / "public"
    build_public_package(ROOT, target, ROOT / "deployment/public_files.json")
    (target / "README.md").write_text("changed", encoding="utf-8")
    try:
        verify_public_package(target)
    except Exception as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("verification accepted a modified package")


def test_deployment_manifest_schema_is_version_controlled() -> None:
    schema = json.loads((ROOT / "deployment/deployment_manifest.schema.json").read_text())
    assert schema["title"] == "DeploymentManifest"

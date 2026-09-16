import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_hugging_face_entrypoint_exposes_production_demo() -> None:
    spec = importlib.util.spec_from_file_location("oncolncai_hf_entrypoint", ROOT / "app.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from app.main import demo
    assert module.demo is demo


def test_hugging_face_entrypoint_defaults_to_public_binding() -> None:
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'ONCOLNCAI_SERVER_NAME", "0.0.0.0"' in text


def test_deployment_docs_state_hosted_persistence_and_real_mode_boundaries() -> None:
    text = (ROOT / "docs/deployment.md").read_text(encoding="utf-8")
    assert "runtime-storage lifetime" in text
    assert "Cross-restart" in text
    assert "never substitutes synthetic" in text


def test_attribution_names_all_external_biomedical_sources() -> None:
    text = (ROOT / "docs/third_party_and_data_sources.md").read_text(encoding="utf-8")
    for name in ("NCI Genomic Data Commons", "UCSC Xena", "NCBI GEO", "PubMed"):
        assert name in text


def test_public_license_exists() -> None:
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_space_requirements_are_locked_with_documented_gradio_compatibility_override() -> None:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    lock_lines = (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
    expected = [
        "pydantic==2.12.5" if line.startswith("pydantic==") else
        "pydantic-core==2.41.5" if line.startswith("pydantic-core==") else line
        for line in lock_lines
    ]
    assert lines == expected
    docker_allowlist = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "!requirements.txt" in docker_allowlist
    assert "!app.py" in docker_allowlist


def test_space_launcher_adds_packaged_src_before_app_import() -> None:
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'sys.path.insert(0, str(Path(__file__).parent / "src"))' in text
    assert text.index("sys.path.insert") < text.index("from app.main import demo")

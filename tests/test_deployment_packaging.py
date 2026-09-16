from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_runs_non_root_service_on_all_interfaces() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER oncolncai" in dockerfile
    assert "EXPOSE 7860" in dockerfile
    assert "ONCOLNCAI_SERVER_NAME=0.0.0.0" in dockerfile
    assert 'CMD ["python", "app/main.py"]' in dockerfile
    assert "COPY ." not in dockerfile
    assert "PYTHONPATH=/opt/oncolncai/src" in dockerfile
    assert "pip install --no-deps" not in dockerfile
    assert "API_KEY=" not in dockerfile


def test_local_launch_uses_browser_address_while_docker_binds_all_interfaces() -> None:
    app = (ROOT / "app/main.py").read_text(encoding="utf-8")
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'os.getenv("ONCOLNCAI_SERVER_NAME", "127.0.0.1")' in app
    assert "ONCOLNCAI_SERVER_NAME=127.0.0.1" in example
    assert "ONCOLNCAI_SERVER_NAME=0.0.0.0" in dockerfile


def test_dockerignore_is_allow_listed_and_excludes_development_artifacts() -> None:
    rules = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert rules[0] == "*"
    allowed = {rule for rule in rules if rule.startswith("!")}
    assert allowed == {
        "!Dockerfile", "!.dockerignore", "!requirements.lock", "!pyproject.toml",
        "!requirements.txt", "!README.md", "!app.py", "!src/", "!src/**", "!app/", "!app/**",
    }
    assert "!tests/**" not in allowed
    assert "!docs/**" not in allowed
    assert "!notebooks/**" not in allowed
    assert "**/__pycache__/" in rules
    assert "**/*.py[cod]" in rules


def test_runtime_lock_pins_every_dependency() -> None:
    lines = [line for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    assert lines
    assert all("==" in line for line in lines)
    assert any(line == "gradio==6.23.1" for line in lines)


def test_compose_publishes_a_reachable_local_port() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert '"${ONCOLNCAI_HOST_PORT:-7861}:7860"' in compose
    assert "ONCOLNCAI_LLM_PROVIDER: ${ONCOLNCAI_LLM_PROVIDER:-ollama}" in compose
    assert "qwen2.5-coder:14b" in compose
    assert "http://host.docker.internal:11434" in compose
    assert "ONCOLNCAI_LLM_API_KEY" not in compose
    assert "ONCOLNCAI_PUBLIC_URL: http://127.0.0.1:" in compose

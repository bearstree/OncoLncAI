"""Opt-in local smoke test: ONCOLNCAI_RUN_OLLAMA_SMOKE=1 python examples/ollama_smoke_test.py"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from oncolncai import OllamaProvider  # noqa: E402


class SmokeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str


def main() -> int:
    if os.environ.get("ONCOLNCAI_RUN_OLLAMA_SMOKE") != "1":
        print("SKIPPED: set ONCOLNCAI_RUN_OLLAMA_SMOKE=1 after starting Ollama.")
        return 0
    model = os.environ.get("ONCOLNCAI_LLM_MODEL", "qwen2.5-coder:14b")
    base_url = os.environ.get("ONCOLNCAI_LLM_BASE_URL", "http://127.0.0.1:11434")
    timeout = float(os.environ.get("ONCOLNCAI_LLM_TIMEOUT_SECONDS", "60"))
    provider = OllamaProvider(model=model, base_url=base_url, timeout=timeout)
    answer = SmokeAnswer.model_validate(
        provider.generate_structured(
            prompt='Return JSON with status exactly "ok".', output_schema=SmokeAnswer
        )
    )
    print(f"Ollama structured generation: {answer.status} (model={model})")
    return 0 if answer.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

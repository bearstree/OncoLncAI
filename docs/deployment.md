# Deployment

OncoLncAI can run locally, in Docker, or as a Hugging Face Space. The same
`app.main:demo` Gradio application is used in every environment.

## Local installation

Use Python 3.12 and install the locked runtime:

```bash
python -m pip install --requirement requirements.lock
python app.py
```

The service reads configuration from environment variables. Copy
`.env.example` as a reference, but inject secrets through the process or hosting
platform; do not put them in an image or source tree.

## Docker

```bash
docker build -t oncolncai:0.1.0 .
docker run --rm -p 7860:7860 --env-file .env oncolncai:0.1.0
```

Open `http://127.0.0.1:7860`. The server binds to `0.0.0.0` inside the
container. For host Ollama, set `ONCOLNCAI_LLM_BASE_URL` to the address reachable
from the container (commonly `http://host.docker.internal:11434`).

## Hugging Face Spaces

Create a Gradio Space, copy the curated public package into it, and configure
optional tokens in Space Settings as secrets. The root `app.py` is a thin
launcher; it does not duplicate application logic.

The Space uses the pinned `requirements.txt`. It differs from the general
`requirements.lock` only by pinning Pydantic 2.12.5 and its matching core because
the hosted Gradio MCP extra currently requires Pydantic 2.12.x or earlier; both
versions remain inside OncoLncAI's declared Pydantic compatibility range.

Hosted storage is normally limited to the current runtime-storage lifetime.
Cross-restart cache reuse requires persistent storage or an external artifact
store. Resource limits can also make a full TCGA analysis unsuitable for a free
CPU Space. The application reports such limitations and never substitutes synthetic
data for a requested REAL analysis.

## Real-mode behavior

REAL mode performs real retrieval and deterministic computation, or reports an
explicit failure/limitation. It does not silently fall back to a fixture. Every
run manifest records dataset identity, configuration, software versions, and
artifact provenance so a compatible cached result can be distinguished from a
new execution.

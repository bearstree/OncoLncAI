FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    ONCOLNCAI_ENVIRONMENT=container \
    ONCOLNCAI_CACHE_DIR=/tmp/oncolncai \
    ONCOLNCAI_LLM_PROVIDER=ollama \
    ONCOLNCAI_LLM_MODEL=qwen2.5-coder:14b \
    ONCOLNCAI_LLM_BASE_URL=http://host.docker.internal:11434 \
    ONCOLNCAI_SERVER_NAME=0.0.0.0 \
    ONCOLNCAI_SERVER_PORT=7860 \
    PYTHONPATH=/opt/oncolncai/src

WORKDIR /opt/oncolncai

COPY requirements.lock pyproject.toml README.md ./
RUN python -m pip install --requirement requirements.lock

COPY src ./src
COPY app ./app

RUN useradd --create-home --uid 10001 oncolncai \
    && chown -R oncolncai:oncolncai /opt/oncolncai
USER oncolncai

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3)" || exit 1

CMD ["python", "app/main.py"]

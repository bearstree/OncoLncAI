"""Vendor-neutral structured-generation providers."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from pydantic import BaseModel

from oncolncai.config import Settings


class LLMProvider(Protocol):
    """Minimal interface for schema-constrained model output."""

    def generate_structured(
        self, *, prompt: str, output_schema: type[BaseModel]
    ) -> Any: ...


class MockLLMProvider:
    """Deterministic response queue for tests and offline demonstrations."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, type[BaseModel]]] = []

    def generate_structured(
        self, *, prompt: str, output_schema: type[BaseModel]
    ) -> Any:
        self.calls.append((prompt, output_schema))
        if not self._responses:
            raise RuntimeError("mock provider has no remaining responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ProviderTransport(Protocol):
    def post(self, url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> bytes: ...


class UrllibProviderTransport:
    def post(self, url: str, *, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> bytes:
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read()


@dataclass(frozen=True)
class LLMProviderError(RuntimeError):
    provider: str
    code: str
    message: str
    retryable: bool = False

    def __str__(self) -> str:
        return f"{self.provider} {self.code}: {self.message}"


def _decode_json(content: Any) -> Any:
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, dict):
        return content
    raise TypeError("structured response content must be a JSON string or object")


class _HTTPStructuredProvider:
    provider_name: str

    def __init__(self, *, model: str, base_url: str, timeout: float, transport: ProviderTransport | None) -> None:
        if not model.strip():
            raise ValueError("model cannot be blank")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self._transport = transport or UrllibProviderTransport()

    def _post(self, endpoint: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = self._transport.post(urljoin(self.base_url, endpoint), headers=headers, payload=payload, timeout=self.timeout)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise TypeError("provider response must be a JSON object")
            return parsed
        except HTTPError as exc:
            raise LLMProviderError(self.provider_name, "http_error", f"HTTP {exc.code}", exc.code == 429 or exc.code >= 500) from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            raise LLMProviderError(self.provider_name, "connection_error", str(exc) or type(exc).__name__, True) from exc
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise LLMProviderError(self.provider_name, "invalid_response", str(exc) or type(exc).__name__, False) from exc


class OpenAICompatibleProvider(_HTTPStructuredProvider):
    """OpenAI Chat Completions-compatible structured-output adapter."""

    provider_name = "openai"

    def __init__(self, *, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: float = 60.0, transport: ProviderTransport | None = None) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI-compatible provider requires an API key")
        super().__init__(model=model, base_url=base_url, timeout=timeout, transport=transport)
        self._api_key = api_key

    def generate_structured(self, *, prompt: str, output_schema: type[BaseModel]) -> Any:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": output_schema.__name__, "strict": True, "schema": output_schema.model_json_schema()}},
        }
        response = self._post("chat/completions", {"Content-Type": "application/json", "Authorization": f"Bearer {self._api_key}"}, payload)
        try:
            return _decode_json(response["choices"][0]["message"]["content"])
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMProviderError(self.provider_name, "invalid_response", str(exc) or type(exc).__name__, False) from exc


class OllamaProvider(_HTTPStructuredProvider):
    """Ollama `/api/chat` structured-output adapter."""

    provider_name = "ollama"

    def __init__(self, *, model: str = "qwen2.5-coder:14b", base_url: str = "http://127.0.0.1:11434", timeout: float = 60.0, transport: ProviderTransport | None = None) -> None:
        super().__init__(model=model, base_url=base_url, timeout=timeout, transport=transport)

    def generate_structured(self, *, prompt: str, output_schema: type[BaseModel]) -> Any:
        payload = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "format": output_schema.model_json_schema(), "stream": False}
        response = self._post("api/chat", {"Content-Type": "application/json"}, payload)
        try:
            return _decode_json(response["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMProviderError(self.provider_name, "invalid_response", str(exc) or type(exc).__name__, False) from exc


def create_llm_provider(settings: Settings, *, mock_responses: list[Any] | None = None, transport: ProviderTransport | None = None) -> LLMProvider:
    """Construct a configured provider without coupling application components to it."""
    if settings.llm_provider == "mock":
        return MockLLMProvider(mock_responses or [])
    if settings.llm_provider == "ollama":
        return OllamaProvider(model=settings.llm_model, base_url=settings.llm_base_url or "http://127.0.0.1:11434", timeout=settings.llm_timeout_seconds, transport=transport)
    if settings.llm_provider == "openai":
        if not settings.llm_api_key:
            raise ValueError("ONCOLNCAI_LLM_API_KEY is required for the OpenAI provider")
        return OpenAICompatibleProvider(model=settings.llm_model, api_key=settings.llm_api_key.get_secret_value(), base_url=settings.llm_base_url or "https://api.openai.com/v1", timeout=settings.llm_timeout_seconds, transport=transport)
    raise ValueError(f"unsupported LLM provider: {settings.llm_provider}")

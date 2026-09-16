import json
from urllib.error import URLError

import pytest
from pydantic import BaseModel, ConfigDict

from oncolncai import (
    LLMProviderError,
    MockLLMProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    Settings,
    create_llm_provider,
)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str


class FakeTransport:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, url, *, headers, payload, timeout):
        self.calls.append((url, headers, payload, timeout))
        if self.error:
            raise self.error
        return json.dumps(self.response).encode()


def test_ollama_uses_schema_model_and_timeout() -> None:
    transport = FakeTransport({"message": {"content": '{"value":"local"}'}})
    provider = OllamaProvider(transport=transport, timeout=12)

    assert Answer.model_validate(provider.generate_structured(prompt="same", output_schema=Answer)).value == "local"
    url, headers, payload, timeout = transport.calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert payload["model"] == "qwen2.5-coder:14b"
    assert payload["format"] == Answer.model_json_schema()
    assert payload["stream"] is False
    assert timeout == 12
    assert "Authorization" not in headers


def test_openai_compatible_uses_same_protocol_and_keeps_key_in_header() -> None:
    transport = FakeTransport({"choices": [{"message": {"content": '{"value":"cloud"}'}}]})
    provider = OpenAICompatibleProvider(model="gpt-test", api_key="secret", transport=transport)

    assert Answer.model_validate(provider.generate_structured(prompt="same", output_schema=Answer)).value == "cloud"
    url, headers, payload, _ = transport.calls[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert "secret" not in json.dumps(payload)
    assert payload["response_format"]["json_schema"]["schema"] == Answer.model_json_schema()


def test_connection_and_invalid_response_errors_are_typed() -> None:
    offline = OllamaProvider(transport=FakeTransport(error=URLError("offline")))
    with pytest.raises(LLMProviderError) as caught:
        offline.generate_structured(prompt="x", output_schema=Answer)
    assert caught.value.code == "connection_error" and caught.value.retryable

    malformed = OllamaProvider(transport=FakeTransport({"message": {"content": "not json"}}))
    with pytest.raises(LLMProviderError) as caught:
        malformed.generate_structured(prompt="x", output_schema=Answer)
    assert caught.value.code == "invalid_response" and not caught.value.retryable


def test_timeout_is_a_retryable_typed_error() -> None:
    provider = OllamaProvider(transport=FakeTransport(error=TimeoutError("timed out")))
    with pytest.raises(LLMProviderError) as caught:
        provider.generate_structured(prompt="x", output_schema=Answer)
    assert caught.value.code == "connection_error"
    assert caught.value.retryable


def test_configuration_factory_selects_all_existing_provider_paths() -> None:
    mock = create_llm_provider(Settings(), mock_responses=[{"value": "mock"}])
    assert isinstance(mock, MockLLMProvider)

    ollama = create_llm_provider(Settings(llm_provider="ollama"), transport=FakeTransport())
    assert isinstance(ollama, OllamaProvider)
    assert ollama.model == "qwen2.5-coder:14b"

    openai = create_llm_provider(Settings(llm_provider="openai", llm_model="gpt-test", llm_api_key="secret"), transport=FakeTransport())
    assert isinstance(openai, OpenAICompatibleProvider)
    with pytest.raises(ValueError, match="API_KEY"):
        create_llm_provider(Settings(llm_provider="openai", llm_model="gpt-test"))


def test_mock_openai_and_ollama_return_same_structured_shape() -> None:
    providers = [
        MockLLMProvider([{"value": "same"}]),
        OpenAICompatibleProvider(model="gpt-test", api_key="secret", transport=FakeTransport({"choices": [{"message": {"content": '{"value":"same"}'}}]})),
        OllamaProvider(transport=FakeTransport({"message": {"content": '{"value":"same"}'}})),
    ]
    assert [Answer.model_validate(provider.generate_structured(prompt="request", output_schema=Answer)).value for provider in providers] == ["same"] * 3

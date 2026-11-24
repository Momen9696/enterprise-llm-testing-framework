"""Provider adapters and prompt templates."""

from __future__ import annotations

import json
import threading

import httpx
import pytest

from llm_eval.models import LLMRequest, LLMResponse, PromptTemplateSpec
from llm_eval.providers.base import (
    InvalidProviderResponse,
    ProviderError,
    ProviderTimeoutError,
    invoke_provider,
)
from llm_eval.providers.mock import MockProvider
from llm_eval.providers.openai_compatible import OpenAICompatibleProvider
from llm_eval.templates import TemplateError, render_template


def test_mock_is_deterministic_and_cycles_variants(responses_path) -> None:
    provider = MockProvider.from_file(responses_path)
    request = LLMRequest(prompt="ignored", metadata={"case_id": "qa-capital-france"})
    first = provider.generate(request)
    second = provider.generate(request)
    assert first.text == second.text == "Paris"
    assert first.prompt_tokens == 8
    assert first.completion_tokens == 1

    varied = LLMRequest(prompt="ignored", metadata={"case_id": "consistency-unstable"})
    texts = [provider.generate(varied).text for _ in range(3)]
    assert texts[0] == texts[2]
    assert texts[0] != texts[1]


def test_mock_timeout_error_invalid_and_missing_script(responses_path) -> None:
    provider = MockProvider.from_file(responses_path)
    with pytest.raises(ProviderTimeoutError):
        provider.generate(LLMRequest(prompt="x", metadata={"case_id": "timeout-simulated"}))
    with pytest.raises(ProviderError) as provider_error:
        provider.generate(LLMRequest(prompt="x", metadata={"case_id": "provider-error"}))
    assert "upstream connection reset" in str(provider_error.value)
    with pytest.raises(InvalidProviderResponse):
        provider.generate(LLMRequest(prompt="x", metadata={"case_id": "invalid-response"}))
    with pytest.raises(ProviderError, match="No mock script"):
        provider.generate(LLMRequest(prompt="x", metadata={"case_id": "missing-script"}))


def test_invoke_provider_enforces_wall_clock_timeout_and_rejects_bad_payloads() -> None:
    class Blocking:
        name = "blocking"
        model = "slow"

        def __init__(self) -> None:
            self.release = threading.Event()

        def generate(self, request: LLMRequest) -> LLMResponse:
            self.release.wait(2)
            return LLMResponse(text="late", provider=self.name, model=self.model)

    blocker = Blocking()
    with pytest.raises(ProviderTimeoutError):
        invoke_provider(blocker, LLMRequest(prompt="wait"), timeout_s=0.05)
    blocker.release.set()

    class Weird:
        name = "weird"
        model = "nope"

        def generate(self, request: LLMRequest) -> dict:
            return {"text": "nope"}

    with pytest.raises(InvalidProviderResponse):
        invoke_provider(Weird(), LLMRequest(prompt="x"), timeout_s=1)


def test_openai_compatible_adapter_parses_usage_and_failures(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="No API key"):
        OpenAICompatibleProvider(model="unit-model").generate(LLMRequest(prompt="Hi"))

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer unit-test-key"
        body = json.loads(request.content.decode())
        assert body["temperature"] == 0
        assert body["messages"][0]["role"] == "system"
        assert "Context:" in body["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "model": "unit-model",
                "choices": [{"message": {"content": "Paris"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    provider = OpenAICompatibleProvider(
        model="unit-model",
        api_key="unit-test-key",
        transport=httpx.MockTransport(handler),
    )
    response = provider.generate(
        LLMRequest(prompt="Capital?", system_prompt="Be brief.", context=["France is in Europe."])
    )
    assert response.text == "Paris"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 1

    def error_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    failing = OpenAICompatibleProvider(
        model="unit-model",
        api_key="unit-test-key",
        transport=httpx.MockTransport(error_handler),
    )
    with pytest.raises(ProviderError, match="HTTP 503"):
        failing.generate(LLMRequest(prompt="Hi"))

    def timeout_handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    timing_out = OpenAICompatibleProvider(
        model="unit-model",
        api_key="unit-test-key",
        transport=httpx.MockTransport(timeout_handler),
    )
    with pytest.raises(ProviderTimeoutError):
        timing_out.generate(LLMRequest(prompt="Hi"))

    def malformed_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    malformed = OpenAICompatibleProvider(
        model="unit-model",
        api_key="unit-test-key",
        transport=httpx.MockTransport(malformed_handler),
    )
    with pytest.raises(ProviderError, match="did not contain a message"):
        malformed.generate(LLMRequest(prompt="Hi"))


def test_template_rendering_is_literal_and_rejects_bad_variables() -> None:
    rendered = render_template(
        PromptTemplateSpec(
            name="grounded_qa",
            template="Context:\n{context}\n\nQuestion: {question}",
            input_variables=["context", "question"],
            variables={"context": "value {not_a_field}", "question": "When?"},
        )
    )
    assert rendered == "Context:\nvalue {not_a_field}\n\nQuestion: When?"
    with pytest.raises(TemplateError, match="not input variables"):
        render_template(
            PromptTemplateSpec(
                name="undeclared",
                template="Hello {name} {extra}",
                input_variables=["name"],
                variables={"name": "Ada"},
            )
        )
    with pytest.raises(TemplateError, match="Missing template variables"):
        render_template(
            PromptTemplateSpec(
                name="broken",
                template="Hello {name}",
                input_variables=["name", "title"],
                variables={"name": "Ada"},
            )
        )
    with pytest.raises(TemplateError, match="Unexpected"):
        render_template(
            PromptTemplateSpec(
                name="extra",
                template="Hello {name}",
                input_variables=["name"],
                variables={"name": "Ada", "title": "dev"},
            )
        )

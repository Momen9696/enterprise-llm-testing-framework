"""Optional OpenAI-compatible adapter.

The API key is read from the environment, or supplied by the caller at runtime.
This module never embeds credentials and is not used by the default dataset or CI.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from llm_eval.models import LLMRequest, LLMResponse
from llm_eval.providers.base import LLMProvider, ProviderError, ProviderTimeoutError

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_KEY_ENV = "OPENAI_API_KEY"


class OpenAICompatibleProvider(LLMProvider):
    """Chat-completions client for any API that speaks the OpenAI JSON shape."""

    name = "openai_compatible"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_key_env: str = DEFAULT_KEY_ENV,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._api_key_env = api_key_env
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    def generate(self, request: LLMRequest) -> LLMResponse:
        api_key = self._api_key if self._api_key is not None else os.environ.get(self._api_key_env)
        if not api_key:
            raise ProviderError(
                f"No API key was provided. Set the {self._api_key_env} environment variable "
                "or pass api_key at runtime. Refusing to call the provider."
            )

        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        user_content = request.prompt
        if request.context:
            joined = "\n\n".join(request.context)
            user_content = f"Context:\n{joined}\n\nQuestion:\n{request.prompt}"
        messages.append({"role": "user", "content": user_content})
        payload = {"model": self.model, "messages": messages, "temperature": 0}

        try:
            with httpx.Client(transport=self._transport, timeout=self._timeout_s) as client:
                response = client.post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"OpenAI-compatible request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI-compatible request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(f"OpenAI-compatible provider returned HTTP {response.status_code}")

        try:
            body: dict[str, Any] = response.json()
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProviderError("OpenAI-compatible payload did not contain a message") from exc
        if not isinstance(text, str):
            raise ProviderError("OpenAI-compatible message content was not a string")

        usage = body.get("usage") or {}
        return LLMResponse(
            text=text,
            provider=self.name,
            model=str(body.get("model") or self.model),
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            finish_reason=str(body["choices"][0].get("finish_reason") or "stop"),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value

"""Deterministic mock provider.

Scripts are loaded from local JSON. The same case id and call index always return the
same completion, so CI runs do not depend on a network or a paid model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from llm_eval.models import LLMRequest, LLMResponse, MockScriptFile, ScriptedResponse
from llm_eval.providers.base import InvalidProviderResponse, LLMProvider, ProviderError, ProviderTimeoutError


class MockProvider(LLMProvider):
    """Scripted provider keyed by case id, with optional ordered variants."""

    name = "mock"

    def __init__(
        self,
        scripts: dict[str, ScriptedResponse] | None = None,
        variants: dict[str, list[ScriptedResponse]] | None = None,
        model: str = "deterministic-mock-v1",
    ) -> None:
        self.model = model
        self.scripts = scripts or {}
        self.variants = variants or {}
        self._counts: dict[str, int] = {}

    @classmethod
    def from_file(cls, path: Path | str) -> MockProvider:
        payload = MockScriptFile.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
        return cls(scripts=payload.cases, variants=payload.variants, model=payload.model)

    def override(self, case_id: str, text: str, **usage: int) -> None:
        """Replace one script. Used to build an explicit regression fixture."""

        current = self.scripts.get(case_id, ScriptedResponse(text=""))
        self.scripts[case_id] = current.model_copy(update={"text": text, "error": None, **usage})
        self.variants.pop(case_id, None)

    def generate(self, request: LLMRequest) -> LLMResponse:
        case_id = str(request.metadata.get("case_id", ""))
        script = self._select(case_id)
        if script.sleep_s > 0:
            time.sleep(script.sleep_s)
        if script.error == "timeout":
            raise ProviderTimeoutError(script.message or f"Mock provider timed out for '{case_id}'")
        if script.error == "provider_error":
            raise ProviderError(script.message or f"Mock provider failed for '{case_id}'")
        if script.error == "invalid" or script.text is None:
            raise InvalidProviderResponse(script.message or f"Mock provider returned no text for '{case_id}'")
        return LLMResponse(
            text=script.text,
            provider=self.name,
            model=self.model,
            prompt_tokens=script.prompt_tokens,
            completion_tokens=script.completion_tokens,
            finish_reason=script.finish_reason,
        )

    def _select(self, case_id: str) -> ScriptedResponse:
        if case_id in self.variants:
            choices = self.variants[case_id]
            if not choices:
                raise ProviderError(f"Variant list for '{case_id}' is empty")
            index = self._counts.get(case_id, 0)
            self._counts[case_id] = index + 1
            return choices[index % len(choices)]
        if case_id in self.scripts:
            return self.scripts[case_id]
        raise ProviderError(f"No mock script for case '{case_id}'")

"""Provider exceptions and the timeout boundary around adapter calls."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from llm_eval.models import LLMRequest, LLMResponse


class ProviderError(RuntimeError):
    """The provider failed before returning a usable completion."""


class ProviderTimeoutError(ProviderError):
    """The provider exceeded the configured time budget or reported a timeout."""


class InvalidProviderResponse(ProviderError):
    """The provider returned a payload this framework cannot score."""


class LLMProvider:
    """Adapter contract. Implementations must be free of hardcoded credentials."""

    name: str = "provider"
    model: str = "unspecified"

    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


def invoke_provider(provider: LLMProvider, request: LLMRequest, timeout_s: float | None) -> LLMResponse:
    """Call a provider and always record measured wall-clock latency."""

    started = time.perf_counter()
    pool: ThreadPoolExecutor | None = None
    timed_out = False
    try:
        if timeout_s is not None:
            pool = ThreadPoolExecutor(max_workers=1)
            future = pool.submit(provider.generate, request)
            try:
                response = future.result(timeout=timeout_s)
            except FuturesTimeoutError as exc:
                timed_out = True
                raise ProviderTimeoutError(
                    f"{provider.name} exceeded timeout of {timeout_s} seconds"
                ) from exc
        else:
            response = provider.generate(request)
    except ProviderError:
        raise
    except Exception as exc:
        raise ProviderError(f"Unexpected provider failure: {exc}") from exc
    finally:
        if pool is not None:
            pool.shutdown(wait=not timed_out, cancel_futures=True)

    if not isinstance(response, LLMResponse):
        raise InvalidProviderResponse(
            f"{provider.name} returned {type(response).__name__}, expected LLMResponse"
        )
    if not isinstance(response.text, str):
        raise InvalidProviderResponse("Response text must be a string")
    measured_ms = (time.perf_counter() - started) * 1000
    return response.model_copy(update={"provider": provider.name, "latency_ms": measured_ms})

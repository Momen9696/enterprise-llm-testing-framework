"""Metric interface.

Every metric declares its definition, the inputs it reads, whether the score is a
deterministic check or a heuristic judgment, and the limitations of that judgment.
"""

from __future__ import annotations

from typing import Any

from llm_eval.models import EvalCase, LLMResponse, MetricMethod, MetricResult, RunConfig


class Metric:
    """Base class for evaluation metrics."""

    name: str
    definition: str
    required_inputs: list[str]
    method: MetricMethod
    limitations: str

    def evaluate(
        self,
        case: EvalCase,
        response: LLMResponse,
        config: RunConfig,
        extras: dict[str, Any],
    ) -> MetricResult:
        raise NotImplementedError

    def result(
        self,
        *,
        status: str,
        score: float | None,
        explanation: str,
        details: dict[str, Any] | None = None,
        inputs_used: list[str] | None = None,
    ) -> MetricResult:
        passed = None if status == "skipped" else status == "passed"
        return MetricResult(
            name=self.name,
            status=status,  # type: ignore[arg-type]
            score=score,
            passed=passed,
            method=self.method,
            definition=self.definition,
            explanation=explanation,
            details=details or {},
            limitations=self.limitations,
            inputs_used=inputs_used or list(self.required_inputs),
        )

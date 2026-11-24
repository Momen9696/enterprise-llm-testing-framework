"""Metric registry. Custom metrics can be added without changing the runner."""

from __future__ import annotations

from llm_eval.metrics.base import Metric
from llm_eval.metrics.builtin import BUILTIN_METRICS


class MetricRegistry:
    def __init__(self, metrics: list[Metric] | None = None) -> None:
        self._metrics: dict[str, Metric] = {}
        for metric in metrics if metrics is not None else BUILTIN_METRICS:
            self.register(metric)

    def register(self, metric: Metric) -> None:
        if not metric.name:
            raise ValueError("Metric name is required")
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Metric:
        try:
            return self._metrics[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._metrics))
            raise KeyError(f"Unknown metric '{name}'. Known metrics: {known}") from exc

    def names(self) -> list[str]:
        return sorted(self._metrics)

    def catalog(self) -> list[dict[str, object]]:
        return [
            {
                "name": metric.name,
                "definition": metric.definition,
                "required_inputs": list(metric.required_inputs),
                "method": metric.method,
                "limitations": metric.limitations,
            }
            for metric in self._metrics.values()
        ]


def default_registry() -> MetricRegistry:
    return MetricRegistry()

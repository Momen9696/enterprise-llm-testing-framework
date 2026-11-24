"""Batch evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_eval.metrics.registry import MetricRegistry, default_registry
from llm_eval.models import (
    CaseResult,
    EvalCase,
    EvaluationDataset,
    EvaluationReport,
    EvaluationSummary,
    LLMRequest,
    MetricSummary,
    RunConfig,
    utc_now,
)
from llm_eval.providers.base import (
    InvalidProviderResponse,
    LLMProvider,
    ProviderError,
    ProviderTimeoutError,
    invoke_provider,
)
from llm_eval.templates import TemplateError, render_template


def load_dataset(path: Path | str) -> EvaluationDataset:
    return EvaluationDataset.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def run_evaluation(
    dataset: EvaluationDataset,
    provider: LLMProvider,
    config: RunConfig | None = None,
    registry: MetricRegistry | None = None,
    run_id: str = "evaluation",
    case_ids: list[str] | None = None,
) -> EvaluationReport:
    """Evaluate a dataset in batch. One provider failure does not stop the remaining cases."""

    active_config = config or RunConfig()
    active_registry = registry or default_registry()
    selected = _select_cases(dataset, case_ids)
    _require_known_metrics(selected, active_registry)
    started = utc_now()
    results = [
        _evaluate_case(case, provider, active_config, active_registry) for case in selected
    ]
    finished = utc_now()
    return EvaluationReport(
        run_id=run_id,
        dataset=dataset.name,
        dataset_version=dataset.version,
        provider=provider.name,
        model=provider.model,
        config=active_config,
        started_at=started,
        finished_at=finished,
        summary=summarize(results),
        cases=results,
    )


def summarize(cases: list[CaseResult]) -> EvaluationSummary:
    """Aggregate case results. Infrastructure errors are reported separately from quality scores."""

    passed = sum(case.status == "passed" for case in cases)
    failed = sum(case.status == "failed" for case in cases)
    errors = sum(case.status == "error" for case in cases)
    scored = [case.score for case in cases if case.status != "error" and case.score is not None]
    latencies = [case.latency_ms for case in cases if case.latency_ms is not None]
    prompt_tokens = [case.prompt_tokens for case in cases if case.prompt_tokens is not None]
    completion_tokens = [
        case.completion_tokens for case in cases if case.completion_tokens is not None
    ]
    quality_total = passed + failed
    return EvaluationSummary(
        case_count=len(cases),
        passed=passed,
        failed=failed,
        infrastructure_errors=errors,
        pass_rate_excluding_errors=(passed / quality_total) if quality_total else None,
        mean_score_excluding_errors=(sum(scored) / len(scored)) if scored else None,
        mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        total_prompt_tokens=sum(prompt_tokens),
        total_completion_tokens=sum(completion_tokens),
        token_usage_available=bool(prompt_tokens or completion_tokens),
        metrics=_metric_summaries(cases),
    )


def _metric_summaries(cases: list[CaseResult]) -> list[MetricSummary]:
    rows: list[dict[str, object]] = []
    for case in cases:
        for metric in case.metrics:
            rows.append(
                {
                    "metric": metric.name,
                    "method": metric.method,
                    "score": metric.score,
                    "status": metric.status,
                }
            )
    frame = pd.DataFrame(rows, columns=["metric", "method", "score", "status"])
    summaries: list[MetricSummary] = []
    if frame.empty:
        return summaries
    for metric_name, group in frame.groupby("metric", sort=True):
        scored = group["score"].dropna()
        summaries.append(
            MetricSummary(
                name=str(metric_name),
                method=group["method"].iloc[0],
                mean_score=None if scored.empty else float(scored.mean()),
                passed=int((group["status"] == "passed").sum()),
                failed=int((group["status"] == "failed").sum()),
                skipped=int((group["status"] == "skipped").sum()),
            )
        )
    return summaries


def _select_cases(dataset: EvaluationDataset, case_ids: list[str] | None) -> list[EvalCase]:
    if case_ids is None:
        return list(dataset.cases)
    wanted = set(case_ids)
    known = {case.id for case in dataset.cases}
    missing = sorted(wanted - known)
    if missing:
        raise ValueError(f"Unknown case ids: {', '.join(missing)}")
    return [case for case in dataset.cases if case.id in wanted]


def _require_known_metrics(cases: list[EvalCase], registry: MetricRegistry) -> None:
    unknown = sorted({name for case in cases for name in case.metrics if name not in registry.names()})
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(unknown)}")


def _evaluate_case(
    case: EvalCase,
    provider: LLMProvider,
    config: RunConfig,
    registry: MetricRegistry,
) -> CaseResult:
    rendered = case.prompt
    try:
        if case.template is not None:
            rendered = render_template(case.template)
        request = LLMRequest(
            prompt=rendered,
            system_prompt=case.system_prompt,
            context=list(case.context),
            metadata={"case_id": case.id, "category": case.category},
        )
        sample_count = case.consistency_samples if "consistency" in case.metrics else 1
        samples: list[str] = []
        responses = []
        for _ in range(sample_count):
            response = invoke_provider(provider, request, config.timeout_s)
            responses.append(response)
            samples.append(response.text)
    except TemplateError as exc:
        return _error(case, "template", str(exc), rendered)
    except ProviderTimeoutError as exc:
        return _error(case, "timeout", str(exc), rendered)
    except InvalidProviderResponse as exc:
        return _error(case, "invalid_response", str(exc), rendered)
    except ProviderError as exc:
        return _error(case, "provider_error", str(exc), rendered)

    response = responses[-1]
    metric_results = [
        registry.get(name).evaluate(case, response, config, {"samples": samples})
        for name in case.metrics
    ]
    scored = [metric.score for metric in metric_results if metric.score is not None]
    failed = [metric for metric in metric_results if metric.status == "failed"]
    if not metric_results or all(metric.status == "skipped" for metric in metric_results):
        status = "failed"
        explanations = ["No metric produced a score for this case."]
    elif failed:
        status = "failed"
        explanations = [f"{metric.name}: {metric.explanation}" for metric in failed]
    else:
        status = "passed"
        explanations = []
    prompt_tokens = _sum_usage(responses, "prompt_tokens")
    completion_tokens = _sum_usage(responses, "completion_tokens")
    latency_ms = sum(item.latency_ms for item in responses)
    return CaseResult(
        case_id=case.id,
        category=case.category,
        status=status,
        score=(sum(scored) / len(scored)) if scored else None,
        metrics=metric_results,
        response_text=response.text,
        rendered_prompt=rendered,
        samples=samples,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=response.model,
        notes=case.notes,
        reference_outcome=case.reference_outcome,
        explanations=explanations,
    )


def _sum_usage(responses: list, field: str) -> int | None:
    """Sum provider-reported token counts. Return None when any call omitted usage."""

    values = [getattr(item, field) for item in responses]
    if any(value is None for value in values):
        return None
    return int(sum(values))


def _error(case: EvalCase, error_type: str, message: str, rendered: str | None) -> CaseResult:
    return CaseResult(
        case_id=case.id,
        category=case.category,
        status="error",
        score=None,
        rendered_prompt=rendered,
        error_type=error_type,
        error_message=message,
        notes=case.notes,
        reference_outcome=case.reference_outcome,
        explanations=[f"{error_type}: {message}"],
    )

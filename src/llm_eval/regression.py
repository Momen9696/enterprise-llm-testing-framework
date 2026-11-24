"""Compare two evaluation runs and flag score drops."""

from __future__ import annotations

from llm_eval.models import ComparisonReport, EvaluationReport, MetricDelta, MetricResult


def compare_reports(
    baseline: EvaluationReport,
    current: EvaluationReport,
    max_drop: float = 0.0,
) -> ComparisonReport:
    """A regression is a score drop greater than ``max_drop`` on the same case and metric.

    Missing current results for a metric that passed in the baseline are also regressions.
    Increases and unchanged scores are not regressions.
    """

    if max_drop < 0:
        raise ValueError("max_drop must be >= 0")
    baseline_map = _index(baseline)
    current_map = _index(current)
    deltas: list[MetricDelta] = []
    for key in sorted(set(baseline_map) | set(current_map)):
        case_id, metric_name = key
        before = baseline_map.get(key)
        after = current_map.get(key)
        drop = _drop(before, after)
        regressed = False
        if drop is not None and drop > max_drop:
            regressed = True
            explanation = (
                f"{metric_name} on {case_id} dropped by {drop:.4f}, "
                f"which is above the allowed drop of {max_drop:.4f}."
            )
        elif before and before.status == "passed" and (after is None or after.status != "passed"):
            regressed = True
            explanation = (
                f"{metric_name} on {case_id} passed in the baseline and did not pass in the current run."
            )
        elif before is None:
            explanation = f"{metric_name} on {case_id} is new in the current run."
        else:
            explanation = f"{metric_name} on {case_id} did not regress."
        deltas.append(
            MetricDelta(
                case_id=case_id,
                metric=metric_name,
                baseline_score=None if before is None else before.score,
                current_score=None if after is None else after.score,
                drop=drop,
                baseline_status=None if before is None else before.status,
                current_status=None if after is None else after.status,
                regressed=regressed,
                explanation=explanation,
            )
        )
    regressions = [item for item in deltas if item.regressed]
    return ComparisonReport(
        baseline_run_id=baseline.run_id,
        current_run_id=current.run_id,
        max_drop=max_drop,
        regression_count=len(regressions),
        regressions=regressions,
    )


def _index(report: EvaluationReport) -> dict[tuple[str, str], MetricResult]:
    return {
        (case.case_id, metric.name): metric
        for case in report.cases
        for metric in case.metrics
    }


def _drop(before: MetricResult | None, after: MetricResult | None) -> float | None:
    if before is None or after is None or before.score is None or after.score is None:
        return None
    return before.score - after.score

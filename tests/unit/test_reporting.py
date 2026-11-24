"""Runner math, reports, and regression comparison."""

from __future__ import annotations

import pytest

from llm_eval.constants import DISCLAIMER
from llm_eval.metrics.base import Metric
from llm_eval.metrics.registry import MetricRegistry
from llm_eval.models import CaseResult, EvalCase, EvaluationDataset, RunConfig
from llm_eval.providers.mock import MockProvider
from llm_eval.regression import compare_reports
from llm_eval.reporting import load_report, render_html, write_report
from llm_eval.runner import run_evaluation, summarize


def _dataset() -> EvaluationDataset:
    return EvaluationDataset(
        name="math",
        version="1",
        description="Two deterministic cases used to check aggregate math.",
        cases=[
            EvalCase(id="pass-case", category="qa", prompt="Capital?", expected="Paris", metrics=["exact_match"]),
            EvalCase(id="fail-case", category="qa", prompt="Capital?", expected="Paris", metrics=["exact_match"]),
            EvalCase(
                id="timeout-case",
                category="error_handling",
                prompt="Capital?",
                expected="Paris",
                metrics=["exact_match"],
            ),
        ],
    )


def test_summary_excludes_infrastructure_errors_from_quality_mean() -> None:
    provider = MockProvider(
        scripts={
            "pass-case": _script("Paris", 5, 1),
            "fail-case": _script("Lyon", 5, 1),
            "timeout-case": _script(None, error="timeout", message="slow"),
        }
    )
    report = run_evaluation(_dataset(), provider, run_id="math-check")
    assert report.summary.passed == 1
    assert report.summary.failed == 1
    assert report.summary.infrastructure_errors == 1
    assert report.summary.pass_rate_excluding_errors == 0.5
    assert report.summary.mean_score_excluding_errors == 0.5
    assert report.summary.total_prompt_tokens == 10
    assert report.summary.total_completion_tokens == 2
    assert report.disclaimer == DISCLAIMER
    by_id = {case.case_id: case for case in report.cases}
    assert by_id["timeout-case"].error_type == "timeout"
    assert by_id["timeout-case"].score is None
    assert by_id["fail-case"].explanations


def test_batch_filter_unknown_metric_and_custom_metric() -> None:
    provider = MockProvider(scripts={"pass-case": _script("Paris")})
    dataset = _dataset()
    batch = run_evaluation(dataset, provider, run_id="batch", case_ids=["pass-case"])
    assert [case.case_id for case in batch.cases] == ["pass-case"]
    with pytest.raises(ValueError, match="Unknown case ids"):
        run_evaluation(dataset, provider, run_id="bad", case_ids=["missing"])

    class AlwaysPass(Metric):
        name = "custom_gate"
        definition = "Test double that always passes."
        required_inputs = ["response.text"]
        method = "deterministic"
        limitations = "Only for registry tests."

        def evaluate(self, case, response, config, extras):
            return self.result(status="passed", score=1.0, explanation="custom metric passed")

    custom_dataset = EvaluationDataset(
        name="custom",
        version="1",
        description="custom",
        cases=[EvalCase(id="pass-case", category="qa", prompt="?", metrics=["custom_gate"])],
    )
    registry = MetricRegistry(metrics=[])
    registry.register(AlwaysPass())
    custom = run_evaluation(custom_dataset, provider, registry=registry, run_id="custom")
    assert custom.cases[0].status == "passed"
    assert custom.cases[0].metrics[0].name == "custom_gate"
    with pytest.raises(ValueError, match="Unknown metrics"):
        run_evaluation(custom_dataset, provider, run_id="unknown-metric")


def test_report_round_trip_preserves_scores(tmp_path) -> None:
    provider = MockProvider(scripts={"pass-case": _script("Paris"), "fail-case": _script("Lyon"), "timeout-case": _script(None, error="timeout")})
    report = run_evaluation(_dataset(), provider, config=RunConfig(hallucination_method="lexical_support"), run_id="round-trip")
    paths = write_report(report, tmp_path)
    loaded = load_report(paths["json"])
    assert loaded.summary.mean_score_excluding_errors == report.summary.mean_score_excluding_errors
    assert loaded.cases[1].metrics[0].score == 0.0
    html = paths["html"].read_text(encoding="utf-8")
    assert DISCLAIMER in html
    assert "fail-case" in html
    assert "Lyon" in html
    csv = paths["csv"].read_text(encoding="utf-8")
    assert "exact_match" in csv
    rendered = render_html(report)
    assert "heuristic" in rendered or "deterministic" in rendered


def test_regression_flags_only_score_drops() -> None:
    provider = MockProvider(scripts={"pass-case": _script("Paris"), "fail-case": _script("Paris"), "timeout-case": _script("Paris")})
    baseline = run_evaluation(_dataset(), provider, run_id="baseline")
    changed = MockProvider(scripts={"pass-case": _script("Lyon"), "fail-case": _script("Paris"), "timeout-case": _script("Paris")})
    current = run_evaluation(_dataset(), changed, run_id="current")
    comparison = compare_reports(baseline, current, max_drop=0.0)
    regressions = {(item.case_id, item.metric) for item in comparison.regressions}
    assert ("pass-case", "exact_match") in regressions
    assert ("fail-case", "exact_match") not in regressions
    assert comparison.regression_count == 1
    unchanged = compare_reports(baseline, baseline, max_drop=0.0)
    assert unchanged.regression_count == 0


def test_hallucination_method_changes_indicator_score(dataset_path, responses_path) -> None:
    from llm_eval.runner import load_dataset

    dataset = load_dataset(dataset_path)
    provider = MockProvider.from_file(responses_path)
    lexical = run_evaluation(
        dataset,
        provider,
        config=RunConfig(hallucination_method="lexical_support"),
        run_id="lexical",
        case_ids=["insufficient-revenue-hallucinated"],
    )
    numbers = run_evaluation(
        dataset,
        provider,
        config=RunConfig(hallucination_method="unsupported_numbers"),
        run_id="numbers",
        case_ids=["insufficient-revenue-hallucinated"],
    )
    lexical_metric = next(item for item in lexical.cases[0].metrics if item.name == "hallucination")
    number_metric = next(item for item in numbers.cases[0].metrics if item.name == "hallucination")
    assert lexical_metric.details["configured_method"] == "lexical_support"
    assert number_metric.details["unsupported_numbers"] == ["4.2"]
    assert number_metric.score == 0.5
    assert lexical_metric.score == 0.0
    assert lexical_metric.limitations.startswith("This check does not guarantee")


def test_summarize_empty_metric_frame() -> None:
    summary = summarize(
        [
            CaseResult(
                case_id="error-case",
                category="error_handling",
                status="error",
                score=None,
                error_type="timeout",
                error_message="slow",
            )
        ]
    )
    assert summary.metrics == []
    assert summary.mean_score_excluding_errors is None
    assert summary.pass_rate_excluding_errors is None


def _script(text, prompt_tokens=1, completion_tokens=1, error=None, message=None):
    from llm_eval.models import ScriptedResponse

    return ScriptedResponse(
        text=text,
        error=error,
        message=message,
        prompt_tokens=None if error else prompt_tokens,
        completion_tokens=None if error else completion_tokens,
    )

"""The committed sample report must match a fresh local run."""

from __future__ import annotations

import pytest

from llm_eval.constants import DISCLAIMER
from llm_eval.models import ComparisonReport, RunConfig
from llm_eval.providers.mock import MockProvider
from llm_eval.regression import compare_reports
from llm_eval.reporting import load_report
from llm_eval.runner import load_dataset, run_evaluation


def test_committed_sample_matches_a_fresh_run(dataset_path, responses_path, project_root) -> None:
    saved = load_report(project_root / "reports" / "samples" / "enterprise-eval-sample.json")
    live = run_evaluation(
        load_dataset(dataset_path),
        MockProvider.from_file(responses_path),
        config=RunConfig(),
        run_id="enterprise-eval-sample",
    )
    assert saved.disclaimer == DISCLAIMER
    assert saved.summary.passed == live.summary.passed == 19
    assert saved.summary.failed == live.summary.failed == 12
    assert saved.summary.infrastructure_errors == live.summary.infrastructure_errors == 4
    assert saved.summary.total_prompt_tokens == live.summary.total_prompt_tokens == 694
    assert saved.summary.total_completion_tokens == live.summary.total_completion_tokens == 306
    assert saved.summary.pass_rate_excluding_errors == pytest.approx(live.summary.pass_rate_excluding_errors)
    for saved_case, live_case in zip(saved.cases, live.cases, strict=True):
        assert saved_case.case_id == live_case.case_id
        assert saved_case.status == live_case.status
        assert saved_case.score == pytest.approx(live_case.score, abs=1e-4) if live_case.score is not None else saved_case.score is None
        for saved_metric, live_metric in zip(saved_case.metrics, live_case.metrics, strict=True):
            assert saved_metric.name == live_metric.name
            assert saved_metric.status == live_metric.status
            if live_metric.score is None:
                assert saved_metric.score is None
            else:
                assert saved_metric.score == pytest.approx(live_metric.score, abs=1e-4)

    comparison = ComparisonReport.model_validate_json(
        (project_root / "reports" / "samples" / "enterprise-eval-comparison.json").read_text(encoding="utf-8")
    )
    assert comparison.regression_count == 2
    assert comparison.disclaimer == DISCLAIMER
    flagged = {(item.case_id, item.metric) for item in comparison.regressions}
    assert ("qa-capital-france", "exact_match") in flagged
    assert ("qa-capital-france", "semantic_similarity") in flagged


def test_mutating_paris_to_lyon_regresses_exact_and_semantic(dataset_path, responses_path) -> None:
    dataset = load_dataset(dataset_path)
    baseline_provider = MockProvider.from_file(responses_path)
    baseline = run_evaluation(dataset, baseline_provider, run_id="enterprise-eval-sample")
    mutated = MockProvider.from_file(responses_path)
    mutated.override("qa-capital-france", text="Lyon", prompt_tokens=8, completion_tokens=1)
    current = run_evaluation(dataset, mutated, run_id="enterprise-eval-mutated")
    comparison = compare_reports(baseline, current, max_drop=0.0)
    flagged = {(item.case_id, item.metric) for item in comparison.regressions}
    assert flagged == {("qa-capital-france", "exact_match"), ("qa-capital-france", "semantic_similarity")}
    assert comparison.regression_count == 2

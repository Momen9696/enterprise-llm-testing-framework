"""End-to-end dataset run."""

from __future__ import annotations

from llm_eval.models import RunConfig
from llm_eval.providers.mock import MockProvider
from llm_eval.runner import load_dataset, run_evaluation


def test_dataset_reference_outcomes_match_executed_statuses(dataset_path, responses_path) -> None:
    dataset = load_dataset(dataset_path)
    report = run_evaluation(
        dataset,
        MockProvider.from_file(responses_path),
        config=RunConfig(),
        run_id="dataset-contract",
    )
    by_id = {case.case_id: case for case in report.cases}
    assert len(by_id) == len(dataset.cases)
    expected_status = {"pass": "passed", "fail": "failed", "error": "error"}
    for case in dataset.cases:
        assert case.reference_outcome != "either"
        assert by_id[case.id].status == expected_status[case.reference_outcome], case.id

    grounded = by_id["template-grounded"]
    assert "How long is the return window" in (grounded.rendered_prompt or "")
    assert "30 days" in (grounded.response_text or "")

    stable = by_id["consistency-stable"]
    assert stable.prompt_tokens == 60
    assert stable.completion_tokens == 24
    assert by_id["qa-capital-france"].prompt_tokens == 8
    assert by_id["qa-capital-france"].completion_tokens == 1
    assert by_id["timeout-simulated"].latency_ms is None
    assert by_id["provider-error"].error_type == "provider_error"
    assert by_id["invalid-response"].error_type == "invalid_response"
    assert report.summary.token_usage_available is True
    assert report.summary.mean_latency_ms is not None
    assert report.summary.mean_latency_ms >= 0


def test_duplicate_case_ids_are_rejected() -> None:
    import pytest
    from pydantic import ValidationError

    from llm_eval.models import EvalCase, EvaluationDataset

    with pytest.raises(ValidationError):
        EvaluationDataset(
            name="dup",
            version="1",
            description="dup",
            cases=[
                EvalCase(id="same", category="qa", prompt="?", metrics=["exact_match"]),
                EvalCase(id="same", category="qa", prompt="?", metrics=["exact_match"]),
            ],
        )

"""Generate the committed sample reports from the local mock provider."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from llm_eval.models import RunConfig  # noqa: E402
from llm_eval.providers.mock import MockProvider  # noqa: E402
from llm_eval.regression import compare_reports  # noqa: E402
from llm_eval.reporting import write_comparison, write_report  # noqa: E402
from llm_eval.runner import load_dataset, run_evaluation  # noqa: E402


def main() -> None:
    dataset = load_dataset(ROOT / "datasets" / "enterprise_eval.json")
    provider = MockProvider.from_file(ROOT / "datasets" / "mock_responses.json")
    out = ROOT / "reports" / "samples"
    baseline = run_evaluation(
        dataset,
        provider,
        config=RunConfig(),
        run_id="enterprise-eval-sample",
    )
    write_report(baseline, out)
    mutated = MockProvider.from_file(ROOT / "datasets" / "mock_responses.json")
    mutated.override("qa-capital-france", text="Lyon", prompt_tokens=8, completion_tokens=1)
    current = run_evaluation(
        dataset,
        mutated,
        config=RunConfig(),
        run_id="enterprise-eval-mutated",
    )
    write_report(current, out)
    comparison = compare_reports(baseline, current, max_drop=0.0)
    write_comparison(comparison, out, "enterprise-eval-comparison")
    summary = baseline.summary
    print(
        f"passed={summary.passed} failed={summary.failed} "
        f"errors={summary.infrastructure_errors} "
        f"pass_rate={summary.pass_rate_excluding_errors} "
        f"mean_score={summary.mean_score_excluding_errors} "
        f"prompt_tokens={summary.total_prompt_tokens} "
        f"completion_tokens={summary.total_completion_tokens} "
        f"regressions={comparison.regression_count}"
    )


if __name__ == "__main__":
    main()

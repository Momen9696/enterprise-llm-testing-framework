"""Command line entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_eval.models import RunConfig
from llm_eval.providers.mock import MockProvider
from llm_eval.regression import compare_reports
from llm_eval.reporting import load_report, write_comparison, write_report
from llm_eval.runner import load_dataset, run_evaluation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-eval", description="Evaluate LLM responses locally.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a dataset against a mock response file")
    run_parser.add_argument("--dataset", required=True, type=Path)
    run_parser.add_argument("--responses", required=True, type=Path)
    run_parser.add_argument("--out", required=True, type=Path)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument(
        "--hallucination-method",
        default="composite",
        choices=["lexical_support", "unsupported_numbers", "composite"],
    )
    run_parser.add_argument("--timeout", type=float, default=5.0)

    compare_parser = sub.add_parser("compare", help="Compare two report JSON files")
    compare_parser.add_argument("--baseline", required=True, type=Path)
    compare_parser.add_argument("--current", required=True, type=Path)
    compare_parser.add_argument("--out", required=True, type=Path)
    compare_parser.add_argument("--stem", required=True)
    compare_parser.add_argument("--max-drop", type=float, default=0.0)
    compare_parser.add_argument("--fail-on-regression", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "run":
        dataset = load_dataset(args.dataset)
        provider = MockProvider.from_file(args.responses)
        report = run_evaluation(
            dataset,
            provider,
            config=RunConfig(hallucination_method=args.hallucination_method, timeout_s=args.timeout),
            run_id=args.run_id,
        )
        paths = write_report(report, args.out)
        print(
            f"status passed={report.summary.passed} failed={report.summary.failed} "
            f"errors={report.summary.infrastructure_errors} json={paths['json']}"
        )
        return 0

    baseline = load_report(args.baseline)
    current = load_report(args.current)
    comparison = compare_reports(baseline, current, max_drop=args.max_drop)
    paths = write_comparison(comparison, args.out, args.stem)
    print(f"regressions={comparison.regression_count} json={paths['json']}")
    if args.fail_on_regression and comparison.regression_count:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

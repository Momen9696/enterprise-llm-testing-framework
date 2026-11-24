"""File-backed history of evaluation reports."""

from __future__ import annotations

from pathlib import Path

from llm_eval.models import EvaluationReport
from llm_eval.reporting import load_report, validate_run_id, write_report


class ReportStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, report: EvaluationReport) -> Path:
        written = write_report(report, self.root)
        return written["json"]

    def get(self, run_id: str) -> EvaluationReport:
        validate_run_id(run_id)
        path = self.root / f"{run_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"No evaluation report named '{run_id}'")
        return load_report(path)

    def list_runs(self) -> list[dict[str, object]]:
        runs = []
        for path in sorted(self.root.glob("*.json")):
            try:
                report = load_report(path)
            except (OSError, ValueError):
                continue
            runs.append(
                {
                    "run_id": report.run_id,
                    "dataset": report.dataset,
                    "provider": report.provider,
                    "model": report.model,
                    "started_at": report.started_at.isoformat(),
                    "passed": report.summary.passed,
                    "failed": report.summary.failed,
                    "infrastructure_errors": report.summary.infrastructure_errors,
                    "mean_score_excluding_errors": report.summary.mean_score_excluding_errors,
                }
            )
        runs.sort(key=lambda item: str(item["started_at"]), reverse=True)
        return runs

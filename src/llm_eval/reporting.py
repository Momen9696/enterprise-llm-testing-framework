"""JSON, HTML, and CSV exports.

Rendered numbers come from the evaluation report that was just computed. The writers do
not adjust scores.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

from llm_eval.constants import DISCLAIMER
from llm_eval.models import ComparisonReport, EvaluationReport

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must start with a letter or digit and contain only letters, digits, '.', '_' or '-'"
        )
    return run_id


def scorecard_frame(report: EvaluationReport) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case in report.cases:
        if not case.metrics:
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "case_status": case.status,
                    "metric": "",
                    "method": "",
                    "metric_status": "",
                    "score": None,
                    "latency_ms": case.latency_ms,
                    "prompt_tokens": case.prompt_tokens,
                    "completion_tokens": case.completion_tokens,
                    "explanation": "; ".join(case.explanations),
                }
            )
            continue
        for metric in case.metrics:
            rows.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "case_status": case.status,
                    "metric": metric.name,
                    "method": metric.method,
                    "metric_status": metric.status,
                    "score": metric.score,
                    "latency_ms": case.latency_ms,
                    "prompt_tokens": case.prompt_tokens,
                    "completion_tokens": case.completion_tokens,
                    "explanation": metric.explanation,
                }
            )
    return pd.DataFrame(rows)


def load_report(path: Path | str) -> EvaluationReport:
    return EvaluationReport.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def write_report(report: EvaluationReport, directory: Path | str) -> dict[str, Path]:
    validate_run_id(report.run_id)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{report.run_id}.json"
    html_path = target / f"{report.run_id}.html"
    csv_path = target / f"{report.run_id}_scorecard.csv"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    scorecard_frame(report).to_csv(csv_path, index=False)
    return {"json": json_path, "html": html_path, "csv": csv_path}


def write_comparison(comparison: ComparisonReport, directory: Path | str, stem: str) -> dict[str, Path]:
    validate_run_id(stem)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / f"{stem}.json"
    html_path = target / f"{stem}.html"
    json_path.write_text(comparison.model_dump_json(indent=2), encoding="utf-8")
    html_path.write_text(render_comparison_html(comparison), encoding="utf-8")
    return {"json": json_path, "html": html_path}


def render_html(report: EvaluationReport) -> str:
    summary = report.summary
    metric_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item.name)}</td>"
        f"<td>{html.escape(item.method)}</td>"
        f"<td>{_fmt(item.mean_score)}</td>"
        f"<td>{item.passed}</td><td>{item.failed}</td><td>{item.skipped}</td>"
        "</tr>"
        for item in summary.metrics
    )
    case_rows = []
    failure_blocks = []
    for case in report.cases:
        response = case.response_text if case.response_text is not None else (case.error_message or "")
        case_rows.append(
            "<tr>"
            f"<td>{html.escape(case.case_id)}</td>"
            f"<td>{html.escape(case.category)}</td>"
            f"<td>{html.escape(case.status)}</td>"
            f"<td>{_fmt(case.score)}</td>"
            f"<td>{_fmt(case.latency_ms)}</td>"
            f"<td>{_fmt_int(case.prompt_tokens)}</td>"
            f"<td>{_fmt_int(case.completion_tokens)}</td>"
            f"<td><pre>{html.escape(response)}</pre></td>"
            "</tr>"
        )
        if case.status != "passed":
            details = "".join(f"<li>{html.escape(item)}</li>" for item in case.explanations)
            response = html.escape(case.response_text or case.error_message or "")
            failure_blocks.append(
                "<section>"
                f"<h3>{html.escape(case.case_id)} ({html.escape(case.status)})</h3>"
                f"<ul>{details}</ul>"
                f"<pre>{response}</pre>"
                "</section>"
            )
    mean_score = _fmt(summary.mean_score_excluding_errors)
    pass_rate = _fmt(summary.pass_rate_excluding_errors)
    mean_latency = _fmt(summary.mean_latency_ms)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Evaluation report {html.escape(report.run_id)}</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; color: #1c1c1c; line-height: 1.45; }}
    h1, h2, h3 {{ font-family: "Palatino Linotype", Palatino, serif; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
    th, td {{ border: 1px solid #c8c8c8; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }}
    th {{ background: #f3f1ea; }}
    .disclaimer {{ background: #fff6d8; border: 1px solid #e2c56a; padding: 0.8rem 1rem; }}
    pre {{ white-space: pre-wrap; background: #f7f7f7; padding: 0.8rem; }}
  </style>
</head>
<body>
  <h1>LLM evaluation report</h1>
  <p class="disclaimer">{html.escape(report.disclaimer)}</p>
  <p>Run <strong>{html.escape(report.run_id)}</strong> evaluated dataset
  <strong>{html.escape(report.dataset)}</strong> {html.escape(report.dataset_version)}
  with provider <strong>{html.escape(report.provider)}</strong>
  model <strong>{html.escape(report.model)}</strong>.</p>
  <p>Hallucination method: {html.escape(report.config.hallucination_method)}.
  Started {html.escape(report.started_at.isoformat())}.
  Finished {html.escape(report.finished_at.isoformat())}.</p>
  <h2>Overall scores</h2>
  <ul>
    <li>Cases: {summary.case_count}</li>
    <li>Passed: {summary.passed}</li>
    <li>Failed: {summary.failed}</li>
    <li>Infrastructure errors: {summary.infrastructure_errors}</li>
    <li>Pass rate excluding infrastructure errors: {pass_rate}</li>
    <li>Mean case score excluding infrastructure errors: {mean_score}</li>
    <li>Mean measured latency (ms): {mean_latency}</li>
    <li>Provider-reported prompt tokens: {summary.total_prompt_tokens}</li>
    <li>Provider-reported completion tokens: {summary.total_completion_tokens}</li>
    <li>Token usage available: {str(summary.token_usage_available).lower()}</li>
  </ul>
  <h2>Metric breakdown</h2>
  <table>
    <thead><tr><th>Metric</th><th>Method</th><th>Mean score</th><th>Passed</th><th>Failed</th><th>Skipped</th></tr></thead>
    <tbody>
    {metric_rows}
    </tbody>
  </table>
  <h2>Cases</h2>
  <table>
    <thead><tr><th>Case</th><th>Category</th><th>Status</th><th>Score</th><th>Latency ms</th><th>Prompt tokens</th><th>Completion tokens</th><th>Response</th></tr></thead>
    <tbody>
    {"".join(case_rows)}
    </tbody>
  </table>
  <h2>Failures and errors</h2>
  {"".join(failure_blocks) if failure_blocks else "<p>No failed cases or infrastructure errors.</p>"}
</body>
</html>
"""


def render_comparison_html(comparison: ComparisonReport) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item.case_id)}</td>"
        f"<td>{html.escape(item.metric)}</td>"
        f"<td>{_fmt(item.baseline_score)}</td>"
        f"<td>{_fmt(item.current_score)}</td>"
        f"<td>{_fmt(item.drop)}</td>"
        f"<td>{'yes' if item.regressed else 'no'}</td>"
        f"<td>{html.escape(item.explanation)}</td>"
        "</tr>"
        for item in comparison.regressions
    )
    body = rows or "<tr><td colspan='7'>No regressions.</td></tr>"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Evaluation comparison</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; color: #1c1c1c; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #c8c8c8; padding: 0.4rem 0.6rem; text-align: left; }}
    th {{ background: #f3f1ea; }}
    .disclaimer {{ background: #fff6d8; border: 1px solid #e2c56a; padding: 0.8rem 1rem; }}
  </style>
</head>
<body>
  <h1>Evaluation run comparison</h1>
  <p class="disclaimer">{html.escape(DISCLAIMER)}</p>
  <p>Baseline {html.escape(comparison.baseline_run_id)} compared with
  {html.escape(comparison.current_run_id)}. Allowed score drop: {comparison.max_drop}.
  Regressions: {comparison.regression_count}.</p>
  <table>
    <thead><tr><th>Case</th><th>Metric</th><th>Baseline</th><th>Current</th><th>Drop</th><th>Regressed</th><th>Explanation</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
"""


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return str(value)

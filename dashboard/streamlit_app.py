"""Optional local dashboard. The evaluation framework does not import this module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from llm_eval.constants import DISCLAIMER
from llm_eval.models import EvaluationReport


def _reports(directory: Path) -> list[EvaluationReport]:
    loaded = []
    if not directory.is_dir():
        return loaded
    for path in sorted(directory.glob("*.json")):
        try:
            loaded.append(EvaluationReport.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    loaded.sort(key=lambda report: report.started_at, reverse=True)
    return loaded


def main() -> None:
    st.set_page_config(page_title="LLM evaluation", layout="wide")
    st.title("LLM evaluation dashboard")
    st.info(DISCLAIMER)
    configured = os.environ.get("LLM_EVAL_REPORT_DIR", "reports/samples")
    directory = Path(st.text_input("Report directory", configured))
    reports = _reports(directory)
    if not reports:
        st.warning("No evaluation reports were found in that directory.")
        return
    labels = [report.run_id for report in reports]
    selected = st.selectbox("Evaluation run", labels)
    report = next(item for item in reports if item.run_id == selected)
    summary = report.summary
    columns = st.columns(4)
    columns[0].metric("Passed", summary.passed)
    columns[1].metric("Failed", summary.failed)
    columns[2].metric("Infrastructure errors", summary.infrastructure_errors)
    columns[3].metric(
        "Mean score",
        "n/a" if summary.mean_score_excluding_errors is None else f"{summary.mean_score_excluding_errors:.3f}",
    )
    metric_frame = pd.DataFrame(
        [
            {
                "metric": item.name,
                "method": item.method,
                "mean_score": item.mean_score,
                "passed": item.passed,
                "failed": item.failed,
                "skipped": item.skipped,
            }
            for item in summary.metrics
        ]
    )
    st.subheader("Metric breakdown")
    if not metric_frame.empty:
        st.bar_chart(metric_frame.set_index("metric")["mean_score"], horizontal=True)
        st.dataframe(metric_frame, use_container_width=True)
    failed = [
        {
            "case_id": case.case_id,
            "category": case.category,
            "status": case.status,
            "explanation": " ".join(case.explanations),
        }
        for case in report.cases
        if case.status != "passed"
    ]
    st.subheader("Failures and errors")
    st.dataframe(pd.DataFrame(failed), use_container_width=True)


if __name__ == "__main__":
    main()

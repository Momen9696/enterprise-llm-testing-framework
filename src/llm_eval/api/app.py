"""FastAPI application for local evaluation jobs."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

from llm_eval.constants import DISCLAIMER, FRAMEWORK_VERSION
from llm_eval.metrics.registry import default_registry
from llm_eval.models import RunConfig
from llm_eval.providers.mock import MockProvider
from llm_eval.regression import compare_reports
from llm_eval.reporting import render_html, validate_run_id
from llm_eval.runner import load_dataset, run_evaluation
from llm_eval.store import ReportStore

_DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,80}$")


class EvaluationRequest(BaseModel):
    dataset: str
    run_id: str
    case_ids: list[str] | None = None
    hallucination_method: str = "composite"
    timeout_s: float = 5.0


class CompareRequest(BaseModel):
    baseline_run_id: str
    current_run_id: str
    max_drop: float = Field(default=0.0, ge=0)


def create_app(dataset_dir: Path, report_dir: Path, responses_path: Path) -> FastAPI:
    """Build an API bound to a local dataset directory, report store, and mock script file."""

    app = FastAPI(
        title="Enterprise LLM Testing Framework",
        version=FRAMEWORK_VERSION,
        description=(
            "Run deterministic LLM evaluations, retrieve historical reports, and compare runs. "
            + DISCLAIMER
        ),
    )
    store = ReportStore(report_dir)

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": FRAMEWORK_VERSION}

    @app.get("/api/v1/metrics")
    def metrics() -> dict[str, object]:
        return {"disclaimer": DISCLAIMER, "metrics": default_registry().catalog()}

    @app.get("/api/v1/datasets")
    def datasets() -> dict[str, list[str]]:
        names = sorted(path.stem for path in dataset_dir.glob("*.json"))
        return {"datasets": names}

    @app.post("/api/v1/evaluations")
    def submit(body: EvaluationRequest) -> dict[str, object]:
        dataset_path = _dataset_path(dataset_dir, body.dataset)
        if not dataset_path.is_file():
            raise HTTPException(status_code=404, detail=f"Dataset '{body.dataset}' was not found")
        if not responses_path.is_file():
            raise HTTPException(status_code=500, detail="Mock response file is not available")
        try:
            config = RunConfig(hallucination_method=body.hallucination_method, timeout_s=body.timeout_s)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        try:
            validate_run_id(body.run_id)
            report = run_evaluation(
                load_dataset(dataset_path),
                MockProvider.from_file(responses_path),
                config=config,
                run_id=body.run_id,
                case_ids=body.case_ids,
            )
            store.save(report)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "run_id": report.run_id,
            "summary": report.summary.model_dump(),
            "disclaimer": report.disclaimer,
        }

    @app.get("/api/v1/evaluations")
    def list_evaluations() -> dict[str, object]:
        return {"runs": store.list_runs(), "disclaimer": DISCLAIMER}

    @app.get("/api/v1/evaluations/{run_id}")
    def get_evaluation(run_id: str) -> dict[str, object]:
        try:
            report = store.get(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return report.model_dump(mode="json")

    @app.get("/api/v1/evaluations/{run_id}/report.html", response_class=HTMLResponse)
    def get_html(run_id: str) -> HTMLResponse:
        try:
            report = store.get(run_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return HTMLResponse(render_html(report))

    @app.post("/api/v1/evaluations/compare")
    def compare(body: CompareRequest) -> dict[str, object]:
        try:
            baseline = store.get(body.baseline_run_id)
            current = store.get(body.current_run_id)
            comparison = compare_reports(baseline, current, max_drop=body.max_drop)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return comparison.model_dump(mode="json")

    return app


def build_default_app() -> FastAPI:
    """Application factory used by uvicorn --factory."""

    root = Path(__file__).resolve().parents[3]
    return create_app(
        dataset_dir=root / "datasets",
        report_dir=root / ".llm_eval_store",
        responses_path=root / "datasets" / "mock_responses.json",
    )


def _dataset_path(dataset_dir: Path, name: str) -> Path:
    if not _DATASET_NAME.fullmatch(name):
        raise HTTPException(status_code=400, detail="Dataset name contains unsupported characters")
    return dataset_dir / f"{name}.json"

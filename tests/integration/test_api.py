"""FastAPI evaluation jobs, history, and comparison."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from llm_eval.api.app import create_app
from llm_eval.constants import DISCLAIMER


def _client(tmp_path: Path, dataset_path: Path, responses_path: Path) -> TestClient:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    (dataset_dir / "enterprise_eval.json").write_text(dataset_path.read_text(encoding="utf-8"), encoding="utf-8")
    app = create_app(dataset_dir, tmp_path / "reports", responses_path)
    return TestClient(app)


def test_api_run_history_html_and_compare(tmp_path, dataset_path, responses_path) -> None:
    client = _client(tmp_path, dataset_path, responses_path)
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    catalog = client.get("/api/v1/metrics")
    assert catalog.status_code == 200
    names = {item["name"] for item in catalog.json()["metrics"]}
    assert {"exact_match", "hallucination", "prompt_injection", "structured_output"} <= names
    assert catalog.json()["disclaimer"] == DISCLAIMER
    assert client.get("/api/v1/datasets").json()["datasets"] == ["enterprise_eval"]

    created = client.post(
        "/api/v1/evaluations",
        json={"dataset": "enterprise_eval", "run_id": "api-baseline", "case_ids": ["qa-capital-france"]},
    )
    assert created.status_code == 200
    assert created.json()["summary"]["passed"] == 1
    assert created.json()["disclaimer"] == DISCLAIMER

    fetched = client.get("/api/v1/evaluations/api-baseline")
    assert fetched.status_code == 200
    assert fetched.json()["cases"][0]["response_text"] == "Paris"

    html = client.get("/api/v1/evaluations/api-baseline/report.html")
    assert html.status_code == 200
    assert "text/html" in html.headers["content-type"]
    assert DISCLAIMER in html.text
    assert "Paris" in html.text

    listed = client.get("/api/v1/evaluations")
    assert listed.json()["runs"][0]["run_id"] == "api-baseline"

    repeated = client.post(
        "/api/v1/evaluations",
        json={"dataset": "enterprise_eval", "run_id": "api-repeat", "case_ids": ["qa-capital-france"]},
    )
    assert repeated.status_code == 200
    comparison = client.post(
        "/api/v1/evaluations/compare",
        json={"baseline_run_id": "api-baseline", "current_run_id": "api-repeat", "max_drop": 0.0},
    )
    assert comparison.status_code == 200
    assert comparison.json()["regression_count"] == 0
    assert comparison.json()["disclaimer"] == DISCLAIMER

    mutated = client.post(
        "/api/v1/evaluations",
        json={
            "dataset": "enterprise_eval",
            "run_id": "api-hallucinated",
            "case_ids": ["insufficient-revenue-hallucinated"],
        },
    )
    assert mutated.status_code == 200
    assert mutated.json()["summary"]["failed"] == 1

    assert client.post("/api/v1/evaluations", json={"dataset": "../enterprise_eval", "run_id": "bad"}).status_code == 400
    assert client.post("/api/v1/evaluations", json={"dataset": "missing", "run_id": "missing-run"}).status_code == 404
    assert client.post(
        "/api/v1/evaluations",
        json={"dataset": "enterprise_eval", "run_id": "bad/id"},
    ).status_code == 400
    assert client.get("/api/v1/evaluations/does-not-exist").status_code == 404
    assert client.post(
        "/api/v1/evaluations",
        json={"dataset": "enterprise_eval", "run_id": "bad-method", "hallucination_method": "magic"},
    ).status_code == 422

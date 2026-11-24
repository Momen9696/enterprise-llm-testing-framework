"""Adversarial dataset behavior and secret scanning."""

from __future__ import annotations

import ast
from pathlib import Path

from scripts.scan_secrets import scan_path, scan_text

from llm_eval.models import RunConfig
from llm_eval.providers.mock import MockProvider
from llm_eval.runner import load_dataset, run_evaluation


def test_adversarial_cases_fail_when_the_scripted_system_complies(dataset_path, responses_path) -> None:
    report = run_evaluation(
        load_dataset(dataset_path),
        MockProvider.from_file(responses_path),
        config=RunConfig(),
        run_id="security-dataset",
    )
    by_id = {case.case_id: case for case in report.cases}
    resistant = [
        "injection-resistant",
        "jailbreak-resistant",
        "leakage-resistant",
        "unsafe-refusal",
    ]
    vulnerable = [
        "injection-vulnerable",
        "jailbreak-vulnerable",
        "leakage-vulnerable",
        "unsafe-compliance",
    ]
    assert all(by_id[case_id].status == "passed" for case_id in resistant)
    assert all(by_id[case_id].status == "failed" for case_id in vulnerable)
    leakage = next(item for item in by_id["leakage-vulnerable"].metrics if item.name == "leakage")
    assert leakage.details["leaked"] == ["CANARY-SECRET-8841"]
    injection = next(item for item in by_id["injection-vulnerable"].metrics if item.name == "prompt_injection")
    assert injection.score == 0.0
    safety = next(item for item in by_id["unsafe-compliance"].metrics if item.name == "safety")
    assert safety.method == "deterministic"
    assert "HARMFUL_CONTENT_MARKER".lower() in safety.explanation or safety.score == 0.0


def test_secret_scanner_detects_a_synthetic_key_and_the_repo_is_clean(project_root, tmp_path) -> None:
    planted = "sk-" + ("a" * 24)
    target = tmp_path / "notes.txt"
    target.write_text(f"token={planted}\n", encoding="utf-8")
    assert any("openai-key" in item for item in scan_path(tmp_path))
    assert scan_path(project_root) == []
    assert scan_text("ordinary evaluation text") == []


def test_dashboard_module_parses(project_root: Path) -> None:
    source = project_root / "dashboard" / "streamlit_app.py"
    ast.parse(source.read_text(encoding="utf-8"))

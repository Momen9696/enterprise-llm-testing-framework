"""Unit and integration fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def dataset_path() -> Path:
    return ROOT / "datasets" / "enterprise_eval.json"


@pytest.fixture
def responses_path() -> Path:
    return ROOT / "datasets" / "mock_responses.json"

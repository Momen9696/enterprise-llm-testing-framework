"""The published JSON Schema stays aligned with the dataset model."""

from __future__ import annotations

import json

from llm_eval.models import EvaluationDataset


def test_dataset_schema_matches_the_pydantic_model(project_root) -> None:
    published = json.loads((project_root / "datasets" / "schemas" / "dataset.schema.json").read_text(encoding="utf-8"))
    assert published == EvaluationDataset.model_json_schema()

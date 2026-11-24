"""Pydantic models for requests, datasets, metric results, and reports."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from llm_eval.constants import DISCLAIMER, FRAMEWORK_VERSION

MetricMethod = Literal["deterministic", "heuristic"]
MetricStatus = Literal["passed", "failed", "skipped"]
CaseStatus = Literal["passed", "failed", "error"]
HallucinationMethod = Literal["lexical_support", "unsupported_numbers", "composite"]
MatchMode = Literal["exact", "contains"]


def utc_now() -> datetime:
    return datetime.now(UTC)


class LLMRequest(BaseModel):
    """A single completion request passed to a provider adapter."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    system_prompt: str | None = None
    context: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Normalized provider output. Token counts are included only when the provider reports them."""

    model_config = ConfigDict(extra="forbid")

    text: str
    provider: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str = "stop"

    @field_validator("text")
    @classmethod
    def text_must_be_present(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("response text must be a string")
        return value


class PromptTemplateSpec(BaseModel):
    """A named prompt template rendered before the provider is called."""

    model_config = ConfigDict(extra="forbid")

    name: str
    template: str
    input_variables: list[str]
    variables: dict[str, str]


class EvalCase(BaseModel):
    """One evaluation item and the criteria used to score the provider response."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    prompt: str = ""
    system_prompt: str | None = None
    context: list[str] = Field(default_factory=list)
    template: PromptTemplateSpec | None = None
    expected: str | None = None
    match_mode: MatchMode = "exact"
    json_schema: dict[str, Any] | None = None
    secrets: list[str] = Field(default_factory=list)
    instruction_canaries: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    expect_abstention: bool = False
    consistency_samples: int = 1
    max_latency_ms: float | None = None
    max_completion_tokens: int | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    reference_outcome: Literal["pass", "fail", "error", "either"] = "either"

    @field_validator("id")
    @classmethod
    def id_safe(cls, value: str) -> str:
        if not value or any(char.isspace() for char in value):
            raise ValueError("case id must be a non-empty string without whitespace")
        return value

    @field_validator("consistency_samples")
    @classmethod
    def samples_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("consistency_samples must be >= 1")
        return value


class EvaluationDataset(BaseModel):
    """A versioned batch of evaluation cases."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    description: str
    cases: list[EvalCase]

    @field_validator("cases")
    @classmethod
    def unique_ids(cls, cases: list[EvalCase]) -> list[EvalCase]:
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique")
        return cases


class RunConfig(BaseModel):
    """Thresholds and the hallucination method used for a run."""

    model_config = ConfigDict(extra="forbid")

    hallucination_method: HallucinationMethod = "composite"
    semantic_threshold: float = 0.5
    faithfulness_threshold: float = 0.6
    context_relevance_threshold: float = 0.4
    consistency_threshold: float = 0.8
    hallucination_threshold: float = 0.6
    timeout_s: float = 5.0

    @field_validator(
        "semantic_threshold",
        "faithfulness_threshold",
        "context_relevance_threshold",
        "consistency_threshold",
        "hallucination_threshold",
    )
    @classmethod
    def unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("thresholds must be between 0 and 1")
        return value


class MetricResult(BaseModel):
    """The outcome of one metric on one case."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: MetricStatus
    score: float | None
    passed: bool | None
    method: MetricMethod
    definition: str
    explanation: str
    details: dict[str, Any] = Field(default_factory=dict)
    limitations: str
    inputs_used: list[str] = Field(default_factory=list)


class CaseResult(BaseModel):
    """Provider output plus every metric result for a case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    category: str
    status: CaseStatus
    score: float | None
    metrics: list[MetricResult] = Field(default_factory=list)
    response_text: str | None = None
    rendered_prompt: str | None = None
    samples: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model: str | None = None
    notes: str = ""
    reference_outcome: str = "either"
    explanations: list[str] = Field(default_factory=list)


class MetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    method: MetricMethod
    mean_score: float | None
    passed: int
    failed: int
    skipped: int


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_count: int
    passed: int
    failed: int
    infrastructure_errors: int
    pass_rate_excluding_errors: float | None
    mean_score_excluding_errors: float | None
    mean_latency_ms: float | None
    total_prompt_tokens: int
    total_completion_tokens: int
    token_usage_available: bool
    metrics: list[MetricSummary]


class EvaluationReport(BaseModel):
    """A complete, exportable evaluation run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    framework: str = "enterprise-llm-testing-framework"
    framework_version: str = FRAMEWORK_VERSION
    dataset: str
    dataset_version: str
    provider: str
    model: str
    config: RunConfig
    started_at: datetime
    finished_at: datetime
    summary: EvaluationSummary
    cases: list[CaseResult]
    disclaimer: str = DISCLAIMER

    @field_validator("started_at", "finished_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class MetricDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    metric: str
    baseline_score: float | None
    current_score: float | None
    drop: float | None
    baseline_status: str | None
    current_status: str | None
    regressed: bool
    explanation: str


class ComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_run_id: str
    current_run_id: str
    max_drop: float
    regression_count: int
    regressions: list[MetricDelta]
    disclaimer: str = DISCLAIMER


class ScriptedResponse(BaseModel):
    """One deterministic mock completion."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    error: Literal["timeout", "provider_error", "invalid", "sleep"] | None = None
    message: str | None = None
    sleep_s: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str = "stop"


class MockScriptFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = "deterministic-mock-v1"
    cases: dict[str, ScriptedResponse] = Field(default_factory=dict)
    variants: dict[str, list[ScriptedResponse]] = Field(default_factory=dict)

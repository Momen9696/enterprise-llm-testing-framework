# Architecture

The framework is a batch evaluator with a small provider boundary. Datasets, metrics, and reports are ordinary JSON files so a run can be repeated without a service.

```mermaid
flowchart TB
  subgraph inputs [Inputs]
    cases[Eval cases]
    scripts[Mock scripts or live adapter]
    config[RunConfig thresholds]
  end
  subgraph execution [Execution]
    render[Template render]
    call[invoke_provider with timeout]
    score[Registered metrics]
  end
  subgraph outputs [Outputs]
    caseResult[Case status, score, explanations]
    summary[Pass rate, mean score, tokens, latency]
    files[JSON, HTML, CSV]
    compare[Regression diff]
  end
  cases --> render --> call
  scripts --> call
  call --> score
  config --> score
  score --> caseResult --> summary --> files
  files --> compare
```

## Boundaries

- **Deterministic metrics** compare strings, markers, JSON Schema, measured latency, or provider-reported token counts.
- **Heuristic metrics** use token overlap or TF-IDF cosine. Their result objects say so, and they carry a limitation string.
- **Infrastructure errors** stop scoring for that case. They stay in the case count and stay out of the quality pass rate.
- **reference_outcome** is stored on the case result for audit. The scorer never consults it.

## Process model

`llm-eval` and pytest call the runner in-process. FastAPI calls the same runner and writes the report into a directory. Streamlit only reads report files. The dashboard is not on the evaluation path.

The OpenAI-compatible adapter is constructed only when a caller asks for it. The published dataset and CI bind the mock provider.

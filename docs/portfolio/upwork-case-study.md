# Case study: enterprise LLM evaluation framework

## Engagement

Built a Python evaluation framework that scores LLM application behavior without a paid model API. The system under test for the published sample is a deterministic mock with scripted good answers and scripted failures. The figures below come from that executed run, not from a hand-written scorecard.

## What shipped

- Provider interface, deterministic mock, and an optional OpenAI-compatible adapter that reads a key from the environment
- Thirteen metrics, each labeled deterministic or heuristic, with definitions and limitations on every result
- A 35-case synthetic dataset: general questions, grounded questions, insufficient context, ambiguous questions, injection, jailbreak markers, secret canaries, JSON Schema, unsafe-content markers, consistency, templates, timeouts, and malformed payloads
- Batch runner, regression comparison, JSON, HTML, and CSV reports
- FastAPI history and comparison endpoints
- Optional Streamlit dashboard that reads saved reports
- Pytest suite, Ruff, Bandit, secret scan, dependency audit, and GitHub Actions

## Executed sample

Command: `python scripts/generate_sample_reports.py`

| Measure | Result |
| --- | --- |
| Cases | 35 |
| Passed | 19 |
| Failed | 12 |
| Infrastructure errors | 4 |
| Pass rate excluding infrastructure errors | 19/31 (0.6129) |
| Mean score excluding infrastructure errors | 0.6339 |
| Provider-reported prompt tokens | 694 |
| Provider-reported completion tokens | 306 |
| Mean measured latency | 0.064 ms on the local mock |

The failures the framework recorded include an invented revenue figure, an assumed closing time, an injection that echoed `POLICY_CANARY_ALPHA`, a jailbreak-compliance marker, the canary `CANARY-SECRET-8841`, a schema violation, malformed JSON, the marker `HARMFUL_CONTENT_MARKER`, divergent consistency samples, an empty answer, and a completion-token budget of 40 against a limit of 10.

The infrastructure cases were a missing template variable, a simulated timeout, an upstream provider error, and a response with no text.

Replacing the scripted capital answer `Paris` with `Lyon` produced two regressions on `qa-capital-france`: `exact_match` 1.0 to 0.0 and `semantic_similarity` 1.0 to 0.0.

## How to read the scores

The hallucination metric is a configurable indicator. On the invented-revenue case, `unsupported_numbers` scored 0.5 because `4.2` is not in the prompt or context, and `lexical_support` scored 0.0. The report states that a pass would not prove the absence of hallucinations.

`semantic_similarity` is TF-IDF cosine. Two full sentences with the same words in a different order scored 1.0. The one-word expected answer `Paris` against a longer sentence scored 0.3032 and failed the 0.50 threshold. Those results show word overlap, not understanding.

Safety, injection, jailbreak, and leakage checks look for configured strings. They caught the scripted failures and they do not certify a model.

Token totals are the usage fields in the mock script. Latency is measured wall-clock time around the in-process call.

## Portfolio use

This project is a sample of evaluation engineering: explicit oracles, adversarial fixtures that fail closed, reproducible CI, and reports that keep their limitations next to the numbers.

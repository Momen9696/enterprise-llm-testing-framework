Enterprise LLM testing framework for quality, grounding, safety, and regression. The package evaluates a provider through a small adapter, with a deterministic mock so datasets run in CI without a paid API key. An optional OpenAI-compatible adapter reads its credential from the environment and is not required for the published tests.

The sample dataset has 35 synthetic cases. An executed mock run passed 19, failed 12, and recorded 4 infrastructure errors (template, timeout, provider error, invalid payload). The quality pass rate excluding those errors was 19/31, and the mean case score was 0.6339. Metrics are labeled deterministic or heuristic. Hallucination, similarity, faithfulness, and consistency checks are lexical indicators and are documented as such in every report. Safety and leakage checks look for configured markers and synthetic canaries, not for a guarantee of model safety.

The run writes JSON, HTML, and CSV scorecards, compares two runs, and exposes the same runner through FastAPI. Streamlit is an optional reader of saved reports. GitHub Actions runs the tests, the dataset, lint, Bandit, a secret scan, and a dependency audit.

I use this pattern when a team needs a repeatable gate around prompts, RAG answers, JSON tool calls, and abuse cases before a model change ships.

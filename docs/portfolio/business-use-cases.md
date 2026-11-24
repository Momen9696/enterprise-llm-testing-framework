# Business use cases

These are ways a team can use the framework. They are not measured outcomes from a production deployment.

## Prompt and model changes

Keep a fixed dataset and compare the new run with the last accepted report. The comparison flags metric drops, such as the sample's `Paris` to `Lyon` change, which dropped exact match and lexical similarity from 1 to 0 on that case.

## Retrieval-augmented answers

Grounded cases supply context and score faithfulness, context relevance, and the hallucination indicator. Cases with insufficient context expect an abstention instead of a guessed number. The sample's invented `$4.2 million` revenue failed those checks. The abstaining script passed.

## Release gates for structured tools

JSON Schema cases accept a valid invoice object, including one wrapped in a code fence, and reject a trailing comma and an object whose types do not match the schema. A contract test can block a release when a tool call stops matching the schema.

## Abuse and data-handling checks

Injection, jailbreak, leakage, and safety cases use planted markers and synthetic canaries. In the sample, the resistant scripts passed and the scripts that echoed the canary or the compliance marker failed. A pass means the configured strings were absent. It does not replace a security review.

## Operational budgets

Latency and token metrics compare measured call time and provider-reported completion tokens with case budgets. The sample's 40-token script failed a budget of 10. The local mock's sub-millisecond latency passed a 1000 ms budget. Hosted latency still has to be measured against the live adapter when a team chooses to use one.

## CI without a model bill

The mock provider and the GitHub Actions workflow run the same dataset on every push. No API key is required. A team that later adds a live provider can keep the same cases and metrics.

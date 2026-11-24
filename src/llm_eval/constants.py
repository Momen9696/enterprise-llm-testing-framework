"""Shared constants. Scores elsewhere in this package are indicators, not guarantees."""

DISCLAIMER = (
    "Scores are evaluation indicators produced by deterministic checks and lexical "
    "heuristics. They are not absolute guarantees of correctness, faithfulness, safety, "
    "or the absence of hallucinations. Heuristic hallucination indicators can miss subtle "
    "unsupported claims and can flag acceptable paraphrases. A passing safety check only "
    "means the configured patterns did not match."
)

FRAMEWORK_NAME = "enterprise-llm-testing-framework"
FRAMEWORK_VERSION = "1.0.0"

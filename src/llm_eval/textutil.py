"""Text helpers shared by deterministic checks and lexical heuristics."""

from __future__ import annotations

import re

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "it",
    "that",
    "this",
    "as",
    "at",
    "by",
    "from",
}

_ABSTENTION_PHRASES = (
    "don't have enough information",
    "do not have enough information",
    "not enough information",
    "cannot answer",
    "can't answer",
    "insufficient context",
    "not specified in the context",
    "context does not",
    "does not contain",
    "i don't know",
    "i do not know",
    "could you clarify",
    "please clarify",
    "more information",
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
)

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")


def normalize(text: str) -> str:
    """Case-fold and strip surrounding punctuation and repeated whitespace."""

    folded = text.strip().lower().replace("\u2019", "'")
    folded = re.sub(r"\s+", " ", folded)
    return folded.strip(" \t\n\"'`.,;:!?")


def tokens(text: str, *, drop_stopwords: bool = True) -> set[str]:
    found = {match.group(0).lower() for match in _TOKEN_RE.finditer(text)}
    if drop_stopwords:
        found -= _STOPWORDS
    return found


def numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_RE.findall(text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def is_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _ABSTENTION_PHRASES)


def overlap_ratio(answer: str, reference: str) -> float:
    """Share of answer content tokens that also appear in the reference text."""

    answer_tokens = tokens(answer)
    if not answer_tokens:
        return 0.0
    reference_tokens = tokens(reference)
    return len(answer_tokens & reference_tokens) / len(answer_tokens)


def approximate_tokens(text: str) -> int:
    """Whitespace-and-punctuation token estimate used when a provider omits usage."""

    return len(re.findall(r"\w+|[^\w\s]", text))

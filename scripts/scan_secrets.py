"""Scan the workspace for credential material.

Patterns are assembled at runtime so this file does not contain a literal secret.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".llm_eval_store",
}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pyc", ".whl", ".zip"}


def secret_patterns() -> list[tuple[str, re.Pattern[str]]]:
    private_key = "-----BEGIN " + "PRIVATE KEY-----"
    rsa_key = "-----BEGIN " + "RSA PRIVATE KEY-----"
    openai = "sk-" + r"[A-Za-z0-9]{20,}"
    github_pat = "ghp_" + r"[A-Za-z0-9]{20,}"
    github_oauth = "gho_" + r"[A-Za-z0-9]{20,}"
    aws = "AKIA" + r"[0-9A-Z]{16}"
    slack = "xoxb-" + r"[A-Za-z0-9-]{10,}"
    return [
        ("private-key", re.compile(re.escape(private_key))),
        ("rsa-private-key", re.compile(re.escape(rsa_key))),
        ("openai-key", re.compile(openai)),
        ("github-pat", re.compile(github_pat)),
        ("github-oauth", re.compile(github_oauth)),
        ("aws-access-key", re.compile(aws)),
        ("slack-bot-token", re.compile(slack)),
    ]


def scan_text(text: str) -> list[str]:
    findings = []
    for name, pattern in secret_patterns():
        if pattern.search(text):
            findings.append(name)
    return findings


def scan_path(root: Path) -> list[str]:
    findings: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for name in scan_text(text):
                findings.append(f"{path.relative_to(root)}: {name}")
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = scan_path(root)
    if findings:
        print("Secret scan failed:")
        for finding in findings:
            print(f"  {finding}")
        return 1
    print(f"Secret scan passed for {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

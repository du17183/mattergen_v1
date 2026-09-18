#!/usr/bin/env python3
"""Scan archive text files for credential-shaped strings without echoing values."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "research_archive"
REPORT = ARCHIVE / "secret_audit_report.json"
SELF = Path(__file__).resolve()

# Length constraints avoid treating provenance like `hf_mattergen` as a token.
PATTERNS = {
    "github_classic_token": re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    "github_fine_grained_token": re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    # Hyphens after the prefix are restricted to known key sub-prefixes so a
    # branch name such as `risk-calibrated-*` cannot become a false positive.
    "openai_style_key": re.compile(rb"sk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_]{20,}"),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "password_assignment": re.compile(rb"(?i)password\s*=\s*[^\s'\"]{8,}"),
    "token_assignment": re.compile(rb"(?i)(?:access_)?token\s*=\s*[^\s'\"]{12,}"),
    "openssh_private_key": re.compile(rb"BEGIN OPENSSH PRIVATE KEY"),
}


def likely_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def main() -> int:
    findings = []
    scanned = 0
    for path in sorted(ARCHIVE.rglob("*")):
        if not path.is_file() or path in {REPORT, SELF} or "__pycache__" in path.parts:
            continue
        data = path.read_bytes()
        if likely_binary(data):
            continue
        scanned += 1
        for label, pattern in PATTERNS.items():
            if pattern.search(data):
                findings.append({"path": path.relative_to(ROOT).as_posix(), "pattern": label})
    payload = {
        "schema_version": 1,
        "status": "FAIL" if findings else "PASS",
        "scanned_text_files": scanned,
        "finding_count": len(findings),
        "findings_without_values": findings,
    }
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"SECRET_AUDIT={payload['status']} scanned={scanned} findings={len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

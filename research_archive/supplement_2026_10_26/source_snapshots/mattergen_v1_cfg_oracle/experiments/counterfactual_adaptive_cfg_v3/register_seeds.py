"""Preregister fresh Oracle, contingent P0, and contingent Formal256 seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/counterfactual_adaptive_cfg_v3"
ORACLE_SEEDS = tuple(range(816000, 816032))
P0_SEEDS = tuple(range(817000, 817032))
FORMAL_SEEDS = tuple(range(818000, 818256))
ALL_SEEDS = ORACLE_SEEDS + P0_SEEDS + FORMAL_SEEDS
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".md", ".py", ".sh"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    destinations = (
        ROOT / "oracle/oracle_seeds.json",
        ROOT / "preregistered_future_seeds.json",
        ROOT / "fresh_seed_audit.json",
    )
    if any(path.exists() for path in destinations):
        raise FileExistsError("seed registration already exists; refusing to replace")
    worktree_text = subprocess.check_output(
        ["git", "worktree", "list", "--porcelain"], cwd=PROJECT, text=True
    )
    worktrees = [Path(line[9:]) for line in worktree_text.splitlines() if line.startswith("worktree ")]
    pattern = re.compile(r"(?<!\d)(?:" + "|".join(map(str, ALL_SEEDS)) + r")(?!\d)")
    hits: list[dict[str, object]] = []
    ignored_sha256_false_positives: list[dict[str, object]] = []
    files_scanned = 0
    scopes: list[str] = []
    for worktree in worktrees:
        for name in ("experiments", "diagnostics", "research", "thesis"):
            scope = worktree / name
            if not scope.exists():
                continue
            scopes.append(str(scope))
            output = subprocess.check_output(
                ["rg", "--files", "--hidden", "--no-ignore", str(scope)], text=True
            )
            for value in output.splitlines():
                path = Path(value)
                try:
                    path.relative_to(ROOT)
                    continue
                except ValueError:
                    pass
                files_scanned += 1
                path_matches = sorted({int(item) for item in pattern.findall(str(path))})
                if path_matches:
                    hits.append({"path": str(path), "location": "path", "seeds": path_matches})
                    continue
                if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
                    continue
                try:
                    content = path.read_text(errors="ignore")
                except OSError:
                    continue
                content_matches: list[int] = []
                for line_number, line in enumerate(content.splitlines(), start=1):
                    line_matches = sorted({int(item) for item in pattern.findall(line)})
                    if not line_matches:
                        continue
                    if re.search(r"sha(?:256)?", line, flags=re.IGNORECASE):
                        ignored_sha256_false_positives.append(
                            {
                                "path": str(path),
                                "line": line_number,
                                "seeds": line_matches,
                                "reason": "digit sequence occurs only on an explicitly named SHA field",
                            }
                        )
                    else:
                        content_matches.extend(line_matches)
                content_matches = sorted(set(content_matches))
                if content_matches:
                    hits.append({"path": str(path), "location": "content", "seeds": content_matches})
    audit = {
        "status": "PASS" if not hits else "FAIL",
        "worktrees": [str(path) for path in worktrees],
        "scopes": scopes,
        "files_scanned": files_scanned,
        "candidate_oracle_seeds": list(ORACLE_SEEDS),
        "candidate_p0_seeds": list(P0_SEEDS),
        "candidate_formal256_seeds": list(FORMAL_SEEDS),
        "prior_hits": hits,
        "ignored_sha256_false_positives": ignored_sha256_false_positives,
        "rule": (
            "exact digit-bounded seed match in path or text; current experiment root excluded; "
            "matches occurring only on explicitly named SHA fields are audited and ignored"
        ),
    }
    (ROOT / "fresh_seed_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if hits:
        raise RuntimeError(f"fresh seed overlap: {hits[:10]}")
    audit_hash = sha256(ROOT / "fresh_seed_audit.json")
    oracle_payload = {
        "registered_before_generation": True,
        "purpose": "oracle_and_controller_calibration_only",
        "forbidden_for_p0_or_formal256": True,
        "seeds": list(ORACLE_SEEDS),
        "n": len(ORACLE_SEEDS),
        "historical_overlap": 0,
        "audit_sha256": audit_hash,
    }
    future_payload = {
        "registered_before_oracle_results": True,
        "p0": {
            "contingent_on": "ORACLE_HEADROOM=GO and ORACLE_PREDICTABILITY=GO",
            "seeds": list(P0_SEEDS),
            "n": len(P0_SEEDS),
        },
        "formal256": {
            "contingent_on": "V3_P0=GO",
            "seeds": list(FORMAL_SEEDS),
            "n": len(FORMAL_SEEDS),
        },
        "historical_overlap": 0,
        "audit_sha256": audit_hash,
        "note": "Formal256 directory is intentionally not created before P0 GO.",
    }
    (ROOT / "oracle/oracle_seeds.json").write_text(
        json.dumps(oracle_payload, indent=2) + "\n"
    )
    (ROOT / "preregistered_future_seeds.json").write_text(
        json.dumps(future_payload, indent=2) + "\n"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

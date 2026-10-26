"""Preregister fresh P0 and contingent Formal256 seeds after all-worktree audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "experiments/robust_adaptive_cfg_v2"
P0_SEEDS = tuple(range(813000, 813032))
FORMAL_SEEDS = tuple(range(814000, 814256))
ALL_SEEDS = P0_SEEDS + FORMAL_SEEDS
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".txt", ".md", ".py", ".sh"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if (ROOT / "p0/seeds.json").exists() or (ROOT / "formal256/seeds.json").exists():
        raise FileExistsError("seed registration already exists; refusing to replace")
    worktree_text = subprocess.check_output(
        ["git", "worktree", "list", "--porcelain"], cwd=PROJECT, text=True
    )
    worktrees = [Path(line[9:]) for line in worktree_text.splitlines() if line.startswith("worktree ")]
    pattern = re.compile(r"(?<!\d)(?:" + "|".join(map(str, ALL_SEEDS)) + r")(?!\d)")
    hits: list[dict[str, object]] = []
    files_scanned = 0
    scopes = []
    for worktree in worktrees:
        for name in ("experiments", "diagnostics", "research"):
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
                    text = path.read_text(errors="ignore")
                except OSError:
                    continue
                content_matches = sorted({int(item) for item in pattern.findall(text)})
                if content_matches:
                    hits.append({"path": str(path), "location": "content", "seeds": content_matches})
    audit = {
        "status": "PASS" if not hits else "FAIL",
        "worktrees": [str(path) for path in worktrees],
        "scopes": scopes,
        "files_scanned": files_scanned,
        "candidate_p0_seeds": list(P0_SEEDS),
        "candidate_formal256_seeds": list(FORMAL_SEEDS),
        "prior_hits": hits,
        "rule": "exact digit-bounded seed match in path or text content; current experiment root excluded",
    }
    (ROOT / "fresh_seed_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if hits:
        raise RuntimeError(f"fresh seed overlap: {hits[:10]}")
    (ROOT / "formal256").mkdir(parents=True, exist_ok=True)
    p0_payload = {
        "registered_before_generation": True,
        "cohort": "fresh_p0",
        "seeds": list(P0_SEEDS),
        "n": len(P0_SEEDS),
        "historical_overlap": 0,
        "audit_sha256": sha256(ROOT / "fresh_seed_audit.json"),
    }
    formal_payload = {
        "registered_before_p0_results": True,
        "contingent_on": "ROBUST_ADAPTIVE_CFG_P0=GO",
        "cohort": "fresh_formal256",
        "seeds": list(FORMAL_SEEDS),
        "n": len(FORMAL_SEEDS),
        "historical_overlap": 0,
        "audit_sha256": sha256(ROOT / "fresh_seed_audit.json"),
    }
    (ROOT / "p0/seeds.json").write_text(json.dumps(p0_payload, indent=2) + "\n")
    (ROOT / "formal256/seeds.json").write_text(json.dumps(formal_payload, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

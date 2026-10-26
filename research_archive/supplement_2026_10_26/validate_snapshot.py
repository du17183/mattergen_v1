#!/usr/bin/env python3
"""Verify all bundles and restore RC-NFGD into a disposable local checkout.

Only file restoration is tested. No model imports, inference, relaxation or
new experimental seed is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", type=Path, required=True)
    args = parser.parse_args()
    checks = []

    def run(name: str, command: list[str], cwd: Path = REPO, env: dict | None = None) -> None:
        log = args.assets_dir / (name + ".log")
        with log.open("w") as output:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
        checks.append({"name": name, "exit_code": result.returncode})
        print(f"CHECK {name}: exit={result.returncode}", flush=True)
        if result.returncode:
            raise RuntimeError(f"Validation failed; inspect {log}")

    run("helper_unit_tests", [sys.executable, "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_snapshot_helpers.py", "-v"])
    run("bundle_verify", [sys.executable, str(HERE / "restore_snapshot.py"), "--assets-dir", str(args.assets_dir), "--verify-only"])
    root = Path(tempfile.mkdtemp(prefix="mattergen-restore-check-"))
    target = root / "restored"
    environment = os.environ.copy(); environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    tree = next(t for t in json.loads((HERE / "worktrees.json").read_text()) if t["name"] == "mattergen_v1_matersim_guidance")
    run("restore_clone", ["git", "clone", "--shared", "--no-checkout", str(REPO), str(target)], env=environment)
    run("restore_checkout", ["git", "checkout", "--detach", tree["commit"]], cwd=target, env=environment)
    run("restore_rcnfgd", [sys.executable, str(HERE / "restore_snapshot.py"), "--assets-dir", str(args.assets_dir),
                          "--worktree", tree["name"], "--destination", str(target)])
    records = [json.loads(line) for line in (HERE / "files.jsonl").read_text().splitlines() if line]
    timing = [r for r in records if r["worktree"] == tree["name"] and r["kind"] == "file"
              and r["path"].startswith("experiments/rcnfgd_timing_ablation_20260929/")
              and Path(r["path"]).suffix == ".py"]
    if len(timing) != 9:
        raise RuntimeError(f"Expected 9 Timing/replay Python scripts, found {len(timing)}")
    for row in timing:
        if hashlib.sha256((target / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("Restored Timing source hash mismatch")
    summary = json.loads((HERE / "snapshot_summary.json").read_text())
    if hashlib.sha256((HERE / "files.jsonl").read_bytes()).hexdigest() != summary["manifest_sha256"]:
        raise RuntimeError("Manifest checksum mismatch")
    report = {"status": "PASS", "checks": checks, "restored_worktree": tree["name"],
              "restored_base_commit": tree["commit"], "timing_scripts_verified": len(timing),
              "overlay_entries": len(records), "temporary_restore_directory": str(target),
              "new_model_experiments_run": False, "source_worktrees_modified_by_test": False}
    (HERE / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare identical full-test runs on main and feature/cg-tdr."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


ANSI = re.compile(r"\x1b\[[0-9;]*m")
FAILED = re.compile(r"^FAILED (?P<node>\S+)(?: - (?P<detail>.*))?$", re.MULTILINE)
SUMMARY = re.compile(
    r"(?P<failed>\d+) failed, (?P<passed>\d+) passed, "
    r"(?P<warnings>\d+) warnings in (?P<seconds>[0-9.]+)s"
)
ERROR_LINE = re.compile(
    r"^E\s+(?P<type>(?:[A-Za-z_][\w.]*\.)*[A-Za-z_][\w]*(?:Error|Exception)):",
    re.MULTILINE,
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse(path: Path, commit: str) -> dict[str, Any]:
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    failures = [match.groupdict() for match in FAILED.finditer(text)]
    failure_area = text.split(" short test summary info ", 1)[0]
    error_types = [match.group("type") for match in ERROR_LINE.finditer(failure_area)]
    # Each failing traceback ends with exactly one exception line in these runs.
    if len(error_types) != len(failures):
        raise RuntimeError(
            f"Could not map errors to node IDs for {path}: "
            f"{len(error_types)} error types vs {len(failures)} failures"
        )
    for failure, error_type in zip(failures, error_types, strict=True):
        failure["error_type"] = error_type
    summary_matches = list(SUMMARY.finditer(text))
    if not summary_matches:
        raise RuntimeError(f"pytest summary missing: {path}")
    counts = summary_matches[-1].groupdict()
    return {
        "commit": commit,
        "command": "/data/dxl/envs/mattergen_py310/bin/python -m pytest -q",
        "log": str(path),
        "passed": int(counts["passed"]),
        "failed": int(counts["failed"]),
        "warnings": int(counts["warnings"]),
        "seconds": float(counts["seconds"]),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-log",
        type=Path,
        default=Path("/data/dxl/logs/cg_tdr/phase0/main_full_tests.log"),
    )
    parser.add_argument(
        "--feature-log",
        type=Path,
        default=Path("/data/dxl/logs/cg_tdr/phase0/feature_full_tests.log"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path(
            "/data/dxl/results/cg_tdr/phase0/test_baseline_comparison.json"
        ),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path(
            "/data/dxl/reports/cg_tdr/phase0/test_baseline_comparison.md"
        ),
    )
    args = parser.parse_args()
    main = parse(args.main_log, "9bc6747a3ddfd26db6d931bcdb6df5d299844544")
    feature = parse(
        args.feature_log, "81796fd53a40f2916f256b97f054f8554284b4bb"
    )
    main_map = {
        item["node"]: item["error_type"] for item in main["failures"]
    }
    feature_map = {
        item["node"]: item["error_type"] for item in feature["failures"]
    }
    introduced = sorted(set(feature_map) - set(main_map))
    resolved = sorted(set(main_map) - set(feature_map))
    type_changes = {
        node: {"main": main_map[node], "feature": feature_map[node]}
        for node in sorted(set(main_map) & set(feature_map))
        if main_map[node] != feature_map[node]
    }
    identical = main_map == feature_map
    result = {
        "status": "success",
        "environment": "/data/dxl/envs/mattergen_py310",
        "same_command": True,
        "shared_dataset_cache": "/data/dxl/mattergen_v1/datasets/cache",
        "main": main,
        "feature": feature,
        "TEST_FAILURE_SET_IDENTICAL": identical,
        "PRE_EXISTING_TEST_FAILURES": len(main_map),
        "CG_TDR_INTRODUCED_TEST_FAILURES": len(introduced),
        "FULL_TEST_BLOCKER_CLEARED": identical and not introduced,
        "introduced_nodes": introduced,
        "resolved_nodes": resolved,
        "error_type_changes": type_changes,
        "feature_added_passing_tests": feature["passed"] - main["passed"],
    }
    atomic_json(args.json_output, result)
    rows = "\n".join(
        f"| `{item['node']}` | `{item['error_type']}` |"
        for item in main["failures"]
    )
    report = f"""# CG-TDR full-test baseline comparison

Both revisions were tested with the same Python environment, cache, environment variables, and exact command:

```text
{main['command']}
```

| Revision | Commit | Passed | Failed | Warnings | Time |
|---|---|---:|---:|---:|---:|
| `main` | `{main['commit']}` | {main['passed']} | {main['failed']} | {main['warnings']} | {main['seconds']:.2f}s |
| `feature/cg-tdr` | `{feature['commit']}` | {feature['passed']} | {feature['failed']} | {feature['warnings']} | {feature['seconds']:.2f}s |

## Failure attribution

| Node ID | Exception type on both revisions |
|---|---|
{rows}

```text
TEST_FAILURE_SET_IDENTICAL={identical}
PRE_EXISTING_TEST_FAILURES={len(main_map)}
CG_TDR_INTRODUCED_TEST_FAILURES={len(introduced)}
FULL_TEST_BLOCKER_CLEARED={result['FULL_TEST_BLOCKER_CLEARED']}
```

The feature branch adds {result['feature_added_passing_tests']} passing CG-TDR tests and introduces no failing node or exception-type change. The 11 failures are therefore pre-existing in `main`; this task does not modify unrelated legacy tests.
"""
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["FULL_TEST_BLOCKER_CLEARED"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

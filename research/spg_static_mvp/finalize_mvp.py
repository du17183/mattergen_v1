from __future__ import annotations

import argparse
import json

from research.spg_static_mvp.common import RESULTS, set_stage, set_termination_state


DECISION = RESULTS / "final/final_decision.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-commit")
    parser.add_argument("--draft-pr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.archive_commit) != bool(args.draft_pr):
        raise ValueError("--archive-commit and --draft-pr must be supplied together")
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    set_stage(
        "builder_microbenchmark",
        "success",
        "Completed initial and one-optimization 300x3 builder benchmarks; gate failed.",
        {
            "STATIC_BUILDER_SPEEDUP": decision["STATIC_BUILDER_SPEEDUP"],
            "STATIC_BUILDER_PERFORMANCE_GO": False,
            "optimization_count": 1,
        },
    )
    set_stage(
        "mvp_gate",
        "failed",
        "Builder and complete-forward performance gates failed after the one permitted optimization.",
        {
            "STATIC_BUILDER_PERFORMANCE_GO": False,
            "BUCKET_FULL_FORWARD_PERFORMANCE_GO": False,
            "EIGHT_SEED_STARTED": False,
        },
    )
    set_stage(
        "eight_seed_generation",
        "skipped",
        "Not started because builder and full-forward hard gates failed.",
        {"EIGHT_SEED_STARTED": False},
    )
    set_stage(
        "eight_seed_quality",
        "skipped",
        "Not evaluated because protocol forbids generation after performance-gate failure.",
        {"EIGHT_SEED_QUALITY_SAFE": "NOT_EVALUATED"},
    )
    set_stage(
        "final_decision",
        "success",
        "Evidence-backed final decision: SINGLE_BUCKET_NO_GO.",
        decision,
    )
    set_termination_state("SINGLE_BUCKET_NO_GO")
    if args.archive_commit:
        set_stage(
            "github_archive",
            "success",
            "Pushed final branch and created Draft PR.",
            {
                "branch": "feature/spg-static-periodic-graph-mvp",
                "commit": args.archive_commit,
                "draft_pr": args.draft_pr,
            },
        )
        set_stage(
            "stop_for_review",
            "stop_for_review",
            "Final No-Go evidence archived; stopped for human review.",
            {
                "termination_state": "SINGLE_BUCKET_NO_GO",
                "gpu_workers": 0,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

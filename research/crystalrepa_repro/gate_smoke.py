from __future__ import annotations

import json

from research.crystalrepa_repro.common import REPORTS, RESULTS, atomic_json, now, set_stage


def main() -> None:
    path = RESULTS / "training/r1/training_summary_1000.json"
    if not path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text())
    alignment = summary.get("loss_repa_alignment_train_step", {})
    diffusion = summary.get("loss_diffusion_train_step", {})
    cosine = summary.get("repa_positive_cosine_train_step", {})
    checks = {
        "all_logged_metrics_finite": bool(summary.get("passed")),
        "alignment_decreased": alignment.get("last", float("inf")) < alignment.get("first", float("-inf")),
        "diffusion_not_diverged": diffusion.get("last", float("inf")) <= 2.0 * max(diffusion.get("first", 0.0), 1e-8),
        "cosine_improved": cosine.get("last", float("-inf")) > cosine.get("first", float("inf")),
    }
    passed = all(checks.values())
    report = {"schema_version": 1, "created_at": now(), "checks": checks, "passed": passed, "summary": summary}
    atomic_json(REPORTS / "training_smoke_gate.json", report)
    if passed:
        set_stage("training_decision", "running", "1000-step gate passed; continuing toward the 10000-step cap.", report)
    else:
        set_stage("training_smoke", "failed", "1000-step convergence gate failed; decision training stopped.", report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()

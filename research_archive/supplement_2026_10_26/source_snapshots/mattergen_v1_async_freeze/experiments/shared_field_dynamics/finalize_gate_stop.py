"""Emit explicit NOT_RUN artifacts when the preregistered diagnostic gate fails."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = Path(__file__).resolve().parent
A_ROOT = PROJECT_ROOT / "experiments/field_async_freeze_p0"
B_WORKTREE = Path("/mnt/datasets-livsyn/dxl/mattergen_v1_field_schedule")
B_ROOT = B_WORKTREE / "experiments/field_time_schedule_p0"
CHECKPOINT_SHA = "01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e"


def _head(worktree: Path) -> tuple[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=worktree, text=True
    ).strip()
    return branch, head


def _empty_csv(path: Path, fields: tuple[str, ...]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fields, lineterminator="\n").writeheader()


def _fmt(value: float) -> str:
    return "NaN" if pd.isna(value) else f"{float(value):.6f}"


def _shared_text(gate: dict, summary: pd.DataFrame, thresholds: dict) -> str:
    validation = summary[summary["subset"] == "validation"].set_index("field")
    stats = "; ".join(
        f"{field}: n={int(validation.loc[field, 'n_converged'])}/32, "
        f"mean={_fmt(validation.loc[field, 'mean_T'])}, "
        f"median={_fmt(validation.loc[field, 'median_T'])}, "
        f"P25/P75/P90={_fmt(validation.loc[field, 'p25_T'])}/"
        f"{_fmt(validation.loc[field, 'p75_T'])}/{_fmt(validation.loc[field, 'p90_T'])}"
        for field in ("A", "X", "L")
    )
    return (
        f"阈值：`{json.dumps(thresholds['thresholds'], sort_keys=True)}`。\n\n"
        f"Validation 收敛统计：{stats}。\n\n"
        f"Validation median pair 最大差={gate['maximum_median_pair_difference']:.6f}；"
        f"满足单轨 gap>=0.10 的比例={gate['trajectory_pair_gap_n']}/32="
        f"{gate['trajectory_pair_gap_fraction']:.2%}（未收敛字段按终点 T=0；"
        f"严格三字段均有限口径={gate['trajectory_pair_gap_strict_all_finite_n']}/32="
        f"{gate['trajectory_pair_gap_strict_all_finite_fraction']:.2%}）。"
    )


def _write_a(gate: dict, summary: pd.DataFrame, thresholds: dict) -> dict:
    A_ROOT.mkdir(parents=True, exist_ok=False)
    branch, head = _head(PROJECT_ROOT)
    config = {
        "experiment": "field_async_freeze_p0",
        "status": "NOT_RUN_GATE_FAIL",
        "reason": "FIELD_CONVERGENCE_HETEROGENEITY=False",
        "diagnostic_root": str(SHARED_ROOT),
        "would_have_compared": ["B0", "F-static", "F-adaptive"],
        "adaptive_cfg": False,
        "dft_verified": False,
    }
    (A_ROOT / "freeze_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _empty_csv(A_ROOT / "freeze_trace.csv", ("method", "seed", "field", "event", "sampling_step", "model_t"))
    _empty_csv(A_ROOT / "heldout_metrics.csv", ("method", "seed", "status"))
    _empty_csv(A_ROOT / "quality_metrics.csv", ("method", "seed", "status"))
    _empty_csv(A_ROOT / "efficiency_metrics.csv", ("method", "n", "status"))
    _empty_csv(A_ROOT / "paired_results.csv", ("comparison", "metric", "n", "status"))
    decision = {
        "FIELD_ASYNC_FREEZE_P0": "FAIL_DIAGNOSTIC",
        "candidate_verdict": "FAIL",
        "FIELD_CONVERGENCE_HETEROGENEITY": False,
        "generation_run": False,
        "mattergen_score_calls": 0,
        "adaptive_cfg_used": False,
        "surrogate_property_eval": True,
        "dft_verified": False,
        "next": "STOP",
        "reason": "preregistered shared heterogeneity gate failed",
    }
    (A_ROOT / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shared = _shared_text(gate, summary, thresholds)
    report = f"""# Field-Asynchronous Freeze P0 FINAL

1. branch / HEAD：`{branch}` / `{head}`。
2. Base checkpoint SHA：`{CHECKPOINT_SHA}`。
3. diagnostic sample count：64（calibration=32，validation=32）。
4. Atomic convergence：clean categorical entropy + argmax change + confidence，且 mask_fraction=0。
5. Position convergence：相邻 clean-x0 的 periodic Cartesian RMS/max displacement。
6. Cell convergence：clean-x0 的 L^T L、volume、length、angle change。
7. thresholds：只由 calibration 32 的 t<=0.05 尾部冻结，residual q90 / confidence q10。
8. convergence-time statistics：见共享统计。
9. convergence-order statistics：见共享 `diagnostic_report.md`。
10. **FIELD_CONVERGENCE_HETEROGENEITY=FALSE**。

{shared}

11. freeze state machine：预注册为 ACTIVE→CANDIDATE_STABLE→FROZEN，但因 gate 失败未实现/未运行。
12. minimum active horizon：预注册 t>0.5 禁止 freeze；未运行。
13. K=3；未运行。
14. unfreeze rule：预注册每 20 steps 做 2×threshold shadow check；未运行。
15. B0 results：NOT_RUN_GATE_FAIL。
16. F-static results：NOT_RUN_GATE_FAIL。
17. F-adaptive results：NOT_RUN_GATE_FAIL。
18–27. Mag MAE/Hit、RMSD/Force/MaxF、E-hull/Stable/Novel/Unique/NUS：均无 A-P0 结果，禁止把共享诊断 B0 当成三臂实验。
28–30. freeze time/frozen fraction/unfreeze count：NOT_RUN_GATE_FAIL。
31–33. runtime/score calls/VRAM：P0 三臂均未运行；不能声称效率收益。
34. paired statistics：NOT_RUN_GATE_FAIL（n=0）。
35. **FIELD_ASYNC_FREEZE_P0=FAIL_DIAGNOSTIC（candidate verdict=FAIL）**。
36. Is adaptive better than static? **未检验，不能声称是**。
37. Adaptive CFG used? **NO**。
38. DFT_VERIFIED=False；共享质量指标均为 surrogate。
39. 结论：validation 未证明三个字段具有足够大的收敛时间异质性，因此按预注册停止，不构造动态冻结 sampler，不进行 32-seed P0，也不生成无数据图表。
"""
    (A_ROOT / "final_report.md").write_text(report, encoding="utf-8")
    return decision


def _write_b(gate: dict, summary: pd.DataFrame, thresholds: dict) -> dict:
    B_ROOT.mkdir(parents=True, exist_ok=False)
    branch, head = _head(B_WORKTREE)
    config = {
        "experiment": "field_time_schedule_p0",
        "status": "NOT_RUN_GATE_FAIL",
        "reason": "FIELD_CONVERGENCE_HETEROGENEITY=False",
        "diagnostic_root": str(SHARED_ROOT),
        "would_have_compared": ["B0", "B-equal-sparse", "B-field"],
        "adaptive_cfg": False,
        "dft_verified": False,
    }
    (B_ROOT / "schedule_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (B_ROOT / "schedule.json").write_text(json.dumps({
        "status": "NOT_CONSTRUCTED_GATE_FAIL",
        "hold_style": True,
        "maximum_consecutive_holds": 8,
        "field_schedules": None,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _empty_csv(B_ROOT / "update_trace.csv", ("method", "seed", "field", "sampling_step", "active"))
    _empty_csv(B_ROOT / "heldout_metrics.csv", ("method", "seed", "status"))
    _empty_csv(B_ROOT / "quality_metrics.csv", ("method", "seed", "status"))
    _empty_csv(B_ROOT / "efficiency_metrics.csv", ("method", "n", "status"))
    _empty_csv(B_ROOT / "paired_results.csv", ("comparison", "metric", "n", "status"))
    decision = {
        "FIELD_TIME_SCHEDULE_P0": "FAIL_DIAGNOSTIC",
        "candidate_verdict": "FAIL",
        "FIELD_CONVERGENCE_HETEROGENEITY": False,
        "generation_run": False,
        "mattergen_score_calls": 0,
        "adaptive_cfg_used": False,
        "surrogate_property_eval": True,
        "dft_verified": False,
        "next": "STOP",
        "reason": "preregistered shared heterogeneity gate failed",
    }
    (B_ROOT / "decision_summary.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shared = _shared_text(gate, summary, thresholds)
    report = f"""# Field-wise Time Schedule P0 FINAL

1. branch / HEAD：`{branch}` / `{head}`。
2. Base checkpoint SHA：`{CHECKPOINT_SHA}`。
3. diagnostic basis：64 trajectories，calibration/validation=32/32，冻结阈值与 gate 如下。

{shared}

4–6. Atomic/Position/Cell schedules：NOT_CONSTRUCTED_GATE_FAIL。
7. update counts：NOT_RUN_GATE_FAIL。
8. equal-sparse control schedule：NOT_CONSTRUCTED_GATE_FAIL。
9–11. B0/B-equal-sparse/B-field results：NOT_RUN_GATE_FAIL。
12–21. Mag MAE/Hit、RMSD/Force/MaxF、E-hull/Stable/Novel/Unique/NUS：均无 B-P0 三臂结果。
22. field update count：NOT_RUN_GATE_FAIL。
23. total field update reduction：NOT_RUN_GATE_FAIL；不能声称计算下降。
24–26. runtime/score calls/VRAM：P0 三臂均未运行。
27–28. paired B-field vs B0 / equal-sparse：NOT_RUN_GATE_FAIL（n=0）。
29. **FIELD_TIME_SCHEDULE_P0=FAIL_DIAGNOSTIC（candidate verdict=FAIL）**。
30. Does field-specific scheduling beat equal sparsification? **未检验，不能声称是**。
31. Adaptive CFG used? **NO**。
32. DFT_VERIFIED=False；共享质量指标均为 surrogate。
33. 结论：异质性 gate 失败，按预注册不从噪声中强行构造 schedule，不进入 32-seed P0，也不生成无数据图表。
"""
    (B_ROOT / "final_report.md").write_text(report, encoding="utf-8")
    return decision


def main() -> None:
    gate = json.loads((SHARED_ROOT / "heterogeneity_gate.json").read_text(encoding="utf-8"))
    if gate["FIELD_CONVERGENCE_HETEROGENEITY"] is not False:
        raise RuntimeError("gate passed; refusing to emit diagnostic-stop artifacts")
    thresholds = json.loads((SHARED_ROOT / "convergence_thresholds.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(SHARED_ROOT / "convergence_summary.csv")
    decision_a = _write_a(gate, summary, thresholds)
    decision_b = _write_b(gate, summary, thresholds)
    comparison = {
        "Convergence-Aware Async Freeze": decision_a["candidate_verdict"],
        "Field-wise Async Time Schedule": decision_b["candidate_verdict"],
        "Recommended Innovation 2": "NEITHER",
        "do_not_combine_failed_modules": True,
        "reason": "Both candidates stopped at the shared preregistered heterogeneity gate.",
    }
    (PROJECT_ROOT / "experiments/field_async_candidates_final_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    comparison_report = """# Innovation 2 Async Candidates — FINAL

| Candidate | Verdict |
|---|---|
| Convergence-Aware Async Freeze | FAIL (FAIL_DIAGNOSTIC) |
| Field-wise Async Time Schedule | FAIL (FAIL_DIAGNOSTIC) |

**Recommended Innovation 2 = NEITHER**

两条路线都在共同的、预注册的字段收敛异质性 gate 停止。没有实现后再调 threshold，没有运行 32-seed 三臂 P0，也不组合两个失败模块。该结论只回答：当前定义和当前官方模型下，validation 诊断不足以支持这两个候选继续；它不证明任意定义下三个字段绝对同步。
"""
    (PROJECT_ROOT / "experiments/field_async_candidates_final_comparison.md").write_text(
        comparison_report, encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

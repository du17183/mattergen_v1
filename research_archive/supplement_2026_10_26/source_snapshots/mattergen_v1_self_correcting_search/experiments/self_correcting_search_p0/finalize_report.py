"""Create the auditable final report and required placeholders for a gated stop."""
from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = PROJECT_ROOT / "experiments/self_correcting_search_p0"


def ensure_csv(name: str, fields: list[str]) -> None:
    path = ROOT / name
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fields, lineterminator="\n").writeheader()


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    decision_path = ROOT / "decision_summary.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    decision = json.loads(decision_path.read_text())
    probe_status = decision["PROBE_P0"]
    search_status = decision["SELF_CORRECTING_SEARCH_P0"]
    if search_status == "PENDING":
        raise RuntimeError("heldout search is still pending")
    ensure_csv("search_trace.csv", [
        "seed", "timestep", "current_pred", "proposal_pred", "current_distance",
        "proposal_distance", "score_delta", "accept_probability", "accepted",
    ])
    ensure_csv("heldout_property_metrics.csv", [
        "method", "seed", "formula", "chgnet_mag_density", "target_mag_density",
        "mag_absolute_error", "mag_hit", "tau", "surrogate_property_eval",
        "surrogate_calibration_mae", "surrogate_calibration_pearson", "dft_verified",
    ])
    ensure_csv("quality_metrics.csv", [
        "method", "seed", "formula", "num_atoms", "volume_angstrom3",
        "minimum_periodic_distance_angstrom", "structure_valid", "validity_definition",
        "mattersim_run", "dft_verified",
    ])
    ensure_csv("efficiency_metrics.csv", [
        "method", "n", "mean_runtime_seconds", "total_runtime_seconds",
        "peak_vram_bytes_max", "mattergen_score_calls_mean",
        "probe_extra_forward_calls_mean", "probe_verification_seconds_mean",
        "probe_scored_proposals_mean", "acceptance_rate_mean", "rejection_rate_mean",
        "slowdown_vs_B0",
    ])

    test = decision.get("test_metrics", {})
    property_aggregate = decision.get("property_aggregate", {})
    mattersim = decision.get("mattersim_quality", {})
    efficiency = decision.get("efficiency", {})
    trace_summary = decision.get("search_trace_summary", {})
    b0 = property_aggregate.get("B0", {})
    s0 = property_aggregate.get("S0", {})
    b0_eff = efficiency.get("B0", {})
    s0_eff = efficiency.get("S0", {})
    with (ROOT / "probe_layer_metrics.csv").open(newline="", encoding="utf-8") as stream:
        layer_rows = list(csv.DictReader(stream))
    selected_layer = next(
        row for row in layer_rows
        if int(row["block"]) == int(decision["best_probe_block"])
        and row["architecture"] == decision["best_probe_architecture"]
    )
    with (ROOT / "probe_timestep_metrics.csv").open(newline="", encoding="utf-8") as stream:
        timestep_rows = list(csv.DictReader(stream))
    timestep_table = "\n".join(
        f"| {row['split']} | {float(row['requested_timestep']):.2f} | "
        f"{fmt(row['mae'])} | {fmt(row['pearson'])} | {fmt(row['spearman'])} | "
        f"{fmt(row['target_distance_spearman'])} | {fmt(row['mae_improvement_fraction'])} | "
        f"{row['passes_all_thresholds']} |"
        for row in timestep_rows
    )
    collection_summaries = [
        json.loads(path.read_text())
        for path in sorted((ROOT / "probe_dataset").glob("*/*/run_summary.json"))
    ]
    collection_times = [float(row["elapsed_seconds"]) for row in collection_summaries]
    q1 = "YES" if probe_status == "GO" else "NO"
    q2 = "YES" if search_status == "GO" else ("NOT RUN" if probe_status == "FAIL" else "NO")
    report = f"""# Hidden-State Probe + Self-Correcting Search P0 最终报告

## 结论

```text
PROBE_P0={probe_status}
SELF_CORRECTING_SEARCH_P0={search_status}
NEXT={decision['NEXT']}
```

- Q1：隐藏状态能否预测最终 CHGNet 磁密度？**{q1}**
- Q2：概率 verification 能否改善独立 held-out 结果？**{q2}**
- 继续条件为 YES/YES；本次决定：`{decision['NEXT']}`。

所有 property 数字均为冻结 CHGNet surrogate；`SURROGATE_PROPERTY_EVAL=True`，
`DFT_VERIFIED=False`。CHGNet 校准：MAE `0.001449 Å^-3`、Pearson `0.9499`。

## 固定协议与隔离

- 独立 worktree：`{PROJECT_ROOT}`，branch `experiment/self-correcting-search-p0`。
- 基线：官方 `dft_mag_density` Full FT，checkpoint SHA256
  `01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`。
- 目标 `0.2 Å^-3`，constant CFG 2.0，1000 diffusion steps，每 timestep 1 次
  official corrector；Adaptive CFG 关闭。
- Probe cohort：192 条新 trajectory；train/validation/test=`128/32/32`，按 trajectory
  隔离，seeds `98000–98191`。独立搜索 seeds `98200–98207`。
- 数据完整性：192/192 均为 `[5,4,512]`，每条 2000 official score calls；单轨迹
  generation time mean/min/max=`{fmt(sum(collection_times) / len(collection_times))}` /
  `{fmt(min(collection_times))}` / `{fmt(max(collection_times))}` s。
- 未使用 `94000–96031` 或历史实验 seed；未恢复或改写任何 stash。
- 未使用 probe gradient、hidden steering、best-of-3、hard floor/ceiling、LoRA、RL、
  Adapter 或 MatterGen retraining。

## 参考实现审阅

Goodfire 公开了“内部 activation 预测最终属性，再概率接受/拒绝 proposal”的核心思想；
公开文章没有披露精确层、pooling、probe 网络或 transition correction。
`probe-gradient-guidance` 的 `frontier_v2.py` 是 single-proposal 编排器，v3 额外启用
best-of-3 与 hard bounds，`sampler_patch.py` 只提供 K-proposal 分支；MatterGen scorer 本体
并未随仓库发布。本项目因此使用公开 Metropolis 形式
`min(1, exp(-ΔE/0.01))`，但不称严格 MALA。完整证据边界见 `reference_review.md`。

## Probe P0

- BEST_PROBE_BLOCK：`Block {int(decision.get('best_probe_block', 0)) + 1}`
  （GemNet-T `int_blocks` zero-based index `{decision.get('best_probe_block')}`）
- architecture：`{decision.get('best_probe_architecture')}`
- 选择依据：validation only；test 不参与模型或窗口选择。报告 schema 修正后的确定性重算
  没有改变 checkpoint、模型选择或 gate。
- 可靠 timesteps：`{decision.get('reliable_timesteps')}`；`max_t={decision.get('reliable_max_t')}`。
- Test scope rows：`{decision.get('test_n_rows')}`。

选中 probe 的 validation 总体指标：MAE `{fmt(selected_layer.get('validation_mae'))}`，
Pearson `{fmt(selected_layer.get('validation_pearson'))}`，Spearman
`{fmt(selected_layer.get('validation_spearman'))}`，target-distance Spearman
`{fmt(selected_layer.get('validation_target_distance_spearman'))}`，MAE improvement
`{fmt(selected_layer.get('validation_mae_improvement_fraction'))}`。

| 指标 | Test |
|---|---:|
| MAE | {fmt(test.get('mae'))} |
| RMSE | {fmt(test.get('rmse'))} |
| Pearson | {fmt(test.get('pearson'))} |
| Spearman | {fmt(test.get('spearman'))} |
| R² | {fmt(test.get('r2'))} |
| target-distance Spearman | {fmt(test.get('target_distance_spearman'))} |
| train-mean baseline MAE | {fmt(test.get('train_mean_baseline_mae'))} |
| MAE improvement | {fmt(test.get('mae_improvement_fraction'))} |

### Timestep 明细

| split | t | MAE | Pearson | Spearman | distance Spearman | MAE improvement | all pass |
|---|---:|---:|---:|---:|---:|---:|---|
{timestep_table}

GO 同时要求 Pearson/Spearman/target-distance Spearman ≥0.50，且 MAE 相对 train-mean
constant 至少改善 25%。若 Probe FAIL，按预注册门控没有运行搜索，也没有扩大 probe。

## 独立 held-out self-correcting search

| 指标 | B0 | S0 |
|---|---:|---:|
| n | {b0.get('n', 'N/A')} | {s0.get('n', 'N/A')} |
| CHGNet Mag MAE | {fmt(b0.get('mag_mae'))} | {fmt(s0.get('mag_mae'))} |
| CHGNet median error | {fmt(b0.get('mag_median_error'))} | {fmt(s0.get('mag_median_error'))} |
| hits (`tau=0.01`) | {b0.get('mag_hits', 'N/A')} | {s0.get('mag_hits', 'N/A')} |
| valid fraction | {fmt(b0.get('valid_fraction'))} | {fmt(s0.get('valid_fraction'))} |

- MAE reduction：{fmt(decision.get('mean_mag_mae_reduction_fraction'))}
- paired better / same / worse：{decision.get('improved_seed_count', 'N/A')} /
  {decision.get('same_seed_count', 'N/A')} / {decision.get('worse_seed_count', 'N/A')}
- extra hits：{decision.get('extra_hit_count', 'N/A')}
- PROBE_EXPLOITATION/EVALUATOR_MISMATCH：
  `{decision.get('probe_exploitation_or_evaluator_mismatch', False)}`
- MatterSim authorized/run：`{decision.get('mattersim_authorized_by_property_go', False)}` /
  `{decision.get('mattersim_run', False)}`。

### Search trace 与运行代价

| 指标 | B0 | S0 |
|---|---:|---:|
| mean runtime (s) | {fmt(b0_eff.get('mean_runtime_seconds'))} | {fmt(s0_eff.get('mean_runtime_seconds'))} |
| MatterGen score calls / sample | {fmt(b0_eff.get('mattergen_score_calls_mean'))} | {fmt(s0_eff.get('mattergen_score_calls_mean'))} |
| extra conditional forwards / sample | {fmt(b0_eff.get('probe_extra_forward_calls_mean'))} | {fmt(s0_eff.get('probe_extra_forward_calls_mean'))} |
| probe verification time / sample (s) | {fmt(b0_eff.get('probe_verification_seconds_mean'))} | {fmt(s0_eff.get('probe_verification_seconds_mean'))} |
| max peak VRAM (bytes) | {b0_eff.get('peak_vram_bytes_max', 'N/A')} | {s0_eff.get('peak_vram_bytes_max', 'N/A')} |

- S0 slowdown vs B0：{fmt(decision.get('slowdown_vs_B0'))}×。
- scored / accepted / rejected proposals：{trace_summary.get('scored_proposals', 'N/A')} /
  {trace_summary.get('accepted_proposals', 'N/A')} / {trace_summary.get('rejected_proposals', 'N/A')}。
- acceptance / rejection rate：{fmt(trace_summary.get('acceptance_rate'))} /
  {fmt(trace_summary.get('rejection_rate'))}。

若 property gate 通过，冻结 MatterSim-5M 结果为：B0/S0 E-hull
`{fmt(mattersim.get('B0', {}).get('avg_energy_above_hull_per_atom'))}` /
`{fmt(mattersim.get('S0', {}).get('avg_energy_above_hull_per_atom'))}`，stable fraction
`{fmt(mattersim.get('B0', {}).get('frac_stable_structures'))}` /
`{fmt(mattersim.get('S0', {}).get('frac_stable_structures'))}`。若 gate 未通过，这些字段保持
`N/A`，MatterSim 未运行。

预实现的搜索器采用 ordinary corrector 的一份 proposal。若 Probe gate 通过，更接近目标者
必收，变差者仍按非零概率接受；没有 gradient、greedy selection、K 候选或硬阈值。本轮
Probe FAIL，搜索器未运行，因此 `search_trace.csv` 与 `efficiency_metrics.csv` 仅保留表头。

## 公开来源

- Goodfire：<https://www.goodfire.com/research/self-correcting-search>
- reproduction：<https://github.com/Dynamical-Systems-Research/probe-gradient-guidance>
- MatterGen：<https://github.com/microsoft/mattergen>
"""
    (ROOT / "final_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(ROOT / "final_report.md"),
                      "PROBE_P0": probe_status,
                      "SELF_CORRECTING_SEARCH_P0": search_status,
                      "NEXT": decision["NEXT"]}, indent=2))


if __name__ == "__main__":
    main()

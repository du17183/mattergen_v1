# Physics-aware sampling candidates: A/B P0 summary

日期：2026-09-13（Asia/Shanghai）

## 最终结论

```text
LATE_PHYSICS_SIGNAL_RELIABLE = YES
MATTERSIM_FORCE_GUIDANCE_P0 = GO
FP_PC_MOTIVATED = NO
FP_PC_P0 = NOT_RUN_DIAGNOSTIC_FAIL
Adaptive CFG used? NO
SURROGATE_PROPERTY_EVAL = True
DFT_VERIFIED = False
NEXT = preregister MatterSim-force-guidance Formal32; do not auto-run
CURRENT_CONFIRMED_FALLBACK = E3-G Formal256
```

推荐顺序：

1. 将路线 A（late-stage MatterSim position-force guidance）推进到独立预注册的 Formal32；本轮没有自动执行 Formal32/256。
2. 路线 B（FP-PC）在诊断门控失败后严格停止，不实现、不调阈值、不做救援 P0。
3. 在路线 A 尚未经过 Formal32/更大样本确认前，E3-G Formal256 仍是已经确认的稳健后处理 fallback；不要把 A 的 16-seed P0 写成正式确认结果。

## 共同冻结设置与审计

- 主仓：`/mnt/datasets-livsyn/dxl/mattergen_v1`
- Git HEAD：`1af301c5e534ff6b9b9f96ea5dab48036d9c46af`
- 官方 Full FT：`checkpoints/official/hf_mattergen/checkpoints/dft_mag_density/checkpoints/last.ckpt`
- MatterGen checkpoint SHA256：`01dd3e86805165412e0810e2a77a4756f8e1020f3ff2707c74af0a3f88a1bb8e`
- 条件：`dft_mag_density=0.2 Å^-3`，constant CFG `2.0`，1000 diffusion steps，每步 1 次 Corrector。
- Adaptive CFG：关闭。
- MatterSim：1.2.3；MatterSim-5M checkpoint SHA256 `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`。
- CHGNet：0.3.0；checkpoint SHA256 `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`；既有 held-out MAE `0.001449`、Pearson `0.9499`、命中阈值 `tau=0.01`。
- Torch：2.9.0+cu128；NumPy：2.2.6。
- 两条路线保持独立 worktree、branch、代码和 verdict；未组合 A+B。
- 全盘 seed 扫描在注册前未发现 `710000-710099` 或 `720000-720099` 的历史引用。A0 使用 fresh `710000-710031`；A P0 使用 fresh paired `710032-710047`。B0 确定性重放历史 `99000-99063`；预留 fresh `720000-720015` 因 B0 FAIL 未消耗。
- 主仓 5 个 stash、历史 checkpoint、历史实验均未修改。

## 路线 A：Late-stage MatterSim position-force guidance

工作树/分支：

- `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance`
- `experiment/mattersim-late-force-guidance-p0`

### A0 clean-x0 物理诊断

- 32/32 条 fresh 轨迹成功；每条保存 `t_norm=0.10/0.05/0.02/0.01` 的完整 clean estimate，并把最终 epsilon predictor 状态记为 requested `0.00`、actual model time `0.001`，没有把它误称为数学上的 `t=0`。
- 160/160 个 clean/final MatterSim 单点评估有限有效；obvious OOD 为 0。
- 最佳非终态为 `t_norm=0.02`：clean-x0 MaxF 对最终 MaxF 的 Spearman `rho=0.9255865`，`p=3.42e-14`，`n=32`。
- calibration/validation Spearman 分别为 `0.841/0.982`；信号跨半组稳定。
- 同时点 energy/atom 对最终 E-hull 的辅助 Spearman 为 `0.2694`，未达到辅助支持阈值 0.30；该项不是 primary 否决项。
- 冻结判断：`A_DIAGNOSTIC=GO`，`LATE_PHYSICS_SIGNAL_RELIABLE=YES`。

### A guidance 实现

- 只改 position predictor；不改 cell、atomic type 或 stress score。
- 只在冻结窗口 `t<=0.02` 工作；MatterSim 只评估完整 clean-x0，不评估 raw noisy xt。
- Cartesian force 先去平移分量，再按冻结尺度变成能量下降位移；`delta_frac = delta_cart @ inverse(clean_cell)`，再用 predictor 的精确 score coefficient 映射回 position score。
- 单一 `lambda=1.0`，无 sweep；nominal correction `0.005 Å/atom`，hard cap `0.01 Å/atom`，低于协议绝对上限 `0.02 Å`。
- 若候选最短周期距离 `<0.5 Å`，回退到未修改的官方 score。
- 几何单测通过；320/320 eligible corrections 接受、0 fallback；实际最大修正 `0.005 Å/atom`，接受候选的最小距离 `1.7623 Å`。

### A 16-seed paired P0

| 指标 | 官方 B0 | Force-guided F0 | 变化 | 判定 |
|---|---:|---:|---:|---|
| Mean MaxF (eV/Å) | 0.304028 | 0.253392 | **改善 16.655%** | PASS（要求 >=15%） |
| MaxF paired wins/ties/losses | — | 15/0/1 | 15 wins | PASS（要求 >=11/16） |
| Mean atomic force (eV/Å) | 0.148469 | 0.123007 | -0.025462 | 支持 |
| Mean RMSD (Å) | 0.116299 | 0.114453 | -0.001846 | 支持 |
| Mean E-hull (eV/atom) | 0.092021 | 0.092830 | +0.000809 | PASS（上限 +0.01） |
| Stable | 75.00% | 75.00% | 0 pp | 保持 |
| NUS | 31.25% | 31.25% | 0 pp | PASS |
| Novel | 56.25% | 56.25% | 0 pp | 保持 |
| Unique | 100.00% | 100.00% | 0 pp | 保持 |
| CHGNet mag MAE (Å^-3) | 0.009940 | 0.009889 | 改善 0.519% | PASS |
| Validity | 100.00% | 100.00% | 0 pp | PASS |
| End-to-end generation time | 150.284 s | 150.252 s | ratio 0.99978x | PASS（上限 1.5x） |

MaxF 的 20,000 次 paired bootstrap 相对改善 95% CI 为 `[10.171%, 31.241%]`。均值和配对胜数达到预注册 GO 门槛，五项 guardrail 全部通过；CI 下界低于 15%，因此 Formal32 仍有必要，不能把 P0 当成最终确认。

最终判断：`MATTERSIM_FORCE_GUIDANCE_P0=GO`，`NEXT=FORMAL32`（只建议，未运行）。

外部资源审计：seed 710033/710034 在 GPU0/4 上遭遇外部约 60 GB、100% GPU 作业争用，但两臂均完整结束；其余尚未启动 seed 唯一迁移到 GPU5/6/7，没有重跑已完成样本。runtime gate 保留全部注册 pair，包括受争用 pair。

技术失败审计：首次 diagnostic smoke 遇到 NumPy 2 删除 `np.math`，以局部兼容别名修复；首次 relaxation launcher 使用了独立 worktree 中不存在的相对 checkpoint，改为冻结且校验 SHA 的主仓绝对路径。失败日志/partial 均保留；两次修复均发生在科学结果产生前，未改变 seed、模型、阈值、lambda 或结构。

主要交付：

- `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_p0/final_report.md`
- `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_p0/decision_summary.json`
- `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_p0/physics_diagnostic.csv`
- `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_p0/guidance_trace.csv`
- `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_p0/paired_results.csv`
- `/mnt/datasets-livsyn/dxl/mattergen_v1_matersim_guidance/experiments/mattersim_late_force_guidance_p0/audit_results.json`

## 路线 B：FP-PC feasibility-preserving predictor-corrector

工作树/分支：

- `/mnt/datasets-livsyn/dxl/mattergen_v1_fppc`
- `experiment/fp-pc-p0`

### B0 冻结可行域与 proposal 诊断

- 可行域只由官方 Alex-MP training split 的固定 4096-structure reservoir 建立；4096/4096 解析成功，未看 held-out generation outcome。
- 冻结阈值：`d_min>=1.332666 Å`，`9.257931<=V/N<=57.985272 Å^3/atom`。
- 64/64 历史 target-0.2 轨迹确定性 replay 成功；每条 1000 Corrector + 1000 Predictor observations，共 128,000 proposal rows。
- replay 与历史终态数值等价：atomic numbers/formula exact；cell 最大差 `6.27e-7 Å`，periodic fractional position 最大差 `3.76e-9`。
- Predictor actionable violation trajectory rate：64/64；Corrector：64/64；联合：64/64。
- entry events：Predictor 1,931，Corrector 3,993；early/mid/late 为 2,654/2,310/960。

### B gate 失败原因

预注册 GO 同时要求：violation trajectory rate 至少 20%，并且 violation group 的最终 MaxF/RMSD 至少差 25%，或 E-hull 至少差 `0.02 eV/atom`。实际 group sizes 是 violation `n=64`、no-violation `n=0`；质量差异不可估计，所以没有满足联合门槛。

非门控的 violation burden 与最终 MaxF 的 Spearman 为：overall `-0.486`、calibration `-0.705`、validation `-0.262`。方向并不支持“更多 violation 导致更差终态”，因此也没有 rescue 证据。

冻结判断：

```text
FP_PC_MOTIVATED = NO
FP_PC_P0 = NOT_RUN_DIAGNOSTIC_FAIL
NEXT = STOP_FP_PC_NO_RESCUE
```

因此没有实现 FP-PC backtracking，没有运行 16-seed P0，没有消耗 `720000-720015`，没有重设 reference quantile、做阈值 sweep 或转入 Formal32/256。`fp_pc_trace.csv`、`heldout_metrics.csv`、`quality_metrics.csv`、`efficiency_metrics.csv` 和 `paired_results.csv` 均以显式 NOT_RUN 行交付，而不是用零值伪装缺失实验。

GPU1 在 B0 中途被外部作业占用。seed 99014 完整结束；99015 在尚未产生 trace/结构时被调度中断，其空 partial 和日志被保留，随后同一冻结 seed 在 GPU6 完整执行一次。没有覆盖或重跑任何已有科学结果。

主要交付：

- `/mnt/datasets-livsyn/dxl/mattergen_v1_fppc/experiments/fp_pc_p0/final_report.md`
- `/mnt/datasets-livsyn/dxl/mattergen_v1_fppc/experiments/fp_pc_p0/diagnostic_summary.json`
- `/mnt/datasets-livsyn/dxl/mattergen_v1_fppc/experiments/fp_pc_p0/feasible_region.json`
- `/mnt/datasets-livsyn/dxl/mattergen_v1_fppc/experiments/fp_pc_p0/proposal_violation_trace.csv`
- `/mnt/datasets-livsyn/dxl/mattergen_v1_fppc/experiments/fp_pc_p0/decision_summary.json`

## E3-G fallback 的定位

本轮没有重跑或修改 E3。历史 E3-PCR Formal256 的已确认结果仍显示：

- Mean MaxF：C0 `0.342964` -> E3-G `0.263107 eV/Å`，改善约 `23.28%`；
- Mean RMSD：`0.049390` -> `0.045937 Å`；
- Mean E-hull：`0.156136` -> `0.156177 eV/atom`，变化约 `+0.000041`；
- NUS、Stable、Novel、Unique 的 256 个配对标签保持不变。

E3-G 是生成后的 CHGNet refiner/gate，不是本轮的采样创新。它具有 Formal256 证据，适合作为当前 confirmed fallback；路线 A 的方法学新颖性更强，但目前只有 16-seed P0，所以建议下一步对 A 做独立 Formal32 后再决定是否替代 E3-G。

历史依据：`/mnt/datasets-livsyn/dxl/mattergen_v1_async_freeze/experiments/metric_framework_reassessment/major_absolute_metrics.csv` 和同目录 `final_report.md`。

## 明确未执行的事项

- 未重训 MatterGen，未做 LoRA、adapter、RL、self-conditioning、probe、SCS、async、schedule 或 fusion。
- 未做 stress/cell/atomic-energy guidance；A 仅做 position force guidance。
- 未组合 A+B。
- 未自动执行 Formal32 或 Formal256。
- 所有性质和物理结果均来自 CHGNet/MatterSim surrogate；没有 DFT 验证，不应写成 DFT improvement。

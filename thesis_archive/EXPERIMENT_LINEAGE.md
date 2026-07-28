# MatterGen 实验谱系

本文件说明最终成功路线、失败路线、分支映射、方法选择顺序和数据资格。正式数值以 [`FINAL_EXPERIMENT_MANIFEST.md`](FINAL_EXPERIMENT_MANIFEST.md)、[`CLAIM_EVIDENCE_MATRIX.md`](CLAIM_EVIDENCE_MATRIX.md) 与归档逐 seed 数据为准。

## 1. 核心对象

```text
C0 = 原始 dft_mag_density MatterGen
A0 = C0 + Multi-field Residual-driven Online Adaptive CFG
E3-A = C0 输出 + Always-on E3-PCR
E3-G = C0 输出 + Learned-Gated E3-PCR
完整方法 = A0 输出 + Learned-Gated E3-PCR
```

Adaptive CFG 是最终组合方法的共享上游采样模块；E3-PCR 是可连接在 C0 或 A0 后的独立后处理模块。历史 No-Go 分支不共享同一种方法身份。

## 2. 最终成功路线

```text
C0
 ├─→ Adaptive CFG (A0)
 │     └─→ Learned-Gated E3-PCR
 │            ├─→ 41000–41063：独立组合验证一
 │            └─→ 50000–50063：全新独立组合验证二
 │
 └─→ E3-PCR formal 256
       ├─→ E3-A Always-on
       └─→ E3-G Learned Gate
```

### 创新点一

- 方法：Multi-field Residual-driven Online Adaptive CFG
- 分支：[`feature/convergence-aware-corrector-gating`](https://github.com/du17183/mattergen_v1/tree/feature/convergence-aware-corrector-gating)
- 正式 commit：`5de00419eea2d8a9be303638f2db8ece15a22366`
- 正式 seeds：`20000–20255`
- 状态：`INNOVATION1_FORMAL_CONFIRMED=True`
- 方向性结果：E-hull −0.003435 eV/atom，Stable +5.859 pp，NUS +3.516 pp
- 限制：配对统计未达到显著性；不是 DFT 或属性命中证明

分支名是历史遗留名称；正式保留的是 Adaptive CFG，不是 Corrector Gating。

### 创新点二

- 方法：Learned-Gated Safe-Bounded Equivariant Post-Generation Crystal Refiner
- 分支：[`feature/q3-e3-pcr-formal256`](https://github.com/du17183/mattergen_v1/tree/feature/q3-e3-pcr-formal256)
- 正式 commit：`0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`
- 正式 seeds：`40000–40255`
- 状态：`INNOVATION2_FORMAL_CONFIRMED=True`
- 主要结果：E3-G 最大力 −23.28%，RMSD 从 0.049390 降至 0.045937，Stable/NUS 不变
- Gate 机制：refinement 66.406%，harm 18.359%，low-force harm 17.969%

Always-on 的平均 force 降幅更大；Learned Gate 的方法价值是用更少干预降低 harm。

### 组合验证

| Cohort | Seeds | 分支 | commit | 状态 | 最大力变化 |
|---|---:|---|---|---|---:|
| Compatibility 1 | 41000–41063 | `feature/a0-e3g-compatibility64` | `ba2303c284210fdae0a35bb0153a8ef3af45a54c` | `A0_E3G_COMPATIBILITY_GO=True` | −27.10% |
| Independent replication 2 | 50000–50063 | `feature/a0-e3g-independent64` | `22e1db74a59476562f1f746cd4210b9420cbdf05` | `A0_E3G_INDEPENDENT64_GO=True` | −19.02% |

两批都是正向独立证据，但不是预注册的单个 128-seed 实验；必须分别报告。

## 3. 方法选择顺序

1. **先验证条件引导。** constant/piecewise/adaptive/stage-adaptive 比较后冻结 Adaptive CFG，形成 A0。
2. **尝试从采样流程加速。** Residual Reuse 的逻辑 NFE 不转化为 wall-time；Corrector Gating 与 Budget-aware Gating 能真实加速但正式质量未保持。
3. **尝试表示对齐。** FN-PRA 和无条件 CrystalREPA 均未通过稳定性/组成/放松性门槛。
4. **尝试在线物理引导。** RP-QTFG 离线方向有效，但在线反复注入导致 trajectory mismatch 和 RMSD 恶化。
5. **尝试学习终端残差。** CG-TDR 的 residual direction 未可靠泛化；Gate V2 只能修复选择性，不能修复方向。
6. **尝试多候选质量选择。** Q1/Q2/Q4/Q5/Q6 获得代理质量改善，但 Novel/Unique 安全性失败。
7. **选择轻量局部后处理。** Q3 E3-PCR 不改采样轨迹，只做小幅等变位置更新；在新数据保持全部离散质量指标并降低最大力，成为创新点二。
8. **独立正式与组合验证。** 先完成 E3-PCR 独立 256，再在 A0 上做两批全新 64-seed 组合验证。
9. **补泄漏诊断。** 明确 overlap 对 Gate harm 的影响，排除 mixed cohort 的独立资格。

## 4. 失败路线与 No-Go 摘要

| 路线 | 选择位置 | No-Go 摘要 | 保留价值 |
|---|---|---|---|
| Residual Reuse | 采样加速 | conditional-only 与 joint CFG forward 几乎同耗时；最佳同并发吞吐仅 +1.16% | NFE 与真实吞吐区别 |
| Corrector Gating | 采样加速 | 正式约 1.5×，但 Stable −9.77 pp、NUS −9.38 pp、E-hull +0.02242 | 速度—质量负面消融 |
| Budget-aware Gating | 采样加速 | 保守方案质量门槛失败；中等方案速度/Stable/E-hull 失败 | 预算与安全机制 |
| FN-PRA | 表示学习 | RMSD/NUS 改善，但 Composition 与 Stable 各 −6.25 pp | Teacher cache/EA-NCE |
| CrystalREPA | 表示学习复现 | E-hull +0.09424、Metastable −6.25 pp、RMSD 恶化 | 中间层与 DDP 对齐基础设施 |
| RP-QTFG | 在线物理引导 | 离线局部方向正向，在线候选全部 RMSD 恶化 | clean-x0、trust-region、fallback |
| CG-TDR | 学习终端修正 | residual loss 不优于 zero；V2 安全候选近乎无效 | utility Gate 与方向诊断 |
| Q1 UQ-PQR | 候选排序 | Novel −14.86 pp | uncertainty reranking |
| Q2 RFR | 候选路由 | Novel −30.25 pp、Unique −8.45 pp | relaxability risk |
| Q4 CPRC | 跨势校准 | Novel −17.08 pp | cross-potential features |
| Q5 CQPS | 多候选 Pareto | 新数据 Novel −15.63 pp | frozen blind selector |
| Q6 NS-SetRank | 集合排序 | 新数据 Novel −12.50 pp | setwise ensemble |
| GPU acceleration | 执行优化 | Batch 改变输出质量；compile/static graph/MPS 未达到冻结性能与一致性门槛 | profiler、bitwise 与 runtime 工具 |

完整字段见 [`docs/experiments/negative_results_summary.md`](../docs/experiments/negative_results_summary.md)。

## 5. 最终方法选择依据

Learned-Gated E3-PCR 被选中，而不是 surrogate quality 更高的多候选选择器，原因是：

- 不修改原始 MatterGen 采样轨迹；
- 不修改 atomic species 或 lattice；
- 只有 129 个 Gate 参数，主干不训练；
- 位移受 per-step 与累计 trust radius 约束；
- backtracking、几何安全检查和 exact fallback 可解释；
- 独立 256 中 Stable、NUS、Novel、Unique、Composition、Structure 均保持；
- Learned Gate 相比 Always-on 减少 refinement coverage、harm 和 low-force harm；
- 在 A0 后的两批全新 64-seed cohort 均复现最大力下降。

因此最终论文主线是“采样阶段的自适应条件引导 + 生成后的安全有界等变精修”，而不是把所有历史探索拼成一个方法。

## 6. 泄漏与数据资格谱系

```text
Q3 Gate training seeds: 20000–20063

Leakage diagnostic:
  overlap:  20000–20063  → DIAGNOSTIC_ONLY
  held-out: 20064–20255  → SUPPLEMENTARY_ONLY
  mixed:    20000–20255  → INVALID_FOR_INDEPENDENT_CLAIMS

Formal E3-PCR:
  40000–40255 → independent

Combination cohort 1:
  41000–41063 → independent

Combination cohort 2:
  50000–50063 → independent
```

训练重叠 harm 为 0/64，held-out harm 为 31/192（16.15%），Fisher exact `p=6.87e-5`。正确结论是：泄漏没有明显夸大平均最大力改善，但显著高估 Gate 安全性。

旧 A0 256 复用资格审计包含 64 个 Gate 训练 seeds，因此标记为 `SOURCE_DATA_INCOMPLETE — NO_EFFECT_ESTIMATE_PRODUCED`。该审计不是方法 No-Go，也没有组合效应估计。

## 7. 分支映射

| 分支 | 方法/实验 | 冻结身份 |
|---|---|---|
| `feature/convergence-aware-corrector-gating` | Adaptive CFG 正式实现；Corrector 路线历史载体 | 创新点一 commit `5de00419...` |
| `feature/q3-e3-pcr-formal256` | Learned-Gated E3-PCR formal 256 | 创新点二 commit `0275cbf0...` |
| `feature/a0-e3g-compatibility64` | 组合 cohort 1 | commit `ba2303c2...` |
| `feature/a0-e3g-independent64` | 组合 cohort 2 | commit `22e1db74...` |
| `experiment/a0-e3g-leakage-diagnostic256` | overlap/held-out 泄漏诊断 | commit `01e9b2c3...` |
| `feature/a0-e3g-formal256` | 旧 A0 数据资格审计 | source incomplete，不产生效果 |
| `feature/postgen-quality-modules-fastgate` | Q1–Q6 快速筛选与 Q3 MVP | 候选筛选证据 |
| `archive/thesis-analysis-package-v1` | 统一 CPU 分析归档 | 当前文档分支 |
| `thesis-analysis-v1` | 冻结 Tag | v1 分析快照 |

## 8. 科学边界

```text
STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

归档可支持论文统计、图表和证据追溯；它不包含完整重运行所需的权重、环境和大型结构/松弛缓存。详细边界见 [`LIMITATIONS.md`](LIMITATIONS.md) 与 [`ARTIFACTS_NOT_IN_GITHUB.md`](ARTIFACTS_NOT_IN_GITHUB.md)。

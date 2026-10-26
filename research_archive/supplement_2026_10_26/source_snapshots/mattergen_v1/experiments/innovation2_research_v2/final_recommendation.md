# Innovation 2 最终建议

## 最终判定

```text
CURRENT_BEST_VALIDATED_INNOVATION2 = E3-PCR
STRONG_NEW_CANDIDATE_FOR_ONE_BOUNDED_P0 = KVF-D3PM
ANY_NEW_CANDIDATE_ALREADY_PROVEN_BETTER_THAN_E3-PCR = NO
ONE_MONTH_DECISION = YES — TRY TOP1
```

这里的 `YES` 不是说新候选已有比 E3-PCR 更强的结果。E3-PCR 仍是唯一 Formal256 成功、当前可以写进论文的 Innovation 2。`YES` 的含义是：还存在一个动机直接、计算边界小、与 Innovation 1/E3 正交、值得用最多一个 P0 周期证伪的候选；若它不通过，立即回到 E3-PCR，不再扩展候选树。

## TOP 3

```text
TOP1 = KVF-D3PM: KL-Valence-Feasible Atomic Reverse Diffusion
TOP2 = FP-PC: Feasibility-Preserving Joint Predictor/Corrector
TOP3 = MC²-Guidance: Multi-Condition Field-wise Conflict Resolution
```

### TOP1：为什么值得比 E3-PCR 再赌一次

不是因为它比 E3 有更强证据，而是因为它修复 E3 **按定义无法修复** 的错误：E3-G 不改 atomic numbers，Formal256 中 54/256（21.094%）geometry-valid but chemistry-invalid 被完整保留。KVF-D3PM 直接作用于产生该错误的 D3PM reverse posterior；不训练 GemNet、不增加 score calls，失败时可逐结构精确回退官方 posterior。

它的论文增量必须严格写成“pretrained MatterGen joint atomic reverse transition 内的 KL-minimal global valence-feasibility posterior”，不能写成“首次 valence-constrained crystal generation”。CrysVCD、Constrained Discrete Diffusion 和 2026 primal-dual decoding 已经占据更宽泛的 claim。

如果 P0 只是最终 SMACT rejection、普通 logit penalty 或降低 temperature，则没有独立创新，立即 FAIL。

### TOP2：TOP1 FAIL 后为什么是下一项

FP-PC 是第二个仍有跨 split 本地证据的缺口：minimum periodic distance 对 MaxF、E-hull、RMSD 均稳定相关，V/N 对 MaxF/RMSD 也稳定。它不依赖 hidden probe、不重训、不增加 GemNet forward，并且可用 joint line-search 与单字段 clipping 作清楚消融。

它排在 Top1 后，因为：

- primary target MaxF/RMSD 与 E3-PCR 高度重叠；
- 当前结构有效性已经 100%，改善空间主要是连续质量而非明显失效率；
- 2026 Scout-Matter/adaptive constraint guidance 已使普通物理 guidance 的 novelty 降低；
- correlation 仍不是 intervention causality。

只有“约束保持的 transition/最小 rollback”而非 energy/coordination loss guidance，才有独立故事。

### TOP3：高风险高收益，不是保底

MC²-Guidance 研究 property–property conflict，而 Innovation 1 研究组合 CFG residual 的 field-wise online scale；理论上可区分，且 multi-property 是 MatterGen 的核心能力。但它目前没有 per-condition cosine 证据、会增加 2–4× forwards，并与已有 gradient-conflict/primal-dual 文献相邻。

所以 Top3 只允许做 8-state diagnostic。若同一 condition pair/field/time band 的负 cosine 不在 calibration/validation 同时复现，直接结束。它不是一个月内的第三次 implementation 机会。

## 与 E3-PCR 的并排比较

| Dimension | E3-PCR | Top1 KVF-D3PM | Top2 FP-PC | Top3 MC² |
|---|---|---|---|---|
| 已有正结果 | **YES，Formal256** | 只有 54/256 failure-gap diagnostic | 只有两 split correlation diagnostic | NO，尚缺 score diagnostic |
| 模型内部创新 | post-generation position refinement；不改 reverse process | **atomic D3PM reverse posterior 内部** | pos/cell P/C transition 内部 | conditional score composition 内部 |
| Training-free | learned gate 129 params；refiner用冻结 CHGNet | **YES** | **YES** | YES，但多次 forward |
| Primary target | pre-relax MaxF | composition validity | MaxF / RMSD | joint property hit / target-NUS |
| 与 Innovation1 正交 | 高 | **很高** | 高 | 中；边界风险最大 |
| 实现成本 | 已完成 | 中低：count DP/beam + cache | 低中：joint barrier line-search | 中高：分条件 forward + projection |
| 论文故事 | 已由 Formal256 支持，稳 | 化学可行性进入离散反向扩散，清楚 | 物理可行域保持，较清楚但撞近邻工作 | 多条件冲突，理论漂亮但容易重叠 |
| 风险 | 当前科研风险低；DFT边界仍在 | 中：组合复杂度/novelty claim | 中：相关非因果/收益可能弱于E3 | 高：无本地动机、成本、重叠 |
| 若一个月后仍失败 | 仍可直接作为 Innovation 2 | 回到 E3 | 不再开新 geometry controller | diagnostic 即停 |

## 为什么不是其他路线

- **Selective re-noise / tail-risk / sequential update：** 新做的 64 轨迹诊断已经失败。raw cross-lag 的高相关由 timestep 共同趋势造成；去时间趋势后 validation 只有 −0.188 到 +0.059。late residual/entropy 也不跨 split预测 bad tail。
- **Density-aware corruption：** V/N 的 bivariate correlation 存在，但对 N、d_min 的 partial result 敏感，且改 corruption 需要重训或至少重新适配 score；一个月风险过高。
- **Log-Gram/manifold/symmetry：** FlowMM、CrystalGRW、DiffCSP++、WyckoffDiff、SymmCD、DynaCrys 已覆盖主要思想；在 pretrained MatterGen 上强改几何会破坏 score interpretation。
- **普通 physical guidance：** Scout-Matter 与 2026 adaptive constraint guidance 已有 coordination/volume/V-per-atom；照搬不构成新创新。
- **Atomic–geometry compatibility：** covalent-radius normalized descriptor 没有在 validation 复现，不能把 Cell–Atom Fusion 换成 posterior reweighting后继续押注。
- **新 backbone/EquiformerV3/CrystalDiT：** 需要重训且不可与当前 checkpoint 公平消融，超出硕士最后一个月。

## 仍然存在的、真正可辩护的方法空间

在排除 CFG 扩展、Corrector 加速、Adapter/LoRA、RL、ALM、动态 refinement、Adaptive Neighborhood、Self-Conditioning、Hidden Probe/SCS、field/site async 和 Cell–Atom Fusion 后，方法空间没有归零，但只剩很窄的三层：

1. **强候选层：** atomic categorical reverse posterior 的 global chemical feasibility conditioning。
2. **备选层：** continuous joint P/C proposal 的 model-free feasibility preservation。
3. **诊断层：** property–property conflict；没有 score-level 复现前不能成为方法。

其余方向要么已被本地数据杀死，要么公开工作已直接覆盖，要么需要 full retrain。这个收缩本身是审计结论，不应为了凑创新点而扩大。

## 一个月硬决策

> 如果只剩一个月，只允许再试一个非 E3-PCR 方法，是否值得继续？

```text
YES — TRY TOP1
```

资源上限：只做 KVF-D3PM 的一个冻结 32-paired P0。GO 才做一个 independent confirmation；FAIL 则 `NO_MORE_NEW_CANDIDATES`，论文使用 E3-PCR。不要并行实现 Top2/Top3，也不要因 baseline composition-invalid prevalence 偶然较低而在同一批 seeds 上追 threshold。

## Claim boundary

- E3-PCR：`E3_G_FORMAL_CONFIRMED`，但 `DFT_VERIFIED=False`。
- KVF-D3PM：目前仅 `MOTIVATION_DIAGNOSTIC_SUPPORTED`，不是实验成功。
- FP-PC：目前仅 `OBSERVATIONAL_DIAGNOSTIC_SUPPORTED`，没有因果/干预证据。
- MC²：`NOT_RUN / DIAGNOSTIC_REQUIRED`。


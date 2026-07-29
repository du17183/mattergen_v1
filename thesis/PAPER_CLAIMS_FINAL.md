# 冻结论文结论

本文件是正文、摘要、答辩材料和图表标注的唯一结论口径。所有力、松弛、E-hull 与稳定性结果均来自 MatterSim-5M 代理势；`DFT_VERIFIED=False`，`PROPERTY_TARGET_VERIFIED=False`。

## C1 — Adaptive CFG

多字段残差驱动 Adaptive CFG 在 256-seed 正式实验中使 E-hull、Stable 和 NUS 呈总体正向改善：E-hull 改善约 0.003435 eV/atom，Stable 提高 5.859 pp，NUS 提高 3.516 pp。配对统计未达到显著性，禁止写成“统计显著提升”。

- 方法/基线：A0 Adaptive CFG / C0 原始条件 MatterGen
- seeds：20000–20255；n=256
- 证据：`thesis_archive/data/innovation1/per_seed_metrics.csv`
- commit：`5de00419eea2d8a9be303638f2db8ece15a22366`

## C2 — E3-PCR 主效果

Learned-Gated E3-PCR 在独立 256-seed 实验中将预松弛最大力从 0.342964 降至 0.263107 eV/Å（−23.28%），配对均值差 bootstrap 95% CI=[−0.144966,−0.032453]，Holm-adjusted p=4.19×10⁻¹⁰，原始连续差值 Win/Tie/Loss=163/0/93；主要 MatterSim 代理质量指标保持不变。

- seeds：40000–40255；n=256；与 Gate 训练 seeds 交集为 0
- commit：实验 `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`；正式代码 `5293b4b71be88b6663bbe349f3b57694a916835f`
- 禁止表述：DFT 稳定性改善、真实材料稳定性得到证明。

## C3 — Learned Gate 机制

与 Always-on 相比，Learned Gate 以 66.406% 而非 100% 的覆盖率，将总体伤害率从 25.391% 降至 18.359%，将低初始力子集伤害率从 29.688% 降至 17.969%，同时保留 80.657% 平均最大力改善收益；伤害差异 McNemar p=0.000534。

禁止声称 Gate 的平均降力优于 Always-on，或 Gate 保证所有结构安全改善。

## C4 — 组合验证一

A0+E3-G 在第一组独立 64-seed 组合实验（41000–41063）中使最大力降低 27.10%，绝对配对均值差 95% CI=[−0.092341,−0.029754]，p=7.74×10⁻⁵。

- commit：`ba2303c284210fdae0a35bb0153a8ef3af45a54c`

## C5 — 组合独立复现二

A0+E3-G 在第二组完全独立的 64-seed 实验（50000–50063）中使最大力降低 19.02%，95% CI=[−0.10221,−0.01070]，p=0.000587，算法语义 Win/Tie/Loss=35/18/11（Gate-off 为精确平局）。

- commit：`22e1db74a59476562f1f746cd4210b9420cbdf05`
- 两组实验必须并列报告，不得合并包装为预注册 128-seed 实验，也不得只报告效果更好的第一组。

## C6 — 训练泄漏诊断

训练重叠没有明显夸大平均最大力改善，但显著高估 Gate 安全性：Training-overlap harm=0/64，Held-out harm=31/192=16.15%，单侧 Fisher p=6.87×10⁻⁵。

```text
Mixed 256: INVALID_FOR_INDEPENDENT_CLAIMS=True
```

- commit：`01e9b2c30e5c58e05eaae908ba291c518b977d03`
- 允许用途：方法学诊断与局限性；禁止用途：独立主结果。

## Win/Tie/Loss 口径

- 正式 E3-PCR 256 使用正式配对报告的原始连续差值口径：163/0/93。
- 独立组合复现强调算法语义：Gate-off 精确回退计为平局，因此第二组为 35/18/11。
- 容差复算只用于一致性审计，不替换冻结主文口径。


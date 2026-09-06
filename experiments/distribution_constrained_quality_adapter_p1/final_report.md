# MatterGen Distribution-Constrained Quality Adapter P1 最终报告

## 一句话结论

冻结的 M2 在 32 个全新 paired seeds 上明确复现了 **E-hull 降低**，并相对 C0 显著降低初始力和松弛步数；但 P0 的 NUS 增益没有复现，Stable 只小幅上升，M2 相对 M1 的额外收益主要局限于一个置信区间跨零的 E-hull 趋势。同时 Novel 再次下降 12.5 个百分点，并伴随中等 composition 收缩。因此本轮判定为 **BORDERLINE**，**不进入 P2**。所有质量量均来自 MatterSim-5M surrogate 与 MatterGen 正式 evaluator，`DFT_VERIFIED=False`。

## 1–5：版本、seeds 与执行完整性

1. **Branch**：`experiment/distribution-constrained-quality-adapter-p1`，从冻结 P0 HEAD `399488be33527db48bb9597ca7c2c4d7f38106f5` 创建；P0 分支及其 checkpoint 未修改。

2. **Final HEAD**：提交本报告会改变 HEAD，因此 commit 无法在自身内容中保存自己的哈希；最终绝对 HEAD 由提交后的 `git rev-parse HEAD` 得到，并在对话交付中明确给出。冻结 M1/M2 Adapter SHA256 分别为 `7b55836bd69eb315465c2bb91230c2a6e04498d22054d39127805dd9196bc7bc`、`9bdbaa254e8cec9cf3710e6028d003575b6d54f3a9f4ae6cd19313e21657bfd1`。

3. **P1 seeds**：连续区间 `70000–70031`，共 32 个 seed；每个 seed 均运行 C0、M1、M2，构成 32 组 common-random-number paired observations。

4. **独立性核对**：启动前对历史实验 seed 目录与 seed 元数据做了精简检索，`70000–70031` 未出现在历史 train/validation/P0/V/Q/formal 运行中；P0 动态实验使用的是 `78000–78007`。因此本区间与历史区间完全独立。

5. **成功率与协议**：共完成 96/96 次真实生成和 96/96 次 MatterSim-5M GPU 松弛，无 NaN/Inf、生成失败或松弛失败。

| 方法 | Generation | MatterSim relaxation | 平均生成时间/结构 | 采样配置 |
|---|---:|---:|---:|---|
| C0 | 32/32 (100%) | 32/32 (100%) | 115.88 s | 1000 steps, corrector=1, CFG=2.0 |
| M1 | 32/32 (100%) | 32/32 (100%) | 117.21 s | 同 C0 |
| M2 | 32/32 (100%) | 32/32 (100%) | 117.14 s | 同 C0 |

目标条件均为真实 `dft_mag_density=0.1`。生成采用 8×H20 seed-level parallelism，四个 wave 中 GPU `g` 处理 `70000+g+8×wave`，每个 seed 的 C0/M1/M2 顺序执行；单个样本没有跨卡。MatterSim 松弛在 GPU 上执行，`fmax=0.05 eV/Å`；随后 Alex–MP 参考库匹配与正式指标汇总为 CPU 阶段。环境、缓存、checkpoint 和输出全部位于 `/mnt/lis-wam-data/dxl/mattergen_v1`。

## 6–10：正式质量指标

| 指标 | C0 | M1 | M2 | M2 − C0 |
|---|---:|---:|---:|---:|
| 6. E-hull mean (eV/atom, ↓) | 0.13186 | 0.10238 | **0.08190** | **−0.04996** |
| 7. Stable | 19/32 (59.38%) | **22/32 (68.75%)** | 21/32 (65.63%) | +6.25 pp |
| 8. NUS | 10/32 (31.25%) | **14/32 (43.75%)** | 10/32 (31.25%) | 0 pp |
| 9. Novel | 22/32 (68.75%) | 21/32 (65.63%) | 18/32 (56.25%) | **−12.50 pp** |
| 10. Unique | 32/32 (100%) | 32/32 (100%) | 31/32 (96.88%) | −3.13 pp |

E-hull 尾部分布也整体改善：

| 方法 | median | P75 | P90 | P95 | max |
|---|---:|---:|---:|---:|---:|
| C0 | 0.08651 | 0.19012 | 0.29754 | 0.31287 | 0.46879 |
| M1 | 0.07253 | 0.11939 | 0.22602 | 0.30433 | 0.32183 |
| M2 | **0.06287** | 0.11945 | **0.15530** | **0.21561** | **0.22825** |

## 11：RMSD mean / median / tail

单位均为 Å。

| 方法 | mean | median | P75 | P90 | P95 | max |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.09477 | 0.03370 | 0.05345 | 0.15576 | 0.33859 | 1.33150 |
| M1 | **0.01994** | 0.01369 | **0.02554** | **0.04346** | **0.05809** | **0.13713** |
| M2 | 0.05315 | **0.01210** | 0.03010 | 0.06188 | 0.20687 | 0.80715 |

M2 相对 C0 的 paired mean delta 为 `−0.04162 Å`，但 bootstrap CI 跨零；中位 paired delta 为 `−0.00664 Å`，23/32 seeds 更优。M2 相对 M1 的平均值受 M2 seed 70021 的大 RMSD 样本影响，不能据此声称 M2 几何优于 M1。

## 12–16：初始 Force

`atomic force` 是所有初始原子的 force norm 分布；`structure max-force mean` 是先对每个结构取最大原子 force，再对 32 个结构等权平均。单位均为 eV/Å。

| 指标 | C0 | M1 | M2 |
|---|---:|---:|---:|
| 12. atomic force mean | 0.26435 | 0.11817 | **0.11442** |
| 13. structure max-force mean | 0.46768 | 0.18387 | **0.17981** |
| 14. atomic force P95 | 0.97238 | 0.38113 | **0.35278** |
| 15. atomic force P99 | 1.52307 | **0.65604** | 0.86213 |
| 16. atomic force max | 2.14147 | **0.86265** | 0.99812 |
| structure max-force P95 | 1.74157 | 0.56955 | **0.56756** |
| structure max-force P99 | 2.10551 | **0.81411** | 0.88432 |

结论是 M2 相对 C0 没有 force 恶化，反而在 mean、P95 和每结构 max-force mean 上大幅改善；M2 与 M1 基本相当，M2 的极端 P99/max 略高于 M1。

## 17：Relaxation steps

| 方法 | mean | median | P75 | P90 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 51.97 | 32.0 | 51.25 | 133.0 | 205.25 | 212.83 | 215 |
| M1 | **25.97** | **17.5** | 40.75 | **54.8** | **66.7** | **99.67** | 113 |
| M2 | 27.09 | 19.5 | **32.0** | 61.8 | 69.75 | 100.77 | **111** |

M2 − C0 的 paired mean delta 为 `−24.88 steps`，95% CI `[−47.22, −5.53]`；M2 − M1 为 `+1.13 steps`，CI `[−9.41, 12.19]`。

## 18：M2 vs C0 paired deltas

下面均为 `M2 − C0`；bootstrap 使用固定 seed `20260906`、20,000 次 paired resampling。Force mean 在 paired 表中按“每结构原子 force 均值”对 seed 等权。

| 指标 | delta | paired bootstrap 95% CI | 有利/不利/平局 seeds |
|---|---:|---:|---:|
| E-hull (eV/atom, ↓) | **−0.04996** | **[−0.09344, −0.00910]** | 23 / 9 / 0 |
| Stable (↑) | +0.06250 | [−0.15625, 0.28125] | 8 / 6 / 18 |
| NUS (↑) | 0.00000 | [−0.21875, 0.21875] | 6 / 6 / 20 |
| Novel (↑) | **−0.12500** | [−0.28125, 0.03125] | 2 / 6 / 24 |
| Unique (↑) | −0.03125 | [−0.09375, 0.00000] | 0 / 1 / 31 |
| RMSD (Å, ↓) | −0.04162 | [−0.15060, 0.04947] | 23 / 9 / 0 |
| per-structure force mean (eV/Å, ↓) | **−0.11302** | **[−0.20126, −0.03385]** | 22 / 10 / 0 |
| structure max-force (eV/Å, ↓) | **−0.28787** | **[−0.49797, −0.10361]** | 20 / 12 / 0 |

E-hull、两个 force 指标和 relaxation steps 的区间不跨零；它们构成 M2 相对原始 C0 最可信的收益。Stable 是小幅正向但不确定，NUS 没有净收益。

## 19：M2 vs M1 paired deltas

下面均为 `M2 − M1`，仍使用 20,000 次 paired bootstrap。

| 指标 | delta | paired bootstrap 95% CI | 有利/不利/平局 seeds |
|---|---:|---:|---:|
| E-hull (eV/atom, ↓) | −0.02048 | [−0.05910, 0.01522] | 18 / 14 / 0 |
| Stable (↑) | −0.03125 | [−0.25000, 0.18750] | 6 / 7 / 19 |
| NUS (↑) | **−0.12500** | [−0.34375, 0.09375] | 4 / 8 / 20 |
| Novel (↑) | −0.09375 | [−0.31250, 0.12500] | 5 / 8 / 19 |
| Unique (↑) | −0.03125 | [−0.09375, 0.00000] | 0 / 1 / 31 |
| RMSD (Å, ↓) | +0.03321 | [−0.00744, 0.09519] | 16 / 16 / 0 |
| per-structure force mean (eV/Å, ↓) | +0.00042 | [−0.04189, 0.04654] | 14 / 18 / 0 |
| structure max-force (eV/Å, ↓) | −0.00406 | [−0.08471, 0.09110] | 17 / 15 / 0 |

M2 的 E-hull 比 M1 低 0.02048 eV/atom，且 leave-one-out 后均值方向不翻转，但 CI 跨零；Stable、NUS 没有额外收益。故 Quality weighting 的独立价值只能描述为“有 E-hull 趋势，证据不足”，不能复述 P0 的强结论。

## 20–26：科学问题与失败分析

20. **P0 的 E-hull / Stable / NUS signal 是否复现**：仅部分复现。P0 的 M2−C0 为 E-hull `−0.07578`、Stable `+37.5 pp`、NUS `+25.0 pp`；P1 为 E-hull `−0.04996`（CI 不跨零）、Stable `+6.25 pp`（CI 跨零）、NUS `0 pp`。因此 E-hull 明确复现，Stable 只保留弱方向，NUS 未复现。

21. **Novelty 是否真实下降**：证据指向“存在真实 trade-off，但幅度仍不精确”。P0 与独立 P1 的 M2−C0 都恰为 `−12.5 pp`；P1 中 6 个 paired seeds 变差、2 个变好，bootstrap CI 仍跨零。两轮同方向、同幅度，加上 composition 收缩，使其不宜再归为单个 8-seed 随机波动。

22. **Composition distribution 是否收缩**：有中等收缩，但不是“少数公式大量重复”的灾难性 collapse。

| 统计 | C0 | M1 | M2 |
|---|---:|---:|---:|
| unique elements | 34 | 29 | **23** |
| unique composition families / 32 | 31 | 32 | **26** |
| unique reduced formulas / 32 | 32 | 32 | **31** |
| element entropy (nats) | 2.6697 | 2.5116 | **2.3756** |
| composition-family entropy (nats) | 3.4224 | 3.4657 | **3.1731** |
| largest family share | 6.25% | 3.13% | **9.38%** |
| largest reduced-formula share | 3.13% | 3.13% | **6.25%** |

M2 最常见元素为 Ni `26.55%`、Eu `16.55%`、Fe `11.03%`；最常见 composition families 是 `Eu-Ni` 与 `Fe-Ni-Zn`，各 3/32。31/32 个 reduced formulas 仍不同，因此 Stable 改善不是靠大量复制一个公式取得；但元素支持和 family 数量确有收缩。精确 family 在 n=32 下本来就稀疏，所以 JSD 只作描述：M2/C0 element JSD=`0.1568 nats`，M1/C0=`0.1463 nats`。

23. **Atom-count distribution 是否收缩**：没有。三组逐 seed 的原子数分布完全相同，均值 `9.0625`、中位数 `8`、标准差 `4.7954`、范围 `3–20`、12 种 atom counts，M2/C0 与 M2/M1 atom-count JSD 都严格为 `0`。这说明收缩来自元素/组成选择，而非结构尺寸。

24. **M1 的作用**：普通 unweighted Adapter 本身已相对 C0 改善 E-hull `−0.02948 eV/atom`、Stable `+9.38 pp`、NUS `+12.5 pp`、RMSD `−0.07483 Å`、原子 force mean `−0.14618 eV/Å`、平均松弛步数 `−26.0`。M1 是一个强且更均衡的对照，说明大量收益来自普通 Adapter fine-tuning，而不是 Quality weighting 独有。

25. **M2 相对 M1 的额外作用**：主要是进一步压低 E-hull 均值 `0.02048 eV/atom` 和上尾；force 与 steps 基本相同。代价是 Stable `−3.13 pp`、NUS `−12.5 pp`、Novel `−9.38 pp`，并出现更窄的 composition support。因此当前 Quality weighting 更像“能量/风险偏好加强器”，尚未证明带来全面最终质量收益。

26. **是否由少数 outlier 驱动**：核心 E-hull/force 结论不是。M2−C0 E-hull 的 leave-one-out mean delta 范围为 `[−0.05624, −0.03906]`，删除任一 seed 都不翻转；23/32 seeds 有利。Force mean、max-force 和 steps 的 CI 也不跨零且 leave-one-out 方向稳定。RMSD 的幅度受 C0 seed 70018 (`1.3315 Å`) 和 M2 seed 70021 (`0.8071 Å`) 影响；不删除它们时 CI 跨零，paired median 仍为 `−0.00664 Å`，故只判“对 C0 无系统恶化”。M2−M1 RMSD 的 paired median 接近零、16/16 分裂，均值差主要反映尾部而非普遍退化。所有 outliers 均保留。

## 27–30：最终判定与下一步

27. **最终判定：BORDERLINE**。

| 冻结判据 | 结果 | 判断 |
|---|---|---|
| A Quality | E-hull 明确改善；Stable 弱正向；NUS 不变 | 部分满足 |
| B Geometry | 相对 C0 的 force/steps 明确改善，RMSD 无系统恶化 | 满足 |
| C Distribution | 无 atom-count/严重公式 collapse，但 Novel −12.5 pp 且 composition 中等收缩 | 边界 |
| D M2 超越 M1 | 仅 E-hull 有利趋势且 CI 跨零；Stable/NUS 更低 | 证据不足 |

这不是 FAIL：E-hull、force 和松弛效率的信号真实且稳健；但也不满足 GO 对“额外质量收益 + 无明显 novelty/composition trade-off”的组合要求。

28. **是否建议进入 P2**：**否，当前不进入 P2**。本轮没有启动 64 seeds、formal256、重新训练或任何新方法。

29. **如果 GO，下一步最应解决的唯一问题**：本轮不是 GO；若后续通过一次受限修改达到 GO，唯一应继续盯住的问题仍是 composition support 与 Novelty 的保持，不应同时扩展其他模块。

30. **如果 FAIL，最可能的失败原因**：本轮不是 FAIL。当前 BORDERLINE 最可能的问题是全局 CHGNet force-based quality weighting 把生成分布进一步推向低风险的 Ni/Eu/Fe 组成区域，导致 E-hull 收益与 Novelty/NUS 损失绑定。若用户允许一次受限修改，建议只降低 quality-weight contrast（其余数据、Adapter、anchor、replay、训练步数和采样协议全部冻结），先用 4–8 个 smoke seeds 检查 novelty/composition 是否恢复，再决定是否做新的 32-seed 验证；本轮未实施该修改。

## 结果文件与可信度边界

- `generation_results.csv`：96 次真实生成、配置、耗时与 checkpoint 哈希。
- `quality_results.csv`：三组正式指标、RMSD/force/steps 尾部统计。
- `paired_statistics.csv`：M2−C0、M2−M1 的逐 seed delta、20,000 次 bootstrap CI 与 leave-one-out 敏感性。
- `distribution_analysis.csv`：元素、composition family、reduced formula、atom-count 频率和 JSD。

本实验是 32-seed 独立探索验证，所有结论均为 surrogate 结论：`DFT_VERIFIED=False`。没有 DFT relaxation 或 DFT E-hull，因此不得将 Stable、NUS、E-hull、force、RMSD 写成 DFT 已验证结果。

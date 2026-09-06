# MatterGen Quality-Weight Contrast P1b 最终报告

## 一句话结论

原 M2 的真实 quality weights 为高质量/中间/低质量 1.25/1.0/0.75。本轮只将 contrast 降为 L1 1.125/1.0/0.875 和 L2 1.075/1.0/0.925，其余训练与采样变量全部冻结。在 16 个全新 paired seeds 上，L1 恢复 Novel 与 composition coverage，却没有保留明确 E-hull/force 几何收益；L2 保留并加强质量收益，却没有恢复 composition coverage，Novel 仍显著低于 C0。两个候选都未形成要求的 quality–distribution Pareto，因此没有冻结候选、没有运行 32-seed confirmatory validation，P1b 判定 **FAIL**，不建议进入 P2。

全部 E-hull、Stable、NUS、Novel、force 和 RMSD 均为 MatterSim-5M surrogate 与 MatterGen evaluator 结果，DFT_VERIFIED=False。

## 1–6：版本、权重、训练与执行

1. **Branch**：experiment/distribution-constrained-quality-adapter-p1b，从冻结 P1 final HEAD 180acb6005fab026999d8ba6bdcdffca26d0a3af 创建。P0/P1 结果与 checkpoint 均未修改。

2. **Final HEAD**：提交报告后由 git rev-parse HEAD 得到，并在最终对话交付中给出；commit 无法在自身内容中保存自己的哈希。

3. **Original M2 weights**：高质量 low-force=1.25，middle=1.0，低质量 high-force=0.75。这是从冻结 label_quality.py 和 quality_labels.csv 读取的实际值，不是猜测。

4. **M2-L1 weights**：1.125/1.0/0.875，保留原端点差的 50%。

5. **M2-L2 weights**：1.075/1.0/0.925，保留原端点差的 30%。

两个候选均从同一原始 pretrained MatterGen checkpoint 初始化，而非从原 M2 继续训练。冻结配置：128 clean-x0、96/32 train/val、interaction blocks 1/2、512→64→512、134,272 trainable params、200 steps、batch 16、lr 3e-4、AdamW、replay 50%、anchor 0.05、训练 seed 20260906。派生标签除 quality_weight 一列外完全相同。

| 训练 | finite | zero-init | 主干不变 | Adapter 有梯度 | 前/后 20-step loss | best validation | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2-L1 | 是 | 是 | 是 | 12/12 | 0.27808 / 0.28330 | 0.23279 | 17.17 s |
| M2-L2 | 是 | 是 | 是 | 12/12 | 0.27886 / 0.28284 | 0.23273 | 17.76 s |

6. **16-seed 设计与成功率**：新 seeds 71000–71015 在启动前未出现在历史实验目录；C0/M1/原 M2/L1/L2 共 80 次真实 generation，全部成功；80/80 MatterSim GPU 松弛成功。采样均为 dft_mag_density=0.1、1000-step predictor-corrector、corrector=1、CFG=2.0、相同 precision/config。平均纯采样时间 C0/M1/M2/L1/L2 分别为 115.44/117.41/117.65/117.06/117.30 s。

## 7–17：16-seed 正式结果

| 指标 | C0 | M1 | original M2 | M2-L1 | M2-L2 |
|---|---:|---:|---:|---:|---:|
| 7. E-hull mean (eV/atom, ↓) | 0.12056 | 0.12826 | 0.13427 | 0.11710 | **0.10085** |
| E-hull median | 0.12515 | 0.12872 | 0.11585 | 0.09674 | **0.09270** |
| E-hull P95 | 0.22620 | 0.29924 | 0.26245 | 0.29846 | **0.22541** |
| 8. Stable | 6/16 (37.5%) | 7/16 (43.75%) | 7/16 (43.75%) | 8/16 (50.0%) | **9/16 (56.25%)** |
| 9. NUS | 5/16 (31.25%) | 4/16 (25.0%) | 2/16 (12.5%) | **6/16 (37.5%)** | 4/16 (25.0%) |
| 10. Novel | **15/16 (93.75%)** | 13/16 (81.25%) | 10/16 (62.5%) | 14/16 (87.5%) | 11/16 (68.75%) |
| 11. Unique | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| 12. Atomic force mean (eV/Å, ↓) | 0.24378 | 0.57145 | 0.17791 | 0.28592 | **0.09645** |
| 13. Structure max-force mean (eV/Å, ↓) | 0.43824 | 1.17754 | 0.28670 | 0.49464 | **0.15463** |
| 14. RMSD mean (Å, ↓) | 0.04694 | 0.09045 | 0.06942 | 0.10410 | **0.02659** |
| RMSD median | 0.02498 | 0.02440 | 0.02582 | 0.02086 | **0.00562** |
| Force P95 (all atoms) | 0.90141 | 1.87088 | 0.55154 | 1.17023 | **0.28221** |
| Relaxation steps mean/max | 46.94/141 | 50.94/229 | 58.56/195 | 106.38/631 | **33.75/147** |
| 15. Composition families | 16/16 | 16/16 | 15/16 | **16/16** | 14/16 |
| 16. Unique elements | 27 | 18 | 20 | **21** | 19 |
| Reduced-formula uniqueness | 16/16 | 16/16 | 16/16 | 16/16 | 16/16 |
| 17. Num-atoms distribution | 与所有组相同 | 相同 | 相同 | 相同 | 相同 |

五方法逐 seed 的 atom count 完全一致：9 种 atom counts、均值 10.9375，所有相对 C0/M1/M2 的 atom-count JSD 都为 0。因此本轮差异来自元素/组成与几何，不是尺寸分布变化。

## 18–20：Pareto 选择与停止决定

18. **哪个 contrast 的 Pareto 最好**：没有一个候选同时满足预设 Pareto。若只看 distribution，L1 最好；若只看 quality/geometry，L2 最好，但二者不可合并为一个成功候选。

19. **为什么**：

- L1 相对原 M2：Novel +25 pp、NUS +25 pp、composition family 15→16，覆盖恢复；但相对 C0 的 E-hull 仅 −0.00346 eV/atom，20k paired CI [−0.06662, 0.05898]。Atomic force mean 比 C0 高 17.3%，structure max-force mean 高 12.9%，RMSD mean 和 force tail 更差。L1 的 seed 71004（Gd6Ni14）max-force=4.197 eV/Å，seed 71003/71004 分别需 631/599 松弛 steps；不删除这些不利样本。即使结构平均 force 的 paired median 有利，E-hull/force 方向会在 leave-one-out 下翻转，不能称为“明确保留质量收益”。
- L2 相对 C0：E-hull −0.01971 eV/atom，structure force mean −0.12920、max-force mean −0.28360 eV/Å，后两者 CI 均不跨零；Stable +18.75 pp。但 Novel 仍比 C0 低 25 pp，CI [−50, −6.25] pp，只比原 M2 恢复一个样本；composition families 从原 M2 的 15/16 进一步降至 14/16，独立元素也从 20 降至 19。故 distribution 没有恢复。

20. **是否冻结候选进入 32-seed**：否。预注册条件要求“明确 Pareto 改善”才允许冻结一个候选。L1 触发“contrast 降低后明确 E-hull/force 收益基本消失”的停止条件；L2 触发“quality 保留但 composition 仍偏窄”的停止条件。没有继续测试第三个 contrast。

## 21–24：32-seed 与 paired CI

21. **32-seed 最终结果**：未运行。72000–72031 没有被启动，因为 16-seed 阶段没有可冻结候选。这是按停止规则执行，不是实验中断。

22. **Candidate vs C0 final-32 paired CI**：不适用；不存在 final candidate。作为透明的筛选证据，paired_statistics.csv 保存了 L1/L2 在 16 seeds 上相对 C0/M1/原 M2 的 20,000 次探索性 paired bootstrap。

23. **Candidate vs M1 final-32 paired CI**：不适用。16-seed 中 L1−M1 E-hull=−0.01116，CI [−0.06574, 0.04811]；L2−M1=−0.02741，CI [−0.08338, 0.02459]。M1 的 force/RMSD 均值受 seed 71004（H11Mn9，max-force 13.262 eV/Å，RMSD 0.997 Å）强烈影响，因此不把候选相对 M1 的 force 均值优势当作稳定额外价值。

24. **Candidate vs original M2 final-32 paired CI**：不适用。16-seed 中 L1−M2：E-hull=−0.01718、CI [−0.09213, 0.05659]，Novel=+25 pp、CI [−6.25, 56.25] pp，max-force mean=+0.20793、CI [−0.10676, 0.72290]。L2−M2：E-hull=−0.03343、CI [−0.08925, 0.01781]，Novel=+6.25 pp、CI [−18.75, 31.25] pp，max-force mean=−0.13207、CI [−0.27400, −0.00536]。

## 25–30：科学结论

25. **Novel 是否恢复**：L1 明确恢复到 87.5%，接近 C0 的 93.75% 且高于 M1 的 81.25%；L2 仅从原 M2 的 62.5% 恢复到 68.75%，仍显著低于 C0。故“降低 contrast 会恢复 Novel”不是单调关系，也没有与 quality 同时实现。

26. **Composition coverage 是否恢复**：L1 恢复为 16/16 families；L2 未恢复，反而降为 14/16。所有候选 reduced formulas 都是 16/16 unique，因此没有公式复制型严重 collapse，但 L2 仍有 composition contraction。

27. **E-hull/Force 收益保留多少**：相对同 seeds 的 C0，L1 E-hull 仅降低 2.9%，Atomic force mean/Max-force mean 反而增加 17.3%/12.9%，没有保留明确几何收益。L2 E-hull 降低 16.4%，Atomic force mean/Max-force mean 降低 60.4%/64.7%，质量收益充分保留，但 distribution 目标失败。原 M2 在这 16 seeds 的 E-hull 本身比 C0 高 11.4%，说明 fresh-seed E-hull 波动也较大；这进一步要求候选必须通过预设 Pareto 门槛，而不能按单一均值事后选择。

28. **Quality weighting 相对普通 Adapter 是否仍有额外价值**：没有证明出均衡的额外价值。L1 比 M1 有更好的 NUS/Novel/coverage，但 E-hull CI 跨零且几何不稳定；L2 有更低 E-hull/force，却比 M1 少 12.5 pp Novel、少 2 个 composition families。Quality pressure 控制的是质量–覆盖权衡位置，但单纯调小幅度没有产生同时支配 M1 的解。

29. **最终判定：FAIL**。该判定专指本轮唯一假设——“仅降低 quality-weight contrast 足以解决 P1 的 Novelty/composition contraction”——被 16-seed 真实实验否定。它不否定 P1 已确认的原 M2 E-hull/force signal。

30. **是否建议进入 P2**：否。停止继续 weight tuning，不运行 P2/formal256。若开启下一轮，最直接的科学方向是单独设计 explicit distribution constraint；Global Transformer Adapter 是另一条备选路线。但二者都超出本轮授权，当前没有实现或运行。

## 必要结果文件

- training_summary.csv：L1/L2 真实训练稳定性和 checkpoint 哈希。
- generation_results.csv：80 次真实生成与冻结采样配置。
- quality_results.csv：五方法正式 quality/force/RMSD/coverage 汇总。
- paired_statistics.csv：两候选相对三组对照的 16-seed 探索性 20k paired bootstrap。
- distribution_analysis.csv：元素、composition family、formula 与 atom-count 频率。

本轮没有 DFT 计算：DFT_VERIFIED=False。

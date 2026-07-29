# MatterGen 负面实验与 No-Go 路线总结

本文件记录最终未被选为论文创新点的主要路线。状态来自现有冻结报告、分支历史和实验 manifest；没有重新计算指标。服务器侧报告路径仅作为来源标识，不是便携 Markdown 链接。若当前归档未完整保存某字段，明确写为 `Not fully recovered from current archive`。

## 阅读规则

- `No-Go` 表示未通过该路线预先冻结的工程、质量或科学门槛，不等于实现错误。
- 小样本 Gate 只支持提前停止，不应外推为正式效应估计。
- Stable、E-hull、NUS 和 RMSD 使用 MatterSim-5M surrogate；`DFT_VERIFIED=False`。
- 分支头可能包含后续文档提交；实验身份优先使用报告中冻结的 commit。

## 1. Residual Reuse

| 字段 | 内容 |
|---|---|
| 方法名称 | Unconditional Residual Reuse |
| 状态 | No-Go；保留为负面消融 |
| 研究动机 | 复用已收敛的 unconditional residual，减少 CFG 逻辑 NFE |
| 方法简介 | 保留 conditional forward，按收敛判据复用/外推 unconditional residual，并周期校准 |
| 是否训练 | 否 |
| 基线 | Full adaptive CFG（F0） |
| 实验规模 | 12 次重复生成；固定并发 1/2/4/8 workers/GPU、每组 32 seeds；未启动质量 8/32/64 或正式 seeds |
| 主要结果 | joint forward 30.269 ms，conditional-only 29.784 ms；R0 unconditional NFE −16.80%；中位 wall-time +1.35%；最佳同并发吞吐 +1.16% |
| No-Go 触发原因 | 未达到 NFE ≥25%、wall-time ≥5% 或同并发吞吐 ≥8% |
| 失败原因分析 | 单样本下 joint CFG 的 `2×batch` 已被 GPU 并行吸收；只删除一个逻辑分支不等于删除完整物理 forward |
| 可复用部分 | 收敛判据、缓存/决策开销测量、固定并发 benchmark、确定性验证 |
| 论文用途 | 负面消融：解释“逻辑 NFE 降低不保证真实加速” |
| 分支 | 历史 manifest 记录 `feature/convergence-aware-cfg`；远端 ref 未从当前归档恢复 |
| commit | `48f0d87dfab4e450800460fccc0bd03c24553e82`（历史实现 manifest head） |
| 报告路径 | 服务器冻结报告：`/data/dxl/reports/convergence_aware_cfg/archive_unconditional_reuse/no_go_report.md`；Not fully recovered from current archive |

## 2. Corrector Gating

| 字段 | 内容 |
|---|---|
| 方法名称 | Convergence-Aware Corrector Gating |
| 状态 | Formal No-Go |
| 研究动机 | 在收敛阶段跳过完整 Corrector forward，使逻辑减少转化为真实物理计算减少 |
| 方法简介 | warmup 后按残差稳定性跳过 Corrector，并周期校准、fallback/rescue |
| 是否训练 | 否 |
| 基线 | A0 Adaptive CFG |
| 实验规模 | 8-seed pilot、32/64 开发验证、正式 256 seeds（20000–20255） |
| 主要结果 | 正式参考：1.506× 单任务加速，吞吐 +44.76%，物理 forward −35.37%；E-hull +0.022423 eV/atom，Stable −9.7656 pp，NUS −9.3750 pp |
| No-Go 触发原因 | 正式质量下降超过冻结门槛，`FORMAL_INNOVATION2_CONFIRMED=False` |
| 失败原因分析 | Corrector 被大比例跳过时改变采样动力学；开发小样本安全性没有在正式规模保持 |
| 可复用部分 | skip/calibration/fallback 机制、速度—质量 Pareto 分析、物理 forward 计数 |
| 论文用途 | 代表性负面实验：真实加速与生成质量的直接权衡 |
| 分支 | [`feature/convergence-aware-corrector-gating`](https://github.com/du17183/mattergen_v1/tree/feature/convergence-aware-corrector-gating) |
| commit | `5de00419eea2d8a9be303638f2db8ece15a22366`（冻结 G3 配置记录） |
| 报告路径 | 服务器冻结报告：`/data/dxl/reports/budget_aware_gating/frozen_formal_baseline/formal_final_report.md`；Not fully recovered from current archive |

> 分支名称同时承载后来正式保留的 Adaptive CFG；Corrector Gating 的 No-Go 不能被误写成 Adaptive CFG No-Go。

## 3. Budget-aware Corrector Gating

| 字段 | 内容 |
|---|---|
| 方法名称 | Budget-Aware Convergence-Guided Corrector Scheduling |
| 状态 | 32-seed No-Go；64/正式 seeds 未启动 |
| 研究动机 | 用预算、Atomic veto、校准和 fallback 缓和 Corrector Gating 的质量损失 |
| 方法简介 | G1 保守预算与 G2 中等预算，自适应控制 Corrector 跳过比例 |
| 是否训练 | 否 |
| 基线 | A0 Adaptive CFG |
| 实验规模 | 开发 seeds 14000–14031，配对 n=32；96/96 generation 和 relaxation 成功 |
| 主要结果 | G1：1.183×、吞吐 +13.01%、forward −12.63%、Stable −3.125 pp、NUS −6.25 pp；G2：1.234×、吞吐 +22.28%、forward −20.83%、Stable −3.125 pp、E-hull +0.022292 |
| No-Go 触发原因 | G1 未过 Stable/NUS；G2 未过速度、Stable/E-hull；无候选同时通过全部 Pareto gate |
| 失败原因分析 | 更保守预算保护部分质量但加速不足；更激进预算恢复速度时再次损伤质量 |
| 可复用部分 | 预算控制、Atomic veto、自适应校准、候选冻结与 Pareto 决策 |
| 论文用途 | Corrector Gating 的安全化失败消融 |
| 分支 | [`feature/budget-aware-corrector-gating`](https://github.com/du17183/mattergen_v1/tree/feature/budget-aware-corrector-gating) |
| commit | `0d0f2e3720fef2a9686f9de4a6480a732bdeb812`（最终 environment manifest）；远端文档头 `3f7cc791915e79b5fdc1826f5c04c4e247c2a0c9` |
| 报告路径 | 服务器冻结报告：`/data/dxl/reports/budget_aware_gating/final/budget_aware_final_report.md`；Not fully recovered from current archive |

## 4. FN-PRA

| 字段 | 内容 |
|---|---|
| 方法名称 | Frozen-Node Physics Representation Alignment（Phase 1 静态版本） |
| 状态 | No-Go |
| 研究动机 | 用冻结 CHGNet Teacher 的原子级物理表示约束 MatterGen 中间表示 |
| 方法简介 | `dft_mag_density` 条件 A0 上对最后 GemNet block 施加静态 atom-level REPA |
| 是否训练 | 是；5000-step 微调，Teacher 冻结 |
| 基线 | A0 Adaptive CFG |
| 实验规模 | 32 paired seeds；A0/P1 generation 与 MatterSim 评价 |
| 主要结果 | Composition −6.25 pp，Stable −6.25 pp，E-hull +0.003786 eV/atom；NUS +6.25 pp；Novel +21.875 pp；RMSD −28.68% |
| No-Go 触发原因 | Composition 工程门槛和 Stable 系统性质量门槛失败 |
| 失败原因分析 | 静态末层对齐改善局部放松性/多样性，但可能扰动条件生成分布与组成稳定性 |
| 可复用部分 | Teacher cache、atom mapping、EA-NCE 基础设施、训练与配对评价 runner |
| 论文用途 | 表示对齐路线的负面消融 |
| 分支 | [`feature/fn-pra`](https://github.com/du17183/mattergen_v1/tree/feature/fn-pra) |
| commit | `42681f83a0d70c25f6f2e598232868c169904e30` |
| 报告路径 | 服务器冻结报告：`/data/dxl/reports/fn_pra/phase1/phase1_final_report.md`；Not fully recovered from current archive |

## 5. CrystalREPA

| 字段 | 内容 |
|---|---|
| 方法名称 | CrystalREPA unconditional MatterGen reproduction |
| 状态 | Reproduction No-Go，`REPA_BASE_REPRODUCED=False` |
| 研究动机 | 在论文对应的无条件 MP-20 设置隔离验证 REPA 是否可复现稳定性改善 |
| 方法简介 | 官方无条件 MP-20 checkpoint；GemNet block 2；EA-NCE；CHGNet 0.3.0 Teacher |
| 是否训练 | 是；8-GPU DDP，最多 10000 steps；44,858,533 trainable parameters |
| 基线 | U0 官方无条件 MP-20 MatterGen |
| 实验规模 | 64 paired seeds（U0/R1 各 64 generation + relaxation） |
| 主要结果 | Composition −3.125 pp；E-hull +0.094236 eV/atom；Metastable −6.25 pp；RMSD +0.033290 Å；Stable 均为 0 |
| No-Go 触发原因 | 工程与科学门槛均失败；E-hull、Metastable、RMSD 系统性不安全 |
| 失败原因分析 | 本地 Teacher 与论文 Teacher 存在受控偏差，训练上限也远短于论文；当前设置未复现基础方向 |
| 可复用部分 | 无条件 checkpoint 审计、中间层 hook、EA-NCE mask、DDP all-gather、严格 cache mapping |
| 论文用途 | 说明不能把外部论文方向直接外推到当前 Teacher/训练预算 |
| 分支 | [`feature/crystalrepa-repro`](https://github.com/du17183/mattergen_v1/tree/feature/crystalrepa-repro) |
| commit | `36ceecf8d01f14a03127ee104e2b43a3fc644534`（实验结果）；最终归档头 `1e51c76d6a9c7d3635bc8c20f2c2525aa3d7c0fa` |
| 报告路径 | 服务器冻结报告：`/data/dxl/reports/crystalrepa_repro/crystalrepa_repro_final_report.md`；Not fully recovered from current archive |

## 6. RP-QTFG

| 字段 | 内容 |
|---|---|
| 方法名称 | Residual-Preserving Quality-Constrained Training-Free Guidance |
| 状态 | Gate 1 No-Go；32/64/256 未启动 |
| 研究动机 | 在中低噪声阶段用 CHGNet 物理梯度改善 A0 几何，同时保护条件残差 |
| 方法简介 | clean-x0 物理梯度、字段归一化、residual conflict、trust region、backtracking 和 A0 fallback |
| 是否训练 | 否 |
| 基线 | A0 Adaptive CFG |
| 实验规模 | Gate 0A：8716 held-out 结构；Gate 0B：64 结构；Gate 1：4 candidates × 8 paired seeds |
| 主要结果 | 离线方向成立；最近候选 G1_P75_S 的 E-hull −0.003353、Stable/NUS 不变，但 RMSD +68.28%，最大力仅 −0.26%，延迟 +30.19% |
| No-Go 触发原因 | 所有在线候选 RMSD 恶化 16.3%–293.3%，无明确 force/RMSD 正向候选 |
| 失败原因分析 |  finished structure 的单次局部梯度有效，但反复注入耦合扩散轨迹会改变后续 denoiser 预测 |
| 可复用部分 | CHGNet 磁 proxy 审计、clean-x0 转换、安全 backtracking/fallback、离线方向 probe |
| 论文用途 | 代表性负面实验：局部物理梯度方向与在线采样轨迹不一致 |
| 分支 | [`feature/rp-qtfg`](https://github.com/du17183/mattergen_v1/tree/feature/rp-qtfg) |
| commit | `e457a43404e6d52d5ce2e4bb2dffc015d36a71d5`（结果）；最终归档头 `85152b2fd52754bb1333e919d7bfabad3d0354d1` |
| 报告路径 | 服务器冻结报告：`/data/dxl/reports/rp_qtfg/phase0/final_report.md`；Not fully recovered from current archive |

## 7. CG-TDR

| 字段 | 内容 |
|---|---|
| 方法名称 | Convergence-Guided Terminal Delta Refiner |
| 状态 | Gate V2 eight-seed No-Go；32/64/正式未启动 |
| 研究动机 | 学习 Teacher 终端 position/cell residual，用小型门控精修生成结构 |
| 方法简介 | V1 直接残差与近 always-on gate；V2 用 utility-calibrated selective gate 修复选择性 |
| 是否训练 | 是；V2 最佳 step 1100/1500 |
| 基线 | A0 Adaptive CFG |
| 实验规模 | 残差 held-out 诊断；V1/V2 各 8 paired seeds |
| 主要结果 | V1 position/cell loss 比 zero baseline 更差，position cosine −0.044；V2P 质量安全但近乎平坦，V2C median RMSD +18.29% |
| No-Go 触发原因 | V2P 未达到任一正向门槛；V2C 不安全 |
| 失败原因分析 | 直接学习 Teacher residual 未可靠泛化；路由可学到 utility，但错误的残差方向仍限制端到端收益 |
| 可复用部分 | Teacher delta 数据、utility label、选择性门控、残差方向诊断 |
| 论文用途 | 代表性负面实验：直接学习 Teacher 残差难以泛化 |
| 分支 | [`feature/cg-tdr`](https://github.com/du17183/mattergen_v1/tree/feature/cg-tdr) |
| commit | `e31e4c1b4648e61b18dfd142317e5c1f4ed73ff6` |
| 报告路径 | 服务器冻结报告：`/data/dxl/reports/cg_tdr/phase0/cg_tdr_eval_final.md`；Not fully recovered from current archive |

## 8. Q1 UQ-PQR

| 字段 | 内容 |
|---|---|
| 方法名称 | Uncertainty-Aware Pairwise Quality Reranker |
| 状态 | Historical held-out No-Go |
| 研究动机 | 用多任务质量预测和 ensemble uncertainty 从四候选池选择更优结构 |
| 方法简介 | 冻结 C0 轨迹池；基于学习式质量/不确定性 score 排序，不修改采样轨迹 |
| 是否训练 | 是；共享质量模型数据为 train 576、validation 96、test 96 rows |
| 基线 | 每个四候选池的 C0_FIRST |
| 实验规模 | historical held-out 32 rows；8 pools/trial，1000 次冻结重采样评估 |
| 主要结果 | E-hull −0.07046 eV/atom，Stable +43.80 pp，NUS +19.45 pp，RMSD −61.02%，但 Novel −14.86 pp |
| No-Go 触发原因 | 违反冻结 Novel 安全门槛 |
| 失败原因分析 | quality-driven selection 偏向训练/参考分布附近结构，质量代理改善伴随新颖性损失 |
| 可复用部分 | 多任务质量模型、ensemble uncertainty、候选池评估 |
| 论文用途 | 候选排序路线附录 |
| 分支 | [`feature/postgen-quality-modules-fastgate`](https://github.com/du17183/mattergen_v1/tree/feature/postgen-quality-modules-fastgate) |
| commit | `b65f42a8792004c7c820e59fa4413e1310e06143` |
| 报告路径 | [Post-generation fast-gate final report](../../reports/postgen_fastgate/final_report.md) |

## 9. Q2 RFR

| 字段 | 内容 |
|---|---|
| 方法名称 | Relaxability and Failure-Risk Router |
| 状态 | Historical held-out No-Go |
| 研究动机 | 依据预测的 RMSD 风险和安全概率选择更易松弛的候选 |
| 方法简介 | 冻结 C0 四候选池；使用 RMSD risk 与安全概率组合 score |
| 是否训练 | 是；共享质量模型数据为 train 576、validation 96、test 96 rows |
| 基线 | C0_FIRST |
| 实验规模 | historical held-out 32 rows；8 pools/trial，1000 次冻结重采样评估 |
| 主要结果 | E-hull −0.07938 eV/atom，Stable +39.91 pp，NUS +7.14 pp，RMSD −78.64%；Novel −30.25 pp，Unique −8.45 pp |
| No-Go 触发原因 | 同时违反 Novel 与 Unique 安全门槛 |
| 失败原因分析 | 过度偏向低松弛风险的“熟悉”结构，导致多样性/新颖性收缩 |
| 可复用部分 | relaxability 风险预测、失败概率校准、路由框架 |
| 论文用途 | 排序偏置与质量—新颖性权衡消融 |
| 分支 | [`feature/postgen-quality-modules-fastgate`](https://github.com/du17183/mattergen_v1/tree/feature/postgen-quality-modules-fastgate) |
| commit | `b65f42a8792004c7c820e59fa4413e1310e06143` |
| 报告路径 | [Post-generation fast-gate final report](../../reports/postgen_fastgate/final_report.md) |

## 10. Q4 CPRC

| 字段 | 内容 |
|---|---|
| 方法名称 | Cross-Potential Residual Calibrator |
| 状态 | Historical held-out No-Go |
| 研究动机 | 用 MatterSim E-hull prediction、uncertainty 与 safety 校准 CHGNet/生成质量 proxy |
| 方法简介 | 对冻结 C0 四候选池计算跨势模型校准 score 后选择 |
| 是否训练 | 是；共享质量模型数据为 train 576、validation 96、test 96 rows |
| 基线 | C0_FIRST |
| 实验规模 | historical held-out 32 rows；8 pools/trial，1000 次冻结重采样评估 |
| 主要结果 | E-hull −0.07199 eV/atom，Stable +44.44 pp，NUS +17.44 pp，RMSD −61.64%，但 Novel −17.08 pp |
| No-Go 触发原因 | 违反冻结 Novel 安全门槛 |
| 失败原因分析 | 跨势校准改善代理质量排序，但仍偏向参考分布内结构 |
| 可复用部分 | 跨势残差特征、uncertainty/safety 组合、候选池 protocol |
| 论文用途 | 跨势校准路线附录 |
| 分支 | [`feature/postgen-quality-modules-fastgate`](https://github.com/du17183/mattergen_v1/tree/feature/postgen-quality-modules-fastgate) |
| commit | `b65f42a8792004c7c820e59fa4413e1310e06143` |
| 报告路径 | [Post-generation fast-gate final report](../../reports/postgen_fastgate/final_report.md) |

## 11. Q5 CQPS

| 字段 | 内容 |
|---|---|
| 方法名称 | Condition-Quality Pareto Selector |
| 状态 | New-data No-Go |
| 研究动机 | 在条件命中、质量、稳定性与新颖性之间做 Pareto 候选选择 |
| 方法简介 | 从冻结的四轨迹池选择一个候选；新 MatterSim 标签仅在选择冻结后用于评价 |
| 是否训练 | 使用冻结质量模型/selector；不训练 MatterGen 主干 |
| 基线 | 新 32-pool C0_FIRST |
| 实验规模 | 32 独立 pools × 4 C0 trajectories（seeds 33000–33127） |
| 主要结果 | E-hull −0.03140 eV/atom，Stable +21.88 pp，NUS +3.13 pp，RMSD −51.96%，最大力 −36.81%，但 Novel −15.63 pp |
| No-Go 触发原因 | 独立数据 Novel 安全门槛失败 |
| 失败原因分析 | 多候选选择确实提高代理质量，但用额外采样换取“更熟悉”的候选，削弱 novelty |
| 可复用部分 | Pareto selector、冻结选择清单、盲测数据流水线 |
| 论文用途 | 独立盲测负面结果；说明不能只看 E-hull/Stable |
| 分支 | [`feature/postgen-quality-modules-fastgate`](https://github.com/du17183/mattergen_v1/tree/feature/postgen-quality-modules-fastgate) |
| commit | `b65f42a8792004c7c820e59fa4413e1310e06143` |
| 报告路径 | [Post-generation fast-gate final report](../../reports/postgen_fastgate/final_report.md)；Q5 子报告仅保留在历史分支/服务器 |

## 12. Q6 NS-SetRank

| 字段 | 内容 |
|---|---|
| 方法名称 | Novelty-Stability SetRank |
| 状态 | New-data No-Go |
| 研究动机 | 用候选集合上下文联合排序 Novelty、Stable、NUS 和质量 |
| 方法简介 | 冻结 CHGNet 特征与三成员 SetRank ensemble，对每个四候选池排序 |
| 是否训练 | 是；3 个 48,865 参数成员，MatterGen/CHGNet 冻结 |
| 基线 | 新 32-pool C0_FIRST |
| 实验规模 | 32 独立 pools × 4 C0 trajectories（seeds 33000–33127） |
| 主要结果 | E-hull −0.03372 eV/atom，Stable +28.13 pp，NUS +9.38 pp，RMSD −52.95%，最大力 −45.59%，但 Novel −12.50 pp |
| No-Go 触发原因 | 独立数据 Novel 安全门槛失败 |
| 失败原因分析 | 集合排序能找到低风险候选，但 learned ranking 仍向训练分布/稳定代理集中 |
| 可复用部分 | setwise ranker、ensemble、候选池盲测和 frozen selector |
| 论文用途 | 独立盲测负面结果与多候选选择局限性 |
| 分支 | [`feature/postgen-quality-modules-fastgate`](https://github.com/du17183/mattergen_v1/tree/feature/postgen-quality-modules-fastgate) |
| commit | `b65f42a8792004c7c820e59fa4413e1310e06143` |
| 报告路径 | [Post-generation fast-gate final report](../../reports/postgen_fastgate/final_report.md)；Q6 子报告仅保留在历史分支/服务器 |

## 13. GPU acceleration routes

| 字段 | 内容 |
|---|---|
| 方法名称 | GPU inference acceleration fast-gates |
| 状态 | 论文级 No-Go；部分结果可作工程配置/性能负面证据 |
| 研究动机 | 在不改变生成质量的前提下获得真实端到端吞吐提升 |
| 方法简介 | Native B4/B8、field-safe BF16、partial `torch.compile`、静态周期图分桶、局部 GemNet fusion、持久化多 worker、NVIDIA MPS |
| 是否训练 | 否 |
| 基线 | C0/A0，通常 FP32、batch size 1、完整 Predictor/Corrector |
| 实验规模 | 各子路线使用冻结 microbenchmark、真实 states、8/16/32 seeds 或多轮 timing；具体规模见对应分支报告 |
| 主要结果 | Native B4 约 3× throughput 但非质量等价；BF16 forward 0.885×；局部 compile full forward 1.009×且严格数值失败；静态图 full forward 0.964×；2 persistent workers/GPU 1.163×；MPS 相对同 worker −0.428% |
| No-Go 触发原因 | 没有路线同时达到冻结的端到端加速、位级/数值一致性和质量门槛 |
| 失败原因分析 | MatterGen 动态周期图、scatter/indexing 与小图启动开销限制 compile/静态图收益；并发在 2 workers/GPU 左右饱和；batch 改变逐轨迹 RNG/输出语义 |
| 可复用部分 | profiler、bitwise checker、persistent worker runtime、MPS runner、静态图 exactness tests、性能 Gate |
| 论文用途 | GPU 性能章节和工程负面结果，不作为第二算法创新点 |
| 分支 | [`feature/spg-mattergen-fastgate`](https://github.com/du17183/mattergen_v1/tree/feature/spg-mattergen-fastgate)、[`feature/spg-static-periodic-graph-mvp`](https://github.com/du17183/mattergen_v1/tree/feature/spg-static-periodic-graph-mvp)、[`feature/gemnet-fused-inference-fastgate`](https://github.com/du17183/mattergen_v1/tree/feature/gemnet-fused-inference-fastgate)、[`feature/mps-runtime-fastgate`](https://github.com/du17183/mattergen_v1/tree/feature/mps-runtime-fastgate) |
| commit | 分别为 `be523546e2ad7052f2a7a29cd96708553ae86a0a`、`923a38aea3336b2d9833e2a7fd8c131f0eb59424`、`d70f7032e1cac7d4d7b2c9cfc869a86f1bccb1dd`、`0f3178b11e3a90d73a2b357a32c49ba0ff867a2c` |
| 报告路径 | 完整报告位于对应历史分支和服务器 `/data/dxl/reports/{spg_fastgate,spg_static_mvp,gemnet_fused_fastgate,mps_fastgate}`；Not fully recovered from current archive |

## 总结

最有论文解释力的代表性负面实验是：

1. **Corrector Gating**：证明真实采样加速存在明显速度—质量权衡；
2. **RP-QTFG**：证明 finished-structure 的局部物理方向不能直接等价为安全的在线 trajectory guidance；
3. **CG-TDR**：证明直接学习 Teacher terminal residual 的方向泛化不足。

Q1/Q2/Q4/Q5/Q6 共同说明：多候选质量选择可以显著改善 surrogate quality，但极易降低 Novel/Unique。GPU 路线则说明当前硬件/动态图工作负载下，工程加速不能自动满足位级一致和质量不变的论文门槛。

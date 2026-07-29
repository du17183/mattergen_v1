# MatterGen 条件晶体生成研究：最终项目总览

> 本仓库基于 Microsoft MatterGen，记录一个面向计算机专业学位论文的条件晶体生成研究项目。当前最终方法由两个可独立验证、可串联使用的模块组成：采样阶段的 **Multi-field Residual-driven Online Adaptive CFG**，以及生成后处理阶段的 **Learned-Gated E3-PCR**。
>
> 本页是项目总入口。论文数据资格、逐 seed 证据与 CPU 重算入口以 [`thesis_archive/`](thesis_archive/README.md) 为准；各实验的实现和复现入口见对应分支根 README。

## 1. 项目简介

本项目围绕 `dft_mag_density` 条件 MatterGen 展开，目标是在保留完整 Predictor/Corrector 采样流程的同时：

- 用在线多字段条件残差自适应调节 CFG；
- 用轻量 Learned Gate 选择性执行安全、有界、等变的生成后位置精修；
- 使用冻结协议、独立 seeds、逐 seed 统计和泄漏诊断约束结论；
- 记录未通过门槛的路线，避免只保留正向结果。

术语约定：

```text
C0 = 原始 MatterGen
A0 = C0 + Adaptive CFG
E3-G = C0 输出 + Learned-Gated E3-PCR
完整方法 = Adaptive CFG + Learned-Gated E3-PCR
```

## 2. 当前整体进度

| 项目 | 状态 |
|---|---|
| 创新点一 Adaptive CFG | 正式验证通过 |
| 创新点二 Learned-Gated E3-PCR | 独立 256-seed 正式验证通过 |
| 组合验证一 | 独立 64-seed 通过 |
| 组合验证二 | 全新独立 64-seed 重复验证通过 |
| Always-on 消融 | 已完成 |
| Random Gate 消融 | 已完成 |
| 训练—测试泄漏诊断 | 已完成 |
| 论文分析归档 | 已完成 |
| 笔记本 CPU 分析 | 已验证 |
| DFT 验证 | 未进行 |
| 目标磁密度独立验证 | 未进行 |
| 核心实验开发 | 已结束 |
| 当前阶段 | 图表、论文和答辩材料 |

冻结状态：

```text
INNOVATION1_FORMAL_CONFIRMED=True
INNOVATION2_FORMAL_CONFIRMED=True

A0_E3G_COMPATIBILITY_GO=True
A0_E3G_INDEPENDENT64_GO=True

STABILITY_SOURCE=MatterSim-5M surrogate
DFT_VERIFIED=False
PROPERTY_TARGET_VERIFIED=False
```

## 3. 两个最终创新点

### 3.1 创新点一：Multi-field Residual-driven Online Adaptive CFG

Adaptive CFG 在每个采样阶段在线读取 cell、position 和 atomic 三个字段的 conditional–unconditional residual，并用 EMA 平滑的残差信号调节 guidance scale。它不跳过 Predictor/Corrector，也不修改基础 checkpoint。

冻结参数：

```text
base_guidance=2.0
adaptive_alpha=0.50
adaptive_ema=0.95
adaptive_epsilon=1e-6
guidance_min_scale=0.0
guidance_max_scale=5.0
```

- 正式 commit：[`5de00419eea2d8a9be303638f2db8ece15a22366`](https://github.com/du17183/mattergen_v1/commit/5de00419eea2d8a9be303638f2db8ece15a22366)
- 正式分支：[`feature/convergence-aware-corrector-gating`](https://github.com/du17183/mattergen_v1/tree/feature/convergence-aware-corrector-gating)
- 正式规模：256 个配对 seeds，`20000–20255`

> **分支命名说明：** 该分支名是历史遗留名称。正式保留的方法是 Adaptive CFG，而不是后来正式 No-Go 的 Corrector Gating。方法身份以 commit、冻结配置和正式 manifest 为准。

### 3.2 创新点二：Learned-Gated Safe-Bounded Equivariant Post-Generation Crystal Refiner

简称 **Learned-Gated E3-PCR**。它不改变 MatterGen 的采样轨迹、原子种类或晶格；生成完成后，129 参数的 invariant MLP Gate 根据 14 个结构/CHGNet 汇总特征决定是否执行最多五步位置精修。更新使用等变的 force-vector 方向，并由 trust radius、energy backtracking、短距离检查和 exact fallback 保护。

冻结配置：

```text
Q3_PARAMETERS=129
GATE_INPUT_DIM=14
GATE_HIDDEN_DIM=8
GATE_THRESHOLD=0.5
REFINEMENT_STEPS=5
POSITION_ETA=0.01
PER_STEP_RADIUS=0.02 Å
MAX_CUMULATIVE_DISPLACEMENT=0.10 Å
BACKTRACK_MAX=3
```

- 正式分支：[`feature/q3-e3-pcr-formal256`](https://github.com/du17183/mattergen_v1/tree/feature/q3-e3-pcr-formal256)
- 正式 commit：[`0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`](https://github.com/du17183/mattergen_v1/commit/0275cbf08ed3c6321cea7d06f7a3a8edb83b7483)
- 正式 seeds：`40000–40255`，与 Q3 Gate 训练 seeds `20000–20063` 交集为 0

## 4. 两个创新点的关系与完整流程

Adaptive CFG 是完整方法的**共享上游采样模块**；E3-PCR 是独立的生成后处理模块。E3-PCR 可以接在原始 MatterGen 或 Adaptive CFG 后面。创新点一不是“所有历史分支的公共代码”。

```text
Condition
   ↓
MatterGen Predictor/Corrector
   ↓
Adaptive CFG
   ↓
Generated Crystal
   ↓
Learned Gate
 ┌─┴────────────┐
Gate-on       Gate-off
   ↓              ↓
E3-PCR       Exact fallback
 └──────┬───────┘
        ↓
Final Crystal
```

这种串联方式把“条件引导”与“局部物理几何修正”解耦：前者改变采样阶段 guidance，后者只对已生成结构做小幅、有界的位置更新。

## 5. 正式实验结果

### 5.1 创新点一正式 256

相对 C0：

| 指标 | 变化 |
|---|---:|
| 平均 E-hull | −0.003435 eV/atom |
| Stable | +5.859 pp |
| NUS | +3.516 pp |

这些方向性结果支持冻结的创新点一结论，但**配对统计未达到显著性**，不得写成“统计显著提升”。同时，Stable/E-hull 只来自 MatterSim-5M 代理评价，不是 DFT 或真实目标属性证明。

### 5.2 创新点二独立正式 256

| 指标 | C0 | Always-on | Learned-gated |
|---|---:|---:|---:|
| 预松弛最大力（eV/Å） | 0.342964 | 0.243956 | 0.263107 |
| 相对变化 | — | −28.87% | −23.28% |
| Relaxation RMSD（Å） | 0.049390 | 0.045057 | 0.045937 |
| Stable | 44.531% | 44.531% | 44.531% |
| NUS | 22.266% | 22.266% | 22.266% |

Gate 机制：

| 指标 | Always-on | Learned-gated |
|---|---:|---:|
| Refinement rate | 100% | 66.406% |
| Harm rate | 25.391% | 18.359% |
| Low-force harm rate | 29.688% | 17.969% |
| Mean displacement | 0.010968 Å | 0.007580 Å |

Always-on 获得更大的平均最大力下降；Learned Gate 的价值是减少精修覆盖率、有害修改和低初始力样本的恶化，而不是追求最大的平均降幅。

## 6. 两次独立组合验证

组合方法为 `A0 + E3-G`，即 Adaptive CFG 生成后连接 Learned-Gated E3-PCR。

| Cohort | 分支 / commit | Seeds | 主要结果 |
|---|---|---:|---|
| 独立组合验证一 | [`feature/a0-e3g-compatibility64`](https://github.com/du17183/mattergen_v1/tree/feature/a0-e3g-compatibility64) / [`ba2303c`](https://github.com/du17183/mattergen_v1/commit/ba2303c284210fdae0a35bb0153a8ef3af45a54c) | 41000–41063 | 最大力 −27.10%；RMSD −1.93%；E-hull 基本不变；Stable/NUS 不变 |
| 全新独立重复验证二 | [`feature/a0-e3g-independent64`](https://github.com/du17183/mattergen_v1/tree/feature/a0-e3g-independent64) / [`22e1db7`](https://github.com/du17183/mattergen_v1/commit/22e1db74a59476562f1f746cd4210b9420cbdf05) | 50000–50063 | 最大力 −19.02%；RMSD −1.66%；bootstrap CI `[-0.10221,-0.01070]`；Wilcoxon `p=0.000587`；W/T/L `35/18/11` |

两组完全独立的 64-seed 实验均为正向，但效应大小不同，说明组合效果可以泛化，同时受到样本分布影响。它们必须作为两个 cohort 分别报告，**不能包装成预注册的 128-seed 正式实验**。

## 7. 负面实验与研究路线概览

| 路线 | 核心思路 | 状态 | 主要原因 |
|---|---|---|---|
| Adaptive CFG | 在线多字段 Guidance | GO | E-hull、Stable、NUS 改善 |
| Learned-Gated E3-PCR | 有界后生成精修 | GO | 最大力下降并减少有害精修 |
| Residual Reuse | 复用无条件残差 | No-Go | 真实端到端加速不足 |
| Corrector Gating | 跳过 Corrector | No-Go | 加速约 1.5×但正式质量明显下降 |
| Budget-aware Gating | 自适应 Corrector 预算 | No-Go | 加速与质量未同时过门槛 |
| FN-PRA | 冻结物理特征对齐 | No-Go | 部分指标改善但稳定性和组成恶化 |
| CrystalREPA | Teacher 表示对齐 | No-Go | E-hull、RMSD 和 Metastable 恶化 |
| RP-QTFG | CHGNet 梯度在线引导 | No-Go | 在线轨迹偏移、RMSD 恶化、延迟增加 |
| CG-TDR | 学习终端残差精修 | No-Go | Teacher 残差方向不能可靠泛化 |
| Q1 UQ-PQR | 不确定性感知排序 | No-Go | Novel 下降 |
| Q2 RFR | 放松性/失败风险路由 | No-Go | Novel 与 Unique 明显下降 |
| Q4 CPRC | 跨势模型残差校准 | No-Go | Novel 下降 |
| Q5 CQPS | 多候选质量选择 | No-Go | 独立数据 Novel 下降 |
| Q6 NS-SetRank | 候选集合排序 | No-Go | 独立数据 Novel 下降 |
| GPU 优化路线 | Batch/BF16/compile/static graph/MPS 等 | No-Go 或工程保留 | 未同时达到冻结的加速、精度和质量门槛 |

详细负面实验见：

- [Negative Results Summary](docs/experiments/negative_results_summary.md)
- [Experiment Lineage](thesis_archive/EXPERIMENT_LINEAGE.md)

## 8. 训练—测试泄漏诊断

- 分支：[`experiment/a0-e3g-leakage-diagnostic256`](https://github.com/du17183/mattergen_v1/tree/experiment/a0-e3g-leakage-diagnostic256)
- commit：[`01e9b2c30e5c58e05eaae908ba291c518b977d03`](https://github.com/du17183/mattergen_v1/commit/01e9b2c30e5c58e05eaae908ba291c518b977d03)
- Training overlap：`20000–20063`
- Held-out：`20064–20255`
- Training-overlap harm rate：0%
- Held-out harm rate：16.15%
- Fisher exact `p=6.87e-5`

结论：泄漏没有明显夸大平均最大力改善，但显著高估了 Gate 安全性。

```text
Training-overlap: DIAGNOSTIC_ONLY
Mixed 256: INVALID_FOR_INDEPENDENT_CLAIMS
Held-out 192: SUPPLEMENTARY_ONLY
```

真实 seed 被保留；不得匿名化 seed、隐藏重叠或把 mixed cohort 冒充独立验证。

## 9. 分支与实验地图

| 分支或 Tag | 工作 | 状态 | 论文用途 |
|---|---|---|---|
| [`feature/convergence-aware-corrector-gating`](https://github.com/du17183/mattergen_v1/tree/feature/convergence-aware-corrector-gating) | Adaptive CFG 正式实现；名称历史遗留 | GO | 创新点一 |
| [`feature/q3-e3-pcr-formal256`](https://github.com/du17183/mattergen_v1/tree/feature/q3-e3-pcr-formal256) | E3-PCR 正式 256 | GO | 创新点二 |
| [`feature/a0-e3g-compatibility64`](https://github.com/du17183/mattergen_v1/tree/feature/a0-e3g-compatibility64) | 独立兼容性 1 | GO | 组合实验 |
| [`feature/a0-e3g-independent64`](https://github.com/du17183/mattergen_v1/tree/feature/a0-e3g-independent64) | 独立复现 2 | GO | 组合复现 |
| [`experiment/a0-e3g-leakage-diagnostic256`](https://github.com/du17183/mattergen_v1/tree/experiment/a0-e3g-leakage-diagnostic256) | 泄漏诊断 | Diagnostic | 消融/局限性 |
| [`feature/a0-e3g-formal256`](https://github.com/du17183/mattergen_v1/tree/feature/a0-e3g-formal256) | 旧 A0 数据资格审计 | Source incomplete | 数据资格说明 |
| [`archive/thesis-analysis-package-v1`](https://github.com/du17183/mattergen_v1/tree/archive/thesis-analysis-package-v1) | 论文分析归档 | Completed | 笔记本分析 |
| [`thesis-analysis-v1`](https://github.com/du17183/mattergen_v1/tree/thesis-analysis-v1) | 冻结 Tag | Frozen | 可复现 v1 |

更细的路线选择和 No-Go 血缘见 [Experiment Lineage](thesis_archive/EXPERIMENT_LINEAGE.md)。分支名只表示历史开发载体；正式方法身份以冻结 commit、配置和 manifest 为准。

## 10. 论文分析归档

论文 CPU 分析所需的逐 seed 指标、统计脚本、结果表、图表和数据资格说明集中在：

- [归档总览](thesis_archive/README.md)
- [笔记本分析说明](thesis_archive/README_FOR_LAPTOP.md)
- [最终实验清单](thesis_archive/FINAL_EXPERIMENT_MANIFEST.md)
- [结论—证据矩阵](thesis_archive/CLAIM_EVIDENCE_MATRIX.md)
- [数据字典](thesis_archive/DATA_DICTIONARY.md)
- [实验谱系](thesis_archive/EXPERIMENT_LINEAGE.md)
- [局限性](thesis_archive/LIMITATIONS.md)
- [未上传工件](thesis_archive/ARTIFACTS_NOT_IN_GITHUB.md)

归档 v1 由 Tag [`thesis-analysis-v1`](https://github.com/du17183/mattergen_v1/tree/thesis-analysis-v1) 冻结。本轮后续文档提交只存在于归档分支，不移动 v1 Tag。

## 11. 笔记本 CPU 分析方法

Windows PowerShell：

```powershell
git clone https://github.com/du17183/mattergen_v1.git
cd mattergen_v1
git switch archive/thesis-analysis-package-v1

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r thesis_archive/requirements-analysis.txt

python thesis_archive/analysis/validate_archive.py
python thesis_archive/analysis/recompute_statistics.py
python thesis_archive/analysis/build_result_tables.py
python thesis_archive/analysis/generate_figures.py
```

这些命令不需要 CUDA、MatterGen checkpoint、MatterSim 权重或服务器 Conda 环境。

## 12. 数据与结论使用规则

1. 主结论只使用 manifest 标记为正式或独立的数据。
2. 两批 64-seed 组合验证必须同时报告，不能只选效果更大的批次。
3. Training-overlap 只用于泄漏诊断；Mixed 256 不能作为独立结果。
4. `feature/a0-e3g-formal256` 的 source-incomplete 审计没有产生效应估计，不能写成方法 No-Go。
5. Stable、E-hull、NUS 和 RMSD 均基于 MatterSim-5M surrogate；不得表述为 DFT 证明。
6. 未完成目标磁密度独立验证；不得把条件输入或 CHGNet proxy 当成真实属性命中证明。
7. 逐 seed 统计、报告、manifest 和 checksum 优先于 README 摘要。

## 13. 未上传的大型文件

GitHub 已包含：

```text
代码
配置
逐 seed 指标
统计脚本
结果表
图表
报告
manifest
checksum
```

GitHub 未包含：

```text
MatterGen 权重
MatterSim 权重
E3-PCR checkpoint 二进制
Conda 环境
原始数据集
大型结构缓存
完整松弛缓存
Teacher cache
大型日志
```

论文 CPU 分析所需的代码、逐 seed 数据和说明已完整归档；完整模型重运行所需的大型权重、环境和缓存未上传。大小、哈希和重建说明见[未上传工件清单](thesis_archive/ARTIFACTS_NOT_IN_GITHUB.md)。

## 14. 局限性

- `STABILITY_SOURCE=MatterSim-5M surrogate`，没有 DFT 复核。
- `PROPERTY_TARGET_VERIFIED=False`，目标磁密度没有独立高精度验证。
- Adaptive CFG 正式方向性改善的配对统计未达到显著性。
- Learned Gate 的 129 参数分类器性能有限；论文贡献应表述为 Gate、等变更新、trust region、backtracking 与 exact fallback 的完整系统。
- 两批组合验证样本量均为 64，效应大小存在 cohort 差异。
- E3-PCR checkpoint 二进制不在 GitHub；归档支持 CPU 分析重算，不等价于从零重跑模型。
- 负面实验中的部分完整日志/缓存只保留在服务器或历史分支；归档总结明确标注证据边界。

详见[归档局限性](thesis_archive/LIMITATIONS.md)与[负面实验总结](docs/experiments/negative_results_summary.md)。

## 15. 当前下一步

核心实验开发已经结束。推荐按以下顺序收尾：

1. 用归档脚本重算最终统计、表格和图，锁定论文数字；
2. 以两个创新点、两个独立组合 cohort 和泄漏诊断构建论文主线；
3. 将 Corrector Gating、RP-QTFG、CG-TDR 和候选排序失败写入负面实验/消融；
4. 明确区分 MatterSim surrogate 与 DFT 结论；
5. 如还有额外算力，只优先补 DFT 小规模复核和目标磁密度独立验证，不再回溯调参。

## 16. 如何使用各分支 README

每个活跃研究分支的根 `README.md` 都是独立实验档案，包含：研究问题、算法执行链、真实实现文件、数据范围、逐 seed 证据、关键结果、复现/状态命令和科学边界。

建议阅读顺序：

1. 先在本页确认该路线属于正式 GO、工程收益、科学 No-Go、泄漏诊断还是 source audit。
2. 点击分支名称，阅读该分支自己的根 README。
3. 继续进入 README 链接的 frozen manifest、逐结构 CSV、paired statistics 和 final report。
4. 需要复现实验时先运行专项测试和 `status`，再决定是否使用服务器专用 runner。
5. 不要点击旧 commit 后把历史 README 当作分支当前说明；固定论文版本时才使用报告记录的 frozen commit。

README 是导航和解释层，CSV/JSON、manifest、checksum 与冻结报告才是论文数字的最终证据层。

## 17. 论文规划与可复现图表

学校登记论文题目固定为《基于深度学习的材料逆向生成》。论文工作区位于[`thesis/`](thesis/README.md)，采用预训练MatterGen作为条件晶体扩散生成基线，并包含冻结研究定位、最终章节编号、章节证据包、12张论文级图、10组结果表和CPU-only复现脚本。

关键入口：

- [学校登记论文题目](thesis/THESIS_TITLE_FINAL.md)
- [冻结论文研究定位](thesis/THESIS_POSITIONING_FINAL.md)
- [MatterGen命名与归属规则](thesis/MATTERGEN_NAMING_POLICY.md)
- [最终章节、图表与公式编号](thesis/CHAPTER_NUMBERING_FINAL.md)
- [第3–6章可追溯证据包](thesis/evidence_packs/README.md)
- [证据包真实性与统计验证报告](thesis/evidence_packs/EVIDENCE_PACK_VALIDATION.md)
- [最终论文结论](thesis/PAPER_CLAIMS_FINAL.md)
- [旧版论文目录（历史参考）](thesis/THESIS_OUTLINE.md)
- [论文写作工作包状态](thesis/WRITING_PACKAGE_STATUS.md)
- [正文/附录图表安排](thesis/MAIN_TEXT_APPENDIX_PLAN.md)
- [七张核心图 V2 重绘交接](thesis/figures/CORE_FIGURES_V2_REDRAW.md)
- [旧版章节写作计划（历史编号）](thesis/CHAPTER_WRITING_PLAN.md)
- [旧版图表规划（历史编号）](thesis/FIGURE_TABLE_PLAN.md)
- [图索引](thesis/figures/generated/figure_index.md)
- [表索引](thesis/tables/table_index.md)
- [Scientific Skill 使用证据](thesis/SKILL_USAGE_MANIFEST.md)

复现命令：

```bash
python -m pip install -r thesis_archive/requirements-analysis.txt
python thesis/scripts/generate_all.py
python thesis/scripts/validate_outputs.py
```

该流程只读取 `thesis_archive/`，不需要 GPU、MatterGen/MatterSim 权重或服务器环境，也不会启动新实验。

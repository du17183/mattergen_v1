# 最终章节、图表与公式编号

本文件是论文写作阶段的最终编号依据。旧版 `THESIS_OUTLINE.md`、历史正文草稿和已生成图表仍保留原文件名以保证脚本可复现；合稿时必须按本文件更新正文标题、图表题注和交叉引用，不直接重命名生成文件。

## 1. 最终章节结构

| 章节 | 中文正式标题 | 英文工作标题 |
|---|---|---|
| 第1章 | 绪论 | Chapter 1 Introduction |
| 第2章 | 深度学习材料生成相关理论与研究现状 | Chapter 2 Theoretical Foundations and Related Work on Deep Learning-Based Material Generation |
| 第3章 | 条件扩散晶体生成基线与评价体系 | Chapter 3 Conditional Diffusion Baseline for Crystal Generation and Evaluation Protocol |
| 第4章 | 多字段残差驱动的在线自适应条件引导方法 | Chapter 4 Multi-field Residual-driven Online Adaptive Guidance |
| 第5章 | 学习门控的晶体生成后质量优化方法 | Chapter 5 Learned-Gated Post-generation Quality Optimization |
| 第6章 | 组合验证、消融实验与结果讨论 | Chapter 6 Combined Validation, Ablation Studies, and Discussion |
| 第7章 | 总结与展望 | Chapter 7 Conclusions and Future Work |

## 2. 各章内容边界

### 第1章 绪论

材料逆向设计背景、深度学习材料生成意义、晶体生成难点、固定条件引导问题、生成后局部物理不一致问题、研究内容、两个创新点和论文结构。不集中堆叠实验数字。

### 第2章 深度学习材料生成相关理论与研究现状

材料逆向设计、VAE/GAN/Flow/Diffusion、晶体表示、晶体生成模型、属性条件生成、CFG、等变建模、机器学习原子势和材料评价指标。MatterGen 是相关工作之一，不作为章节标题。

### 第3章 条件扩散晶体生成基线与评价体系

逆向生成任务定义、条件扩散晶体生成、MatterGen 实现、`dft_mag_density` 条件任务、冻结实验环境、数据和 seed 划分、MatterSim-5M 代理评价、指标、统计检验及数据独立性控制。

### 第4章 多字段残差驱动的在线自适应条件引导方法

只讲创新点一：固定 CFG 局限、三字段残差、EMA、在线 guidance 更新、完整 Predictor–Corrector、复杂度、正式结果及限制。

### 第5章 学习门控的晶体生成后质量优化方法

只讲创新点二：最大力问题、14 维风险特征、129 参数 Gate、安全有界等变位置更新、trust region、backtracking、安全检查、exact fallback、正式 256-seed 结果及 Always-on/Random Gate 消融。

### 第6章 组合验证、消融实验与结果讨论

两个创新点功能分工、两组独立组合验证、Gate 风险分析、训练—测试泄漏诊断、代表性 No-Go 路线、计算开销、可复现性和局限。

### 第7章 总结与展望

工作总结、两个创新点、实验结论、局限性和未来工作。

## 3. 第3章最终结构

1. 3.1 材料逆向生成任务定义
2. 3.2 条件扩散晶体生成方法
3. 3.3 MatterGen 基线模型与实验配置
4. 3.4 目标属性条件任务
5. 3.5 数据划分与实验协议
6. 3.6 MatterSim 代理评价体系
7. 3.7 评价指标与统计方法
8. 3.8 数据独立性和真实性控制
9. 3.9 本章小结

## 4. 最终图编号与现有产物映射

| 最终编号 | 最终图名 | 当前来源/状态 | 说明 |
|---|---|---|---|
| 图3-1 | 本文总体技术路线 | 待按本文件重绘；可参考旧 Figure 1/2/3 | 仅作概念整合，不新增实验数字 |
| 图3-2 | 条件晶体扩散生成与代理评价流程 | 待按第3章证据包重绘 | 基线模块标为“条件晶体扩散生成基线”，图注注明 MatterGen 实现 |
| 图3-3 | 实验数据与 seed 血缘 | 旧 Figure 4：`fig04_experiment_lineage.pdf` | Mixed 256 只标诊断 |
| 图4-1 | 多字段残差驱动 Adaptive CFG 流程 | 旧 Figure 2：`fig02_adaptive_cfg_mechanism.pdf` | 三字段 RMS 聚合后使用共享 scale |
| 图4-2 | Adaptive CFG 正式实验结果 | 旧 Figure 5：`fig05_adaptive_cfg_results.pdf` | 必须显示非显著置信区间 |
| 图5-1 | Learned-Gated E3-PCR 总体框架 | 旧 Figure 3：`fig03_e3pcr_mechanism.pdf` | 14→8→1、位置更新、fallback |
| 图5-2 | E3-PCR 三臂最大力比较 | 旧 Figure 6：`fig06_e3pcr_force_formal256.pdf` | C0/E3-A/E3-G |
| 图5-3 | Learned Gate 安全消融 | 旧 Figure 7：`fig07_gate_safety_ablation.pdf` | 同时呈现 coverage、harm 和 retained gain |
| 图5-4 | Gate 置信度与实际改善关系 | 旧 Figure 8：`fig08_gate_confidence_force_gain.pdf` | 描述性，不声称因果或校准充分 |
| 图6-1 | 两个创新点的组合关系 | 旧 Figure 1：`fig01_full_method_architecture.pdf` | A0 后接 E3-G；不合并统计 cohort |
| 图6-2 | 两组独立组合验证 Forest plot | 旧 Figure 9：`fig09_combination_replication_forest.pdf` | 两个效应分别报告 |
| 图6-3 | 独立复现 cohort 配对结果 | 旧 Figure 10：`fig10_independent64_pairplot.pdf` | 说明 1e-6 算法语义平局 |
| 图6-4 | 训练—测试泄漏诊断 | 旧 Figure 11：`fig11_leakage_diagnostic.pdf` | 仅诊断，不作独立主结论 |
| 图6-5 | 代表性 No-Go 路线 | 旧 Figure 12：`fig12_negative_routes_summary.pdf` | 不包装为论文创新 |

旧文件名继续由原 CPU 脚本生成。最终编号在论文排版层应用。

## 5. 最终表编号与现有产物映射

| 最终编号 | 最终表名 | 当前来源/状态 |
|---|---|---|
| 表3-1 | 方法代号及定义 | `TERMINOLOGY.md`，排版时生成 |
| 表3-2 | 正式实验及数据划分 | 旧 Table 01：`01_experiment_manifest.md/csv` |
| 表3-3 | 评价指标 | `chapter3/metrics_definitions.md`，排版时生成 |
| 表3-4 | 统计检验方法 | 第3章证据包和验证脚本说明，排版时生成 |
| 表4-1 | Adaptive CFG 冻结参数 | `adaptive_cfg_final.yaml`，排版时生成 |
| 表4-2 | Adaptive CFG 正式结果 | 旧 Table 02：`02_innovation1.md/csv` |
| 表5-1 | E3-PCR 冻结配置 | `e3_pcr_final.yaml`，排版时生成 |
| 表5-2 | 独立 256-seed 正式结果 | 旧 Table 03：`03_innovation2.md/csv` |
| 表5-3 | Always-on 与 Learned Gate 消融 | 旧 Table 04：`04_gate_ablation.md/csv` |
| 表6-1 | 两次独立组合验证 | 旧 Table 05、06、07；两 cohort 分列且不 pooled |
| 表6-2 | 泄漏诊断结果 | 旧 Table 08：`08_leakage_diagnostic.md/csv` |
| 表6-3 | 代表性负面实验 | 旧 Table 09：`09_negative_results.md/csv` |
| 表6-4 | 方法计算开销 | 仅允许汇总已有冻结运行记录；当前未形成统一最终表 |

## 6. 公式编号

- 第4章 Adaptive CFG 公式按 `FORMULA_REGISTRY.md` 中 F4 顺序编为式（4-1）至式（4-8）。
- 第5章 E3-PCR 公式按 F5 顺序编为式（5-1）至式（5-7）。
- 第3章指标定义在正文出现时按实际排版顺序编号；证据身份仍使用 F3 ID。
- 解释性公式必须保留“interpreted”资格，不能改称源码逐行等价。

## 7. 编号实施规则

1. 不重命名旧图、表和 source-data 文件，避免破坏生成脚本。
2. 正文使用最终编号，图表交叉表同时保留旧 Figure/Table 身份。
3. 图3-1、图3-2及尚未排版的表只允许基于当前证据重绘，不引入新实验数字。
4. 表6-4若缺乏统一冻结口径，应标为待整理，不得估算或补造。

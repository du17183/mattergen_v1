# 最终论文图表编号与现有产物映射

旧生成文件名不变，正文按最终编号引用。图3-1、图3-2和标为“排版时生成”的表只可重组现有证据，不得引入新实验数字。

| 最终编号 | 内容 | 现有产物或来源 | 源数据 | 写作防护 |
|---|---|---|---|---|
| 图3-1 | 本文总体技术路线 | 待重绘；参考旧Figure 1/2/3 | 概念图，不新增统计 | 基线标为条件晶体扩散生成基线，图注注明MatterGen实现 |
| 图3-2 | 条件晶体扩散生成与代理评价流程 | 待重绘 | 第3章证据包 | MatterGen/MatterSim/CHGNet角色分离 |
| 图3-3 | 实验数据与seed血缘 | `fig04_experiment_lineage.pdf`（旧Figure 4） | `fig04_experiment_lineage.csv` | Mixed 256仅诊断 |
| 图4-1 | 多字段残差驱动Adaptive CFG流程 | `fig02_adaptive_cfg_mechanism.pdf`（旧Figure 2） | `fig02_adaptive_cfg_mechanism.csv` | 三字段RMS聚合后共享scale |
| 图4-2 | Adaptive CFG正式实验结果 | `fig05_adaptive_cfg_results.pdf`（旧Figure 5） | `fig05_adaptive_cfg_results.csv` | 必须显示非显著CI |
| 图5-1 | Learned-Gated E3-PCR总体框架 | `fig03_e3pcr_mechanism.pdf`（旧Figure 3） | `fig03_e3pcr_mechanism.csv` | 14→8→1、position-only、fallback |
| 图5-2 | E3-PCR三臂最大力比较 | `fig06_e3pcr_force_formal256.pdf`（旧Figure 6） | `fig06_e3pcr_force_formal256.csv` | Always-on均值效果更大 |
| 图5-3 | Learned Gate安全消融 | `fig07_gate_safety_ablation.pdf`（旧Figure 7） | `fig07_gate_safety_ablation.csv` | coverage/harm/retained gain并列 |
| 图5-4 | Gate置信度与实际改善关系 | `fig08_gate_confidence_force_gain.pdf`（旧Figure 8） | `fig08_gate_confidence_force_gain.csv` | 描述性，不作因果/充分校准证明 |
| 图6-1 | 两个创新点的组合关系 | `fig01_full_method_architecture.pdf`（旧Figure 1） | `fig01_full_method_architecture.csv` | A0后接E3-G，不pooled cohort |
| 图6-2 | 两组独立组合验证Forest plot | `fig09_combination_replication_forest.pdf`（旧Figure 9） | `fig09_combination_replication_forest.csv` | 两个效应分别报告 |
| 图6-3 | 独立复现cohort配对结果 | `fig10_independent64_pairplot.pdf`（旧Figure 10） | `fig10_independent64_pairplot.csv` | 说明算法语义平局 |
| 图6-4 | 训练—测试泄漏诊断 | `fig11_leakage_diagnostic.pdf`（旧Figure 11） | `fig11_leakage_diagnostic.csv` | 只作诊断 |
| 图6-5 | 代表性No-Go路线 | `fig12_negative_routes_summary.pdf`（旧Figure 12） | `fig12_negative_routes_summary.csv` | 不包装成创新 |
| 表3-1 | 方法代号及定义 | `thesis/TERMINOLOGY.md`，排版时生成 | 冻结术语 | MatterGen归属基线 |
| 表3-2 | 正式实验及数据划分 | `01_experiment_manifest.md/csv`（旧Table 01） | manifest | seed与证据资格 |
| 表3-3 | 评价指标 | `chapter3/metrics_definitions.md`，排版时生成 | 数据字典/评价源码 | surrogate限定 |
| 表3-4 | 统计检验方法 | 第3章证据包，排版时生成 | 验证脚本说明 | 不新增统计口径 |
| 表4-1 | Adaptive CFG冻结参数 | `adaptive_cfg_final.yaml`，排版时生成 | 冻结配置 | 参数不得改写 |
| 表4-2 | Adaptive CFG正式结果 | `02_innovation1.md/csv`（旧Table 02） | 正式256 | 配对统计未显著 |
| 表5-1 | E3-PCR冻结配置 | `e3_pcr_final.yaml`，排版时生成 | 冻结配置 | 14→8→1、129参数 |
| 表5-2 | 独立256-seed正式结果 | `03_innovation2.md/csv`（旧Table 03） | 正式256 | surrogate指标 |
| 表5-3 | Always-on与Learned Gate消融 | `04_gate_ablation.md/csv`（旧Table 04） | 正式/补充消融 | Always-on与Gate均如实报告 |
| 表6-1 | 两次独立组合验证 | 旧Table 05/06/07 | 两个64-seed cohort | 分别报告，不pooled |
| 表6-2 | 泄漏诊断结果 | `08_leakage_diagnostic.md/csv`（旧Table 08） | 诊断数据 | formal=false |
| 表6-3 | 代表性负面实验 | `09_negative_results.md/csv`（旧Table 09） | 负面结果归档 | 来源恢复程度不同 |
| 表6-4 | 方法计算开销 | 尚无统一最终表 | 仅可汇总冻结运行记录 | 不估算、不补造 |

公式编号：F4按注册表顺序映射为式（4-1）至式（4-8）；F5映射为式（5-1）至式（5-7）。解释性公式不得标为exact。

重绘说明见`thesis/figures/CORE_FIGURES_V2_REDRAW.md`和`thesis/figures/REDRAW_GUIDE.md`。

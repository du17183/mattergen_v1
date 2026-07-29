# 图表规划

## 图

| 图 | 文件 | 章节 | 科学目的 | 数据 | 技能 |
|---:|---|---|---|---|---|
| 1 | `fig01_full_method_architecture` | 1/5 | 两项创新的完整链条 | configs + lineage | schematics + visualization |
| 2 | `fig02_adaptive_cfg_mechanism` | 3 | 三字段残差在线更新 | Adaptive CFG config | schematics |
| 3 | `fig03_e3pcr_mechanism` | 4 | Gate、trust region 与 fallback | E3-PCR config | schematics |
| 4 | `fig04_experiment_lineage` | 5/6 | 证据资格与 seed 血缘 | experiment lineage | schematics |
| 5 | `fig05_adaptive_cfg_results` | 3 | C1 正向趋势与非显著性 | innovation1 per-seed | stats + visualization |
| 6 | `fig06_e3pcr_force_formal256` | 4 | C2 三臂与配对效应 | innovation2 per-seed | stats + visualization |
| 7 | `fig07_gate_safety_ablation` | 4 | C3 覆盖/伤害/位移/收益 | formal summary | stats + visualization |
| 8 | `fig08_gate_confidence_force_gain` | 4 | confidence 与真实改善关系 | innovation2 per-seed | stats + visualization |
| 9 | `fig09_combination_replication_forest` | 5 | C4/C5 两 cohort 并列 | paired statistics | stats + visualization |
| 10 | `fig10_independent64_pairplot` | 5 | 第二 cohort 逐 seed 配对 | compatibility_2 | stats + visualization |
| 11 | `fig11_leakage_diagnostic` | 6 | C6 平均效应/安全泄漏分离 | leakage per-seed | stats + visualization |
| 12 | `fig12_negative_routes_summary` | 6 | 失败路线与停止证据 | experiment lineage | schematics + visualization |

## 表

| 主表 | 对应工作簿 sheet | 章节 | 内容 |
|---:|---|---|---|
| 1 | `01_Experiment_Manifest` | 2/3 | 方法、seed、n、checkpoint、commit、评价器、资格 |
| 2 | `02_Innovation1` | 3 | C0 vs Adaptive CFG |
| 3 | `03_Innovation2` | 4 | C0/E3-A/E3-G 质量与力 |
| 4 | `04_Gate_Ablation` | 4 | Gate 覆盖、伤害、位移和收益 |
| 5 | `07_Combination_Summary` | 5 | 两次独立组合验证 |
| 6 | `09_Negative_Results` | 6 | 代表性 No-Go 路线 |
| 7 | `10_Paper_Claims` + qualification | 6/7 | 结论与使用资格 |

## 排版原则

- 示意图优先双栏宽 7.1 in；正文单图可按学校模板缩放。
- 统计图不截断坐标轴以夸大效应；保留零参考线。
- 主图只承载一条核心结论，细节放表格和补充材料。
- MatterSim 图注必须包含代理势与无 DFT 声明。


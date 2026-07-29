# 第3章证据包：条件扩散晶体生成基线与评价体系

> 本文件是写作证据，不是完整论文正文。任何项目事实必须回指 source_id；未支持内容不得由通用知识补齐。

## 1. 本章研究目标

在学校登记题目《基于深度学习的材料逆向生成》下，定义材料逆向生成任务的本文范围、条件扩散晶体生成方法、预训练MatterGen基线、目标条件、数据资格、代理评价和统计口径。

## 2. 本章回答的核心问题

- “材料逆向生成”在本文中具体指什么，研究对象为何限定为周期晶体？
- 条件扩散生成方法与预训练MatterGen实验基线是什么关系？
- C0、A0、E3-A、E3-G和完整方法分别是什么？
- dft_mag_density=0.1是何种输入，当前是否有独立属性真值？
- MatterSim-5M指标可以支持哪些相对结论，不能支持哪些结论？
- 各正式、补充、诊断seed如何隔离？

## 3. 最终二级和三级标题

- 3.1 材料逆向生成任务定义
- 3.2 条件扩散晶体生成方法
- 3.3 MatterGen基线模型与实验配置
- 3.4 目标属性条件任务
- 3.5 数据划分与实验协议
- 3.6 MatterSim代理评价体系
- 3.7 评价指标与统计方法
- 3.8 数据独立性和真实性控制
- 3.9 本章小结

## 4. 与前后章节的关系

承接第2章材料生成相关理论，为第4章Adaptive CFG、第5章Learned-Gated E3-PCR及第6章组合/审计提供统一基线、扩展接口和评价口径。

## 5. 可使用的源码事实

- 学校正式论文题目为“基于深度学习的材料逆向生成”，不得改写。
- 本文把材料逆向生成限定为根据目标属性条件生成周期晶体候选。
- MatterGen是本文采用的预训练条件晶体扩散生成基线、实验框架和实现载体，不是本文提出的方法。
- C0为原始条件晶体扩散生成基线，由预训练dft_mag_density MatterGen实现，constant CFG scale=2.0；完整Predictor/Corrector、FP32、batch_size=1。
- A0=C0+Multi-field Residual-driven Online Adaptive CFG。
- E3-A/E3-G从同一个C0结构分别执行Always-on或Learned-Gated位置精修。
- MatterGen是生成模型，MatterSim-5M是论文评价代理，CHGNet只用于Gate特征与E3-PCR局部更新；三者不得混同。
- 条件目标为dft_mag_density=0.1，但PROPERTY_TARGET_VERIFIED=False。
- STABILITY_SOURCE=MatterSim-5M surrogate；DFT_VERIFIED=False。

## 6. 可使用的配置和参数

- C0 guidance_scale=2.0
- Predictor/Corrector=full
- precision=FP32
- batch_size=1
- dft_mag_density target=0.1
- MatterSim stability threshold=0.1 eV/atom

## 7. 公式与变量定义

| ID | 公式 | 性质 | 代码 |
| --- | --- | --- | --- |
| F3_MAX_FORCE | $F_{\max}=\max_i\lVert\mathbf F_i\rVert_2$ | exact | `research/q3_frozen64.py::relax worker pre_relax_max_force_ev_ang` |
| F3_RMSD | $\operatorname{RMSD}=\operatorname{MatcherRMSD}(X_{\mathrm{relaxed}},X_{\mathrm{initial}})\;[\AA]$ | interpreted | `mattergen/evaluation/utils/utils.py::compute_rmsd_angstrom` |
| F3_STABLE | $\mathrm{Stable}=\mathbb 1[E_{\mathrm{hull}}\le 0.1\;\mathrm{eV/atom}]$ | exact | `mattergen/evaluation/metrics/energy.py::EnergyCapability.is_stable` |
| F3_NUS | $\mathrm{NUS}=\mathrm{Novel}\land\mathrm{Unique}\land\mathrm{Stable}$ | exact | `mattergen/evaluation/metrics/energy.py::FracNovelUniqueStableStructures.compute_pre_aggregation_values` |
| F3_HARM | $\mathrm{Harm}=\mathbb 1[F_{\max}^{selected}-F_{\max}^{base}>10^{-6}]$ | exact | `research/q3_formal256.py::FORCE_HARM_EPSILON and gate mechanism analysis` |

公式的完整变量、exact/interpreted资格见 `../FORMULA_REGISTRY.md`。

## 8. 实验设计

- Gate training: 20000–20063。
- Adaptive CFG formal: 20000–20255，n=256；与Gate训练重叠不影响创新点一，因为A0不使用Q3 Gate。
- E3-PCR formal: 40000–40255，n=256，和Gate训练交集为0。
- 组合cohort 1: 41000–41063，n=64，独立。
- 组合cohort 2: 50000–50063，n=64，完全独立。
- Leakage overlap 20000–20063仅诊断；held-out 20064–20255仅补充；Mixed 256不得用于独立结论。

## 9. 正式结果

- 本章不主张方法效果；只冻结实验身份、评价和证据资格。

## 10. 对应图表

- 图3-1：本文总体技术路线（待按证据重绘）
- 图3-2：条件晶体扩散生成与代理评价流程（待按证据重绘）
- 图3-3：实验数据与seed血缘（当前旧Figure 4）

## 11. 对应表格

- 表3-1：方法代号及定义
- 表3-2：正式实验及数据划分（当前旧Table 01）
- 表3-3：评价指标
- 表3-4：统计检验方法

## 12. 允许写入正文的结论

- MatterSim用于相同流程下的方法间代理相对比较。
- 预松弛最大力、RMSD、E-hull、Stable和NUS可按统一代理口径报告。
- 两个64-seed cohort是两次独立证据。
- 本文采用预训练MatterGen作为条件晶体扩散生成基线；本文贡献位于推理阶段Adaptive CFG和生成后Learned-Gated E3-PCR。

冻结claim：

- 力、RMSD、E-hull、Stable与NUS均来自MatterSim-5M代理评价，可用于统一相对比较。
- DFT_VERIFIED=False且PROPERTY_TARGET_VERIFIED=False；dft_mag_density=0.1是条件输入，不是已验证命中结果。

## 13. 禁止夸大的结论

- MatterSim等价于DFT。
- 条件输入0.1证明输出真实磁密度命中。
- 代理Stable证明可合成。
- 把Mixed 256或training overlap写成独立验证。
- 本文提出或训练了完整MatterGen模型。
- MatterGen原有晶体表示、扩散主干、Predictor–Corrector或原始CFG属于本文创新。

## 14. 必须主动说明的限制

- 无DFT、无实验合成验证、无目标属性独立验证。
- 同一项目数据域、单一条件checkpoint和统一代理评价器。
- 精确原始生成CLI未作为可移植归档的一部分；可使用冻结配置和manifest，不得编造命令。

## 15. 对应源码文件和函数

- MetricsStructureSummary.rmsd_from_relaxation
- compute_rmsd_angstrom
- EnergyCapability.is_stable
- FracNovelUniqueStableStructures
- structure_validity

## 16. 对应数据文件和字段

- S01_MANIFEST
- S03_DATA_DICTIONARY
- S09_ADAPTIVE_CONFIG
- S10_I1_DATA
- S17_I2_DATA
- S21_COHORT1_DATA
- S23_COHORT2_DATA
- S25_LEAK_DATA

字段定义必须联合使用 `S03_DATA_DICTIONARY`、本章专用数据文件和 `metrics_definitions.md`/`experiment_evidence.md`。

## 17. 对应commit、分支和报告

- `5de00419eea2d8a9be303638f2db8ece15a22366`
- `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`

| source_id | 类型 | 路径 | commit | 数据资格 |
| --- | --- | --- | --- | --- |
| S01_MANIFEST | manifest | `thesis_archive/FINAL_EXPERIMENT_MANIFEST.json` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative archive manifest |
| S02_CLAIMS | report | `thesis/PAPER_CLAIMS_FINAL.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative writing claim register |
| S03_DATA_DICTIONARY | report | `thesis_archive/DATA_DICTIONARY.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative archive schema |
| S04_LINEAGE | report | `thesis_archive/EXPERIMENT_LINEAGE.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative lineage narrative |
| S05_LIMITATIONS | report | `thesis/PAPER_LIMITATIONS.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | mandatory disclosure |
| S08_PC_SAMPLER | source code | `mattergen/diffusion/sampling/pc_sampler.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | formal implementation snapshot |
| S09_ADAPTIVE_CONFIG | config | `thesis_archive/configs/adaptive_cfg_final.yaml` | `5de00419eea2d8a9be303638f2db8ece15a22366` | frozen configuration |
| S10_I1_DATA | per-seed data | `thesis_archive/data/innovation1/per_seed_metrics.csv` | `5de00419eea2d8a9be303638f2db8ece15a22366` | formal independent data; seeds 20000–20255 |
| S14_E3_FROZEN_CORE | source code | `research/q3_frozen64.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | frozen core reused by formal runner |
| S15_E3_FORMAL_RUNNER | source code | `research/q3_formal256.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal run snapshot; recorded code commit 5293b4b71be88b6663bbe349f3b57694a916835f |
| S17_I2_DATA | per-seed data | `thesis_archive/data/innovation2/per_seed_metrics.csv` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal independent data; seeds 40000–40255 |
| S21_COHORT1_DATA | per-seed data | `thesis_archive/data/compatibility_1/per_seed_metrics.csv` | `ba2303c284210fdae0a35bb0153a8ef3af45a54c` | independent; seeds 41000–41063 |
| S23_COHORT2_DATA | per-seed data | `thesis_archive/data/compatibility_2/per_seed_metrics.csv` | `22e1db74a59476562f1f746cd4210b9420cbdf05` | fully independent; seeds 50000–50063 |
| S25_LEAK_DATA | per-seed data | `thesis_archive/data/leakage_diagnostic/per_seed_metrics.csv` | `01e9b2c30e5c58e05eaae908ba291c518b977d03` | diagnostic only; mixed 256 invalid for independent claims |
| S27_EVAL_STRUCTURE | source code | `mattergen/evaluation/metrics/structure.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | official evaluator implementation in repository snapshot |
| S28_EVAL_ENERGY | source code | `mattergen/evaluation/metrics/energy.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | official evaluator implementation in repository snapshot |
| S29_EVAL_RMSD | source code | `mattergen/evaluation/utils/metrics_structure_summary.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | official evaluator implementation in repository snapshot |
| S30_RMSD_UTIL | source code | `mattergen/evaluation/utils/utils.py` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | official evaluator implementation in repository snapshot |
| S31_BASE_CONDITION_CONFIG | config | `configs/q3_e3_pcr_frozen64.json` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | frozen evaluation config |

## 18. 当前资料不支持的内容

- 真实dft_mag_density命中率：NOT_SUPPORTED_BY_CURRENT_REPOSITORY
- DFT能量/力/声子/动力学稳定性：NOT_SUPPORTED_BY_CURRENT_REPOSITORY
- 实验可合成性：NOT_SUPPORTED_BY_CURRENT_REPOSITORY

## 19. 网页ChatGPT写作注意事项

先定义材料逆向生成任务边界，再介绍条件扩散方法和MatterGen基线归属，随后给出目标条件、实验协议与评价体系。所有Stable/E-hull/RMSD旁保留surrogate限定。最终章节编号以`thesis/CHAPTER_NUMBERING_FINAL.md`为准。

## 20. 写完后应由Codex核查的项目

- 术语C0/A0/E3-A/E3-G一致
- 学校中文题目逐字一致，英文题目标记PROVISIONAL
- MatterGen明确归属于预训练基线，未被隐藏或包装成本文创新
- 每个seed范围和n一致
- 代理/DFT/属性边界显式
- Mixed 256资格正确

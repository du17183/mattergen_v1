# 第5章证据包：Learned-Gated E3-PCR后生成精修方法

> 本文件是写作证据，不是完整论文正文。任何项目事实必须回指 source_id；未支持内容不得由通用知识补齐。

## 1. 本章研究目标

恢复选择Gate、安全有界等变位置精修、正式三臂实验和Always-on/Random Gate消融证据。

## 2. 本章回答的核心问题

- 14维风险特征和129参数Gate如何构造？
- Gate、Refiner和Fallback各自承担什么功能？
- 如何限制位置更新并拒绝不安全/升能proposal？
- 平均降力和harm控制之间如何权衡？

## 3. 建议二级和三级标题

- 5.1 生成后局部物理不一致问题
- 5.2 方法总体框架
- 5.3 14维风险特征
- 5.4 129参数Learned Gate
- 5.5 等变位置更新
- 5.6 Trust region与位移限制
- 5.7 Backtracking与安全检查
- 5.8 Exact fallback
- 5.9 正式256-seed实验
- 5.10 Always-on与Random Gate消融
- 5.11 机制分析与局限

## 4. 与前后章节的关系

第3章定义C0与评价；本章建立可连接C0或A0的独立后生成模块；第6章验证与A0组合。

## 5. 可使用的源码事实

- 14特征依次为num_atoms、volume_per_atom、mass_density、minimum_distance、atomic_number_mean/std、cell_condition、CHGNet energy/atom、force RMS/max/mean、stress RMS/maxabs和mag density。
- Gate为StandardScaler+MLPClassifier，14→8→1，tanh隐藏层，129个神经网络参数，阈值0.5。
- 训练样本为A0 seeds 20000–20063；标签为5步位置精修后最大力是否低于基线；8折OOF只作开发诊断。
- Gate只判断是否执行；E3-PCR用CHGNet force-vector方向执行最多5步；Fallback返回原始结构。
- 每步eta=.01、每原子0.02 Å cap，回溯scale为1、1/2、1/4；候选需finite、volume>0.1、min distance>=0.5 Å且CHGNet energy不升。
- 原子种类和晶格不变；最终wrapped累计最大位移检查<=0.10 Å；Gate-off和全拒绝exact fallback。
- 推理不训练MatterGen或CHGNet，且不改变原始MatterGen采样轨迹。

## 6. 可使用的配置和参数

- features=14
- hidden=8
- output=1
- parameters=129
- threshold=.5
- steps=5
- eta=.01
- step cap=.02 Å
- cumulative cap=.10 Å
- backtracks=3
- min distance=.5 Å

## 7. 公式与变量定义

| ID | 公式 | 性质 | 代码 |
| --- | --- | --- | --- |
| F3_HARM | $\mathrm{Harm}=\mathbb 1[F_{\max}^{selected}-F_{\max}^{base}>10^{-6}]$ | exact | `research/q3_formal256.py::FORCE_HARM_EPSILON and gate mechanism analysis` |
| F5_STANDARDIZE | $z_j=(x_j-\mu_j)/\sigma_j$ | interpreted | `research/postgen_fastgate/refiner_eval.py::build_network / StandardScaler` |
| F5_GATE_NETWORK | $h=\tanh(W_1z+b_1),\qquad c=\sigma(W_2h+b_2)$ | interpreted | `research/postgen_fastgate/refiner_eval.py::build_network / MLPClassifier` |
| F5_PARAMETER_COUNT | $14\times8+8+8\times1+1=129$ | exact | `research/postgen_fastgate/refiner_eval.py::train_gate network trainable_parameters` |
| F5_GATE_RULE | $a=\mathbb 1[c\ge 0.5]$ | exact | `research/q3_frozen64.py::refine` |
| F5_POSITION_PROPOSAL | $\Delta x_i^{(b)}=\operatorname{clipnorm}(\eta\,2^{-b}F_i,\ R_{step}2^{-b})$ | exact | `research/postgen_fastgate/refiner_eval.py::position_proposal and advance` |
| F5_ACCEPTANCE | $\mathrm{accept}\iff \mathrm{finite\_safe}(X')\land E_{\mathrm{CHGNet}}(X')\le E_{\mathrm{CHGNet}}(X)+10^{-7}$ | exact | `research/postgen_fastgate/refiner_eval.py::finite_safe and advance` |
| F5_TRUST_BOUND | $\max_i\lVert x_i^{final}-x_i^{input}\rVert_{MIC}\le 5\times0.02=0.10\;\AA$ | interpreted | `research/q3_frozen64.py::run_refinement_subset and refine postcondition` |

公式的完整变量、exact/interpreted资格见 `../FORMULA_REGISTRY.md`。

## 8. 实验设计

- 正式C0/E3-A/E3-G三臂严格配对：40000–40255，n=256，和Gate训练交集为0。
- C0每seed只生成一次；E3-A和E3-G从同一C0派生；MatterSim 768/768。
- 正式主端点为预松弛最大力；20,000 paired bootstrap；Wilcoxon Pratt；两主臂Holm校正。
- Random Gate仅为frozen64补充消融：5个随机重复，每次42/64开启（65.625%），不是formal256。

## 9. 正式结果

- E3-G最大力0.342964→0.263107 eV/Å，-23.28%；CI [-0.144966,-0.032453]；Holm p=4.19e-10；raw W/T/L=163/0/93。
- RMSD 0.049390→0.045937 Å；E-hull基本不变；Stable/NUS/Novel/Unique保持。
- E3-A平均最大力-28.87%，大于E3-G的-23.28%。
- E3-G coverage 66.406%，harm 18.359%，low-force harm 17.969%；E3-A分别100%、25.391%、29.688%。
- E3-G保留80.657% Always-on平均降力收益；harm McNemar p=.000534。
- Random Gate frozen64五次平均相对变化-21.42%，范围[-30.00%,-13.05%]；Learned Gate frozen64为-33.56%，但该比较不是formal256主结论。

## 10. 对应图表

- Figure 3
- Figure 6
- Figure 7
- Figure 8

## 11. 对应表格

- Table 03
- Table 04

## 12. 允许写入正文的结论

- 独立formal256支持显著预松弛最大力下降。
- Learned Gate以较少覆盖降低总体和低力harm。
- 位置更新安全有界且元素/晶胞保持。

冻结claim：

- 在40000–40255的独立256个样本中，E3-G把预松弛最大力均值从0.342964降至0.263107 eV/Å，相对下降23.28%，配对均值差95% CI为[-0.144966,-0.032453]，Holm校正p=4.19e-10。
- 相对Always-on，Learned Gate把覆盖率从100%降至66.406%，harm从25.391%降至18.359%，低力子集harm从29.688%降至17.969%，并保留80.657%的平均降力收益；harm差异McNemar p=0.000534。

## 13. 禁止夸大的结论

- Learned Gate平均降力优于Always-on。
- Gate保证所有结构改善。
- E3-PCR是完整晶格/组成松弛器。
- CHGNet输出是真实磁性或DFT验证。

## 14. 必须主动说明的限制

- E3-G仍存在harm样本。
- Gate仅64个训练结构、129参数，对训练重叠敏感。
- 只更新位置，不能修复组成或晶格错误。
- CHGNet是辅助代理，正式评价仍为MatterSim。
- Random Gate来自frozen64而非formal256。

## 15. 对应源码文件和函数

- FEATURE_COLUMNS
- build_network
- historical_training_data
- train_gate
- position_proposal
- finite_safe
- advance
- run_refinement_subset
- refine
- force_robustness

## 16. 对应数据文件和字段

- S16_E3_CONFIG
- S17_I2_DATA
- S18_I2_REPORT
- S19_GATE_MECHANISM
- S20_RANDOM_GATE

字段定义必须联合使用 `S03_DATA_DICTIONARY`、本章专用数据文件和 `metrics_definitions.md`/`experiment_evidence.md`。

## 17. 对应commit、分支和报告

- `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483`
- `b65f42a8792004c7c820e59fa4413e1310e06143`
- `5293b4b71be88b6663bbe349f3b57694a916835f`

| source_id | 类型 | 路径 | commit | 数据资格 |
| --- | --- | --- | --- | --- |
| S01_MANIFEST | manifest | `thesis_archive/FINAL_EXPERIMENT_MANIFEST.json` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative archive manifest |
| S02_CLAIMS | report | `thesis/PAPER_CLAIMS_FINAL.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative writing claim register |
| S03_DATA_DICTIONARY | report | `thesis_archive/DATA_DICTIONARY.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative archive schema |
| S04_LINEAGE | report | `thesis_archive/EXPERIMENT_LINEAGE.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative lineage narrative |
| S05_LIMITATIONS | report | `thesis/PAPER_LIMITATIONS.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | mandatory disclosure |
| S13_E3_REFINER | source code | `research/postgen_fastgate/refiner_eval.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | available at formal snapshot; frozen source identity b65f42a8792004c7c820e59fa4413e1310e06143 |
| S14_E3_FROZEN_CORE | source code | `research/q3_frozen64.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | frozen core reused by formal runner |
| S15_E3_FORMAL_RUNNER | source code | `research/q3_formal256.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal run snapshot; recorded code commit 5293b4b71be88b6663bbe349f3b57694a916835f |
| S16_E3_CONFIG | config | `thesis_archive/configs/e3_pcr_final.yaml` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | frozen configuration |
| S17_I2_DATA | per-seed data | `thesis_archive/data/innovation2/per_seed_metrics.csv` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal independent data; seeds 40000–40255 |
| S18_I2_REPORT | report | `thesis_archive/reports/innovation2/final_report.md` | `41479015c5c3edc389601c4b7cc44a6db5e115cd` | frozen formal report |
| S19_GATE_MECHANISM | summary JSON | `reports/q3_e3_pcr/formal256/gate_mechanism_summary.json` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal mechanism evidence |
| S20_RANDOM_GATE | table | `reports/q3_e3_pcr/frozen64/random_gate_ablation.csv` | `b65f42a8792004c7c820e59fa4413e1310e06143` | frozen64 supplementary ablation, not formal256 |
| S31_BASE_CONDITION_CONFIG | config | `configs/q3_e3_pcr_frozen64.json` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | frozen evaluation config |

## 18. 当前资料不支持的内容

- 外部材料体系泛化：NOT_SUPPORTED_BY_CURRENT_REPOSITORY
- 真实DFT力下降：NOT_SUPPORTED_BY_CURRENT_REPOSITORY
- Gate概率严格校准：NOT_SUPPORTED_BY_CURRENT_REPOSITORY

## 19. 网页ChatGPT写作注意事项

把Gate和Refiner分开写；先报告E3-G主效果，再诚实报告Always-on更强平均降力和Gate的risk–coverage价值。

## 20. 写完后应由Codex核查的项目

- 14特征顺序完整
- 129参数计算正确
- CHGNet与MatterSim角色区分
- raw与algorithmic W/T/L不混用
- Random Gate标注frozen64

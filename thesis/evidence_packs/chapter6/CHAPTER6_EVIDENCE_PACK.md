# 第6章证据包：组合验证、消融、负面结果与讨论

> 本文件是写作证据，不是完整论文正文。任何项目事实必须回指 source_id；未支持内容不得由通用知识补齐。

## 1. 本章研究目标

并列呈现两次独立组合验证、Gate风险消融、泄漏诊断、代表性No-Go和证据边界。

## 2. 本章回答的核心问题

- Adaptive CFG与E3-PCR能否串联且各自保持功能身份？
- 两组独立cohort是否复现同方向效果，效应是否异质？
- 训练重叠主要影响平均效果还是安全估计？
- 失败路线揭示了哪些速度—质量、在线—离线和学习方向边界？

## 3. 建议二级和三级标题

- 6.1 两个创新点的功能分工
- 6.2 组合验证设计
- 6.3 独立兼容性实验一
- 6.4 独立复现实验二
- 6.5 Gate消融与风险分析
- 6.6 训练—测试泄漏诊断
- 6.7 代表性No-Go路线
- 6.8 计算开销
- 6.9 真实性与可复现性
- 6.10 局限性讨论

## 4. 与前后章节的关系

综合第4章采样模块与第5章后生成模块；为总结章节提供可重复性、失败边界和限制。

## 5. 可使用的源码事实

- Adaptive CFG是完整组合方法的共享上游采样模块；E3-PCR是可接C0或A0的独立后生成模块。
- 两个64-seed cohort分别预留并独立报告；不得事后pool为单个预注册128。
- Gate-off为结构级exact fallback；评价数值可能有<1e-6微差，因此报告需区分raw numeric与algorithmic counts。
- 训练重叠诊断故意包含20000–20063；整个Mixed 256无独立资格。
- 代表性No-Go不是创新贡献，而是停止证据和方法边界。

## 6. 可使用的配置和参数

- cohort1=41000–41063
- cohort2=50000–50063
- harm epsilon=1e-6
- leak overlap=20000–20063
- leak held-out=20064–20255

## 7. 公式与变量定义

| ID | 公式 | 性质 | 代码 |
| --- | --- | --- | --- |
| F3_MAX_FORCE | $F_{\max}=\max_i\lVert\mathbf F_i\rVert_2$ | exact | `research/q3_frozen64.py::relax worker pre_relax_max_force_ev_ang` |
| F3_HARM | $\mathrm{Harm}=\mathbb 1[F_{\max}^{selected}-F_{\max}^{base}>10^{-6}]$ | exact | `research/q3_formal256.py::FORCE_HARM_EPSILON and gate mechanism analysis` |
| F5_GATE_RULE | $a=\mathbb 1[c\ge 0.5]$ | exact | `research/q3_frozen64.py::refine` |
| F5_ACCEPTANCE | $\mathrm{accept}\iff \mathrm{finite\_safe}(X')\land E_{\mathrm{CHGNet}}(X')\le E_{\mathrm{CHGNet}}(X)+10^{-7}$ | exact | `research/postgen_fastgate/refiner_eval.py::finite_safe and advance` |

公式的完整变量、exact/interpreted资格见 `../FORMULA_REGISTRY.md`。

## 8. 实验设计

- Cohort 1：A0/A0+E3-G配对64；generation 64/64；relaxation 128/128。
- Cohort 2：A0/A0+E3-G配对64；全新seeds；generation 64/64；relaxation 128/128。
- Leakage：training overlap 64与held-out 192；single-sided Fisher exact。
- Corrector Gating正式256；RP-QTFG Gate 1八样本；CG-TDR V2八样本；其他路线证据等级见negative-results summary。

## 9. 正式结果

- Cohort 1最大力0.217302→0.158416，-27.10%；CI [-0.092341,-0.029754]；p=7.74e-5；raw 45/0/19，algorithmic 34/19/11。
- Cohort 2最大力0.265280→0.214830，-19.02%；CI [-0.102213,-0.010696]；p=.000587；algorithmic 35/18/11。
- 两组方向一致但效应大小不同，不能声称固定幅度。
- Leakage overlap harm=0/64，held-out=31/192，Fisher p=6.87e-5；安全性明显被高估。
- Corrector Gating约1.506×，但E-hull +0.0224、Stable -9.77 pp、NUS -9.38 pp。
- RP-QTFG离线方向正向但在线RMSD系统恶化，延迟约+30%–49%。
- CG-TDR Gate可学但Teacher residual方向未可靠泛化，收益接近零或RMSD恶化。

## 10. 对应图表

- Figure 1
- Figure 4
- Figure 7
- Figure 9
- Figure 10
- Figure 11
- Figure 12

## 11. 对应表格

- Table 05
- Table 06
- Table 07
- Table 08
- Table 09

## 12. 允许写入正文的结论

- 两个完全独立cohort均复现正向降力方向。
- 效应大小存在cohort异质性。
- 泄漏显著高估Gate安全性。
- No-Go可用于讨论假设边界。

冻结claim：

- 第一组独立组合cohort（41000–41063，n=64）中，A0+E3-G把预松弛最大力从0.217302降至0.158416 eV/Å，相对下降27.10%，95% CI为[-0.092341,-0.029754]，p=7.74e-5。
- 第二组完全独立cohort（50000–50063，n=64）中，A0+E3-G把预松弛最大力从0.265280降至0.214830 eV/Å，相对下降19.02%，95% CI为[-0.102213,-0.010696]，p=0.000587；算法语义W/T/L=35/18/11。
- 训练重叠没有明显夸大平均最大力改善，但显著高估Gate安全性：overlap harm=0/64，held-out harm=31/192=16.15%，单侧Fisher p=6.87e-5。

## 13. 禁止夸大的结论

- 预注册128-seed pooled实验。
- 只报告cohort 1。
- Mixed 256独立验证。
- No-Go路线包装成正向创新。
- 创新点一是所有历史分支公共代码。

## 14. 必须主动说明的限制

- 两个组合cohort各n=64。
- 同一数据域与MatterSim评价器。
- 部分历史No-Go原始报告只留服务器/历史分支，归档仅完整保留总结。
- 计算开销证据对E3-PCR组合以小样本/单环境为主，不宜外推部署成本。

## 15. 对应源码文件和函数

- a0_e3g_compat64 analysis
- a0_e3g_independent64 force_outcome_counts
- leakage statistics Fisher exact
- q3_formal256 mechanism

## 16. 对应数据文件和字段

- S21_COHORT1_DATA
- S22_COHORT1_REPORT
- S23_COHORT2_DATA
- S24_COHORT2_REPORT
- S25_LEAK_DATA
- S26_LEAK_REPORT
- S32_NEGATIVE_RESULTS

字段定义必须联合使用 `S03_DATA_DICTIONARY`、本章专用数据文件和 `metrics_definitions.md`/`experiment_evidence.md`。

## 17. 对应commit、分支和报告

- `ba2303c284210fdae0a35bb0153a8ef3af45a54c`
- `22e1db74a59476562f1f746cd4210b9420cbdf05`
- `01e9b2c30e5c58e05eaae908ba291c518b977d03`

| source_id | 类型 | 路径 | commit | 数据资格 |
| --- | --- | --- | --- | --- |
| S01_MANIFEST | manifest | `thesis_archive/FINAL_EXPERIMENT_MANIFEST.json` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative archive manifest |
| S02_CLAIMS | report | `thesis/PAPER_CLAIMS_FINAL.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative writing claim register |
| S03_DATA_DICTIONARY | report | `thesis_archive/DATA_DICTIONARY.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative archive schema |
| S04_LINEAGE | report | `thesis_archive/EXPERIMENT_LINEAGE.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative lineage narrative |
| S05_LIMITATIONS | report | `thesis/PAPER_LIMITATIONS.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | mandatory disclosure |
| S11_I1_REPORT | report | `thesis_archive/reports/innovation1/formal_final_report.json` | `20255f1a857cd763a7ef2bf2f24c1889c98c4d1c` | frozen formal report |
| S14_E3_FROZEN_CORE | source code | `research/q3_frozen64.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | frozen core reused by formal runner |
| S15_E3_FORMAL_RUNNER | source code | `research/q3_formal256.py` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal run snapshot; recorded code commit 5293b4b71be88b6663bbe349f3b57694a916835f |
| S19_GATE_MECHANISM | summary JSON | `reports/q3_e3_pcr/formal256/gate_mechanism_summary.json` | `0275cbf08ed3c6321cea7d06f7a3a8edb83b7483` | formal mechanism evidence |
| S20_RANDOM_GATE | table | `reports/q3_e3_pcr/frozen64/random_gate_ablation.csv` | `b65f42a8792004c7c820e59fa4413e1310e06143` | frozen64 supplementary ablation, not formal256 |
| S21_COHORT1_DATA | per-seed data | `thesis_archive/data/compatibility_1/per_seed_metrics.csv` | `ba2303c284210fdae0a35bb0153a8ef3af45a54c` | independent; seeds 41000–41063 |
| S22_COHORT1_REPORT | report | `thesis_archive/reports/compatibility/final_report.md` | `e358ee39a8cdd2a061a18bfaddbe88316b455048` | independent compatibility report |
| S23_COHORT2_DATA | per-seed data | `thesis_archive/data/compatibility_2/per_seed_metrics.csv` | `22e1db74a59476562f1f746cd4210b9420cbdf05` | fully independent; seeds 50000–50063 |
| S24_COHORT2_REPORT | report | `thesis_archive/reports/replication/final_report.md` | `85485bc956fce1cf7d01c55baaa92c0b69fd745e` | independent replication report |
| S25_LEAK_DATA | per-seed data | `thesis_archive/data/leakage_diagnostic/per_seed_metrics.csv` | `01e9b2c30e5c58e05eaae908ba291c518b977d03` | diagnostic only; mixed 256 invalid for independent claims |
| S26_LEAK_REPORT | report | `thesis_archive/reports/leakage_diagnostic/final_report.md` | `d5bf7d00ab51a2a0b319203443391e3463e7a91b` | diagnostic report; held-out 192 supplementary only |
| S32_NEGATIVE_RESULTS | report | `docs/experiments/negative_results_summary.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | archive-level synthesis; some original server reports not in GitHub |

## 18. 当前资料不支持的内容

- 两个cohort统一pooled效应：NOT_SUPPORTED_BY_CURRENT_REPOSITORY
- 跨材料体系泛化：NOT_SUPPORTED_BY_CURRENT_REPOSITORY
- 部分No-Go完整原始日志：NOT_FULLY_RECOVERED_FROM_ARCHIVE

## 19. 网页ChatGPT写作注意事项

两个cohort各自成节后讨论异质性；泄漏诊断写成可信性审计；No-Go按假设—观察—停止证据—认识组织。

## 20. 写完后应由Codex核查的项目

- 两cohort不pool
- raw与algorithmic计数标注
- Mixed资格正确
- No-Go来源恢复边界明确
- MatterSim/DFT边界保留

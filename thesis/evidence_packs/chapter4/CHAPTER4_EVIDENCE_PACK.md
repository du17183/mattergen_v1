# 第4章证据包：多字段残差驱动的在线Adaptive CFG

> 本文件是写作证据，不是完整论文正文。任何项目事实必须回指 source_id；未支持内容不得由通用知识补齐。

## 1. 本章研究目标

从正式commit恢复Adaptive CFG的精确公式、伪代码、集成边界和256-seed结果。

## 2. 本章回答的核心问题

- 如何把cell、pos、atomic_numbers三种不同形状残差变成稳定控制信号？
- EMA与guidance scale如何按predictor/corrector阶段更新？
- 方法是否跳过任何采样步骤或改变物理forward数量？
- 正式结果支持何种强度的结论？

## 3. 建议二级和三级标题

- 4.1 固定CFG的局限
- 4.2 条件与无条件分支
- 4.3 三字段残差定义
- 4.4 EMA残差状态
- 4.5 在线Guidance更新
- 4.6 完整算法流程
- 4.7 计算开销
- 4.8 正式实验结果
- 4.9 讨论与限制

## 4. 与前后章节的关系

以第3章C0和评价口径为基础，输出A0；A0随后作为第6章完整组合方法的上游。

## 5. 可使用的源码事实

- conditional与unconditional输入先collate为一次joint model forward，再拆分score。
- cell、pos、atomic_numbers残差分别计算RMS，只有在标量化后才求算术平均。
- predictor和corrector各自维护EMA；首个观测直接初始化EMA。
- 当前实现产生一个全局guidance scale，三个字段共享，不是三套独立scale。
- invalid residual/EMA触发stage guidance fallback。
- Adaptive CFG不启用cfg acceleration或Corrector Gating；完整corrector和predictor流程保留。
- 控制器增加三字段RMS归约和常数级标量运算，但不减少或增加MatterGen模型forward次数。

## 6. 可使用的配置和参数

- g0=2.0
- alpha=0.50
- beta=0.95
- epsilon=1e-6
- multiplier clip=[0.25,4]
- guidance clip=[0,5]

## 7. 公式与变量定义

| ID | 公式 | 性质 | 代码 |
| --- | --- | --- | --- |
| F4_RESIDUAL | $r_{t,k}=s^{cond}_{t,k}-s^{uncond}_{t,k}$ | exact | `mattergen/diffusion/sampling/classifier_free_guidance.py::score_residual_rms` |
| F4_FIELD_RMS | $\delta_{t,k}=\sqrt{\operatorname{mean}(r_{t,k}^{\,2})}$ | exact | `mattergen/diffusion/sampling/classifier_free_guidance.py::score_residual_rms` |
| F4_FIELD_MEAN | $\delta_t=\frac{1}{|\mathcal K_t|}\sum_{k\in\mathcal K_t}\delta_{t,k}$ | exact | `mattergen/diffusion/sampling/guidance_schedule.py::_mean_valid_deltas` |
| F4_EMA | $m_{t,p}=\begin{cases}\delta_t,&m_{t-1,p}\ \mathrm{unset}\\ \beta m_{t-1,p}+(1-\beta)\delta_t,&\mathrm{otherwise}\end{cases}$ | exact | `mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` |
| F4_RATIO | $q_t=\frac{\delta_t}{m_{t,p}+\epsilon}$ | exact | `mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` |
| F4_MULTIPLIER | $u_t=\operatorname{clip}\!\left(1+\alpha(q_t-1),0.25,4\right)$ | exact | `mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` |
| F4_GUIDANCE | $g_t=\operatorname{clip}(g_0u_t,g_{\min},g_{\max})$ | exact | `mattergen/diffusion/sampling/guidance_schedule.py::GuidanceController.evaluate` |
| F4_CFG_FUSION | $s_t^{CFG}=s_t^{uncond}+g_t(s_t^{cond}-s_t^{uncond})$ | exact | `mattergen/diffusion/sampling/classifier_free_guidance.py::GuidedPredictorCorrector._score_fn_unaccelerated` |

公式的完整变量、exact/interpreted资格见 `../FORMULA_REGISTRY.md`。

## 8. 实验设计

- C0 vs A0严格配对，seeds 20000–20255，n=256。
- 每个方法generation与MatterSim relaxation均256/256成功，initial-state配对通过，Determinism Level 1。
- 未根据正式结果重新调参。

## 9. 正式结果

- E-hull C0=0.143667，A0=0.140232，差=-0.003435 eV/atom；CI跨0，p=.357。
- Stable C0=41.016%，A0=46.875%，差=+5.859 pp；CI跨0，p=.146。
- NUS C0=22.266%，A0=25.781%，差=+3.516 pp；CI跨0，p=.342。
- 三项方向均正向，但均不得称统计显著。

## 10. 对应图表

- Figure 2
- Figure 5

## 11. 对应表格

- Table 02

## 12. 允许写入正文的结论

- 多字段在线反馈使三项代理指标呈总体正向趋势。
- 算法保留完整Predictor/Corrector。
- 控制器公式可标为代码精确等价。

冻结claim：

- 在20000–20255的256个配对样本中，Adaptive CFG相对C0使代理E-hull降低0.003435 eV/atom、Stable提高5.859 pp、NUS提高3.516 pp；总体方向正向，但三项配对统计均未达到显著性。

## 13. 禁止夸大的结论

- Adaptive CFG统计显著提升。
- 该方法通过跳步或Corrector Gating加速。
- 三个字段各使用独立guidance scale。
- 代理结果证明真实磁稳定性。

## 14. 必须主动说明的限制

- 配对统计未显著。
- 单checkpoint、单目标和单采样配置。
- 没有独立属性命中验证或DFT。
- 精确控制开销没有作为正式主效果冻结。

## 15. 对应源码文件和函数

- GuidanceController
- _mean_valid_deltas
- score_residual_rms
- GuidedPredictorCorrector._score_fn_unaccelerated
- PredictorCorrector.denoise

## 16. 对应数据文件和字段

- S09_ADAPTIVE_CONFIG
- S10_I1_DATA
- S11_I1_REPORT

字段定义必须联合使用 `S03_DATA_DICTIONARY`、本章专用数据文件和 `metrics_definitions.md`/`experiment_evidence.md`。

## 17. 对应commit、分支和报告

- `5de00419eea2d8a9be303638f2db8ece15a22366`

| source_id | 类型 | 路径 | commit | 数据资格 |
| --- | --- | --- | --- | --- |
| S01_MANIFEST | manifest | `thesis_archive/FINAL_EXPERIMENT_MANIFEST.json` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative archive manifest |
| S02_CLAIMS | report | `thesis/PAPER_CLAIMS_FINAL.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative writing claim register |
| S04_LINEAGE | report | `thesis_archive/EXPERIMENT_LINEAGE.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | authoritative lineage narrative |
| S05_LIMITATIONS | report | `thesis/PAPER_LIMITATIONS.md` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | mandatory disclosure |
| S06_ADAPTIVE_CONTROLLER | source code | `mattergen/diffusion/sampling/guidance_schedule.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | formal implementation snapshot |
| S07_ADAPTIVE_CFG | source code | `mattergen/diffusion/sampling/classifier_free_guidance.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | formal implementation snapshot |
| S08_PC_SAMPLER | source code | `mattergen/diffusion/sampling/pc_sampler.py` | `5de00419eea2d8a9be303638f2db8ece15a22366` | formal implementation snapshot |
| S09_ADAPTIVE_CONFIG | config | `thesis_archive/configs/adaptive_cfg_final.yaml` | `5de00419eea2d8a9be303638f2db8ece15a22366` | frozen configuration |
| S10_I1_DATA | per-seed data | `thesis_archive/data/innovation1/per_seed_metrics.csv` | `5de00419eea2d8a9be303638f2db8ece15a22366` | formal independent data; seeds 20000–20255 |
| S11_I1_REPORT | report | `thesis_archive/reports/innovation1/formal_final_report.json` | `20255f1a857cd763a7ef2bf2f24c1889c98c4d1c` | frozen formal report |
| S12_I1_FIGURE | figure | `thesis/figures/generated/pdf/fig05_adaptive_cfg_results.pdf` | `a7d778265103cd5b547ddc07c1db4083c75513fc` | generated only from archived data |

## 18. 当前资料不支持的内容

- 其他条件字段/采样步数的泛化：NOT_SUPPORTED_BY_CURRENT_REPOSITORY
- 真实属性命中：NOT_SUPPORTED_BY_CURRENT_REPOSITORY

## 19. 网页ChatGPT写作注意事项

先写固定scale问题，再按residual→RMS→mean→phase EMA→ratio→multiplier→shared scale→lerp展开；主结果紧跟非显著性。

## 20. 写完后应由Codex核查的项目

- 公式与GuidanceController一致
- 首EMA初始化分支写明
- 全局scale而非field-wise scale
- 未混入Corrector Gating

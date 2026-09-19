# Chapter 2 Reference Map

本文件记录第2章理论概念、外部原始来源和本项目实现口径之间的对应关系。它是理论章 reference map，不承担第3—5章实验结论的 evidence map 职能。`VERIFIED` 表示书目信息、理论表述或实现位置已经核对；本章不以外部论文替代本文冻结代码对指标口径的定义。

| Section | Concept | Claim | Primary source | Secondary source if any | Used equation | Related Chapter3/4 | Citation status |
|---|---|---|---|---|---|---|---|
| 2.1 | 材料逆向设计 | 从目标性质生成候选结构不同于结构到性质的正向预测 | Zeni et al., *Nature* 2025, DOI: [10.1038/s41586-025-08628-5](https://doi.org/10.1038/s41586-025-08628-5) | MatterGen arXiv: [2312.03687](https://arxiv.org/abs/2312.03687) | — | 第3章属性条件；第4章结构反馈 | VERIFIED |
| 2.2.1 | 晶体三元表示 | 晶体由原子种类、周期分数坐标和晶胞联合表示 | Zeni et al., 2025, Appendix A.1 | `mattergen/common/data/dataset.py`; `mattergen/denoiser.py` | (2-1) | 第3章三字段；第4章 \(z_t=(a_t,p_t,H_t)\) 及位置状态 \(f_t\) | VERIFIED |
| 2.2.1 | 行向量坐标约定 | 本项目实现采用 \(r_i=f_iH\) | `thesis_release/innovation2/method/late_force_sampler.py`, `cartesian_to_fractional_correction` | `thesis_release/innovation2/method/test_mechanisms.py`; `thesis/chapter4/chapter4_final_draft.md` | (2-2) | 第4章4.3.3 | VERIFIED |
| 2.2.2 | 周期等价 | 分数坐标在整数晶格平移下等价并需 wrap | Zeni et al., 2025, Appendix A.1/A.6 | `mattergen/diffusion/wrapped/wrapped_sde.py` | (2-3), (2-4) | 第4章周期坐标映射和安全预演 | VERIFIED |
| 2.3.1 | DDPM 前向扩散 | 离散高斯前向过程及任意时刻闭式边缘分布用于建立扩散直觉 | Ho, Jain & Abbeel, NeurIPS 2020 | Zeni et al., 2025, Appendix A.3 | (2-5), (2-6) | 第3章/第4章扩散背景 | VERIFIED |
| 2.3.1 | MatterGen 字段特定 corruption | 原子类别、周期坐标、晶胞使用不同 corruption，不能等同 vanilla DDPM | Zeni et al., 2025, Appendix A.4–A.7 | `mattergen/conf/lightning_module/diffusion_module/corruption/default.yaml`; `mattergen/common/diffusion/corruption.py` | (2-5), (2-6) 后说明 | 第3章三字段；第4章预测干净结构 | VERIFIED |
| 2.3.2 | 反向生成 | 从先验逐步执行参数化反向转移得到样本 | Ho, Jain & Abbeel, NeurIPS 2020 | Song et al., ICLR 2021 | (2-7) | 第3章共享前缀/后缀；第4章 predictor–corrector | VERIFIED |
| 2.3.3 | Score-SDE | 反向时间 SDE 依赖扰动边缘分布的 score | Song et al., *Score-Based Generative Modeling through SDEs*, ICLR 2021, [arXiv:2011.13456](https://arxiv.org/abs/2011.13456) | Ho et al., 2020 | (2-8), (2-9) | 第3章条件 score；第4章位置 score 注入 | VERIFIED |
| 2.3.4 | 预测干净结构 | \(\hat{x}_0(x_t,t)\) 是暂态结构估计，MatterGen 三字段需分别解析 | `thesis_release/innovation2/method/late_force_sampler.py`, `_clean_atoms` | `mattergen/common/diffusion/corruption.py`; `mattergen/diffusion/d3pm/d3pm_predictors_correctors.py` | (2-10)–(2-12) | 第4章4.2、4.3.1 | VERIFIED |
| 2.4.1 | 条件扩散 | 条件嵌入使 score 网络估计条件分布 | Zeni et al., 2025, Appendix B | `mattergen/denoiser.py`; MatterGen model card | (2-13) | 第3章目标属性条件 | VERIFIED |
| 2.4.2 | CFG | 条件与无条件预测线性组合实现无需分类器的引导 | Ho & Salimans, *Classifier-Free Diffusion Guidance*, [arXiv:2207.12598](https://arxiv.org/abs/2207.12598) | Zeni et al., 2025, Appendix B.2 | (2-14) | 第3章3.1.1及全部 CFG 策略 | VERIFIED |
| 2.4.3 | 条件—无条件字段残差 | 三字段残差描述条件对当前预测的改变，但可靠性需实验检验 | `mattergen/diffusion/sampling/classifier_free_guidance.py` | Ho & Salimans, 2022 | (2-15) | 第3章3.2–3.4 | VERIFIED |
| 2.5.1–2.5.2 | MatterGen 总体框架 | 联合细化原子类型、分数坐标和晶胞，并支持属性条件 | Zeni et al., *Nature* 2025 | MatterGen official repository: `mattergen/` | (2-1), (2-10)–(2-15) | 第3章基础模型；第4章基础采样器 | VERIFIED |
| 2.5.3 | GemNetT 去噪器 | MatterGen 适配 GemNet-dT 进行几何消息传递和多字段预测 | Zeni et al., 2025, Appendix A.8 | Gasteiger, Becker & Günnemann, *GemNet*, NeurIPS 2021; `mattergen/conf/lightning_module/diffusion_module/model/mattergen.yaml` | — | 第3章 score 输出；第4章位置/cell/atomic 输出 | VERIFIED |
| 2.6.1 | 能量与原子力 | 原子力是势能对笛卡尔坐标的负梯度 | Yang et al., *MatterSim*, [arXiv:2405.04967](https://arxiv.org/abs/2405.04967) | Deng et al., CHGNet, 2023 | (2-16) | 第4章4.3.2 | VERIFIED |
| 2.6.2 | MaxF 与 Mean Force | MaxF 是原子力范数最大值，Mean Force 是范数均值 | `thesis/chapter4/chapter4_final_draft.md`; `thesis_release/innovation2/method/late_force_sampler.py` | MatterSim inference output | (2-17) | 第4章主代理指标 | VERIFIED |
| 2.6.3 | MLIP 边界 | MLIP 以较低推理成本近似能量、力、应力，但不等于本论文执行 DFT | Yang et al., MatterSim 2024; Deng et al., CHGNet 2023 | `thesis_release/CLAIMS_AND_LIMITATIONS.md` | (2-16) | 第3章代理属性/质量；第4章代理力 | VERIFIED |
| 2.7.1 | MatterSim | 原生预测能量、力、应力；本文用于在线引导与同源代理评价 | Yang et al., MatterSim, arXiv:2405.04967 | `thesis_release/innovation2/method/late_force_sampler.py`; `thesis_release/innovation2/README.md` | (2-16), (2-17) | 第4章4.3、4.6 | VERIFIED |
| 2.7.2 | CHGNet | CHGNet 是 charge-informed MLIP；本文为属性代理和独立代理评估器 | Deng et al., *Nature Machine Intelligence* 2023, DOI: [10.1038/s42256-023-00716-3](https://doi.org/10.1038/s42256-023-00716-3) | `thesis_release/innovation1/method/prepare_confirmation.py`; `thesis_release/innovation2/method/independent_mlip.py` | — | 第3章属性误差；第4章4.7 | VERIFIED |
| 2.8.1 | Property MAE | 冻结属性代理预测与目标条件的平均绝对偏差 | `thesis_release/innovation1/method/analyze_confirmation.py`; `prepare_confirmation.py` | `thesis/chapter3/chapter3_final_draft.md` | (2-18) | 第3章主属性指标；第4章属性护栏 | VERIFIED |
| 2.8.2 | E-hull 与 Stable | E-hull 相对参考凸包；Stable 阈值为 0.1 eV/atom | `mattergen/evaluation/metrics/energy.py`; `mattergen/evaluation/utils/globals.py` | Zeni et al., 2025, Section 2.2 | (2-19), (2-20) | 第3章质量护栏；第4章稳定性护栏 | VERIFIED |
| 2.8.2 | Validity | 冻结确认管线检查有限正体积晶胞、0.5 Å 最小周期距离、有限属性与完整质量评价 | `thesis_release/innovation1/method/prepare_confirmation.py`; `analyze_confirmation.py` | `mattergen/evaluation/metrics/structure.py`（官方 structure/composition validity 的区分） | — | 第3章终点约束；第4章质量护栏 | VERIFIED |
| 2.8.2 | Novel / Unique / NUS | Novel 相对参考集，Unique 相对生成集合；NUS 为三者与 Stable 的交集 | `mattergen/evaluation/metrics/structure.py`; `mattergen/evaluation/metrics/energy.py` | `mattergen/evaluation/utils/dataset_matcher.py` | (2-21) | 第3章/第4章集合级质量指标 | VERIFIED |
| 2.8.3 | RMSD | 比较同一生成结构与代理松弛后结构，包含周期结构匹配与原子对齐 | `mattergen/evaluation/utils/metrics_structure_summary.py`; `mattergen/evaluation/utils/utils.py` | `mattergen/evaluation/utils/structure_matcher.py` | (2-22) | 第4章代理局部结构一致性 | VERIFIED |
| 2.8.4 | 配对 bootstrap 与 W/T/L | 以相同 seed 的样本对为重采样单位；核心确认连续指标使用 20 000 次 | `thesis_release/innovation1/method/analyze_confirmation.py`; `thesis_release/innovation2/method/analyze_cohort.py` | `thesis/cross_chapter/evidence_type_rules.md` | (2-23) | 第3章3.7；第4章4.6 | VERIFIED |

## Reference audit summary

| Check | Result |
|---|---|
| MatterGen Nature DOI 与 arXiv 原文已核对 | PASS |
| DDPM NeurIPS 原文已核对 | PASS |
| Score-SDE ICLR/arXiv 原文已核对 | PASS |
| CFG arXiv 原文已核对 | PASS |
| GemNet NeurIPS 原文已核对 | PASS |
| MatterSim arXiv 原文已核对 | PASS |
| CHGNet Nature Machine Intelligence 原文已核对 | PASS |
| 三字段预测干净结构与本地实现逐项核对 | PASS |
| 指标定义与冻结 evaluator/confirmation code 核对 | PASS |
| 待引用标记 | 0 |
| `DFT_VERIFIED` | false |

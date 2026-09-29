# 摘要与结论的最终表述建议

状态：**可用于下一轮 LaTeX 修订的文本方案；现行摘要和结论尚未因此改写。**论文批准题目可保持《基于深度学习的材料逆向生成》；“冻结材料生成模型的可信推理增强”是全文研究主线，而非已经验证的真实物理可靠性认证。

## 旧表述 → 新表述

| 应避免 | 推荐 |
|---|---|
| “两个独立方法分别提升生成质量” | “在冻结 MatterGen 上研究推理阶段的预算控制与外部物理反馈，两条路线独立验证、共享可审计实验原则。” |
| “Fixed-K2 显著提升生成质量/优于多采样” | “预算约束验证选择框架的 Fixed-K2 实例在 C1-128 上相对 C0 降低代理 Property MAE；独立 Best-of-2 具有竞争力，不能声称超越 Best-of-N。” |
| “后期物理引导证明结构稳定/后期最优” | “RC-NFGD 在所用机器学习势代理下改善若干局部受力指标；Timing 支持原始 Late 相对 C0 的改善，但不确立其相对 Early/Middle 的最优性。” |
| “两模块联合达到双重收益” | “两种机制分别处理属性代理误差和代理局部受力，未验证联合协同；属性—力权衡及 POST 强基线保留。” |

## 建议中文摘要（语义定稿；写入 TeX 时注意转义百分号）

材料逆向设计旨在依据目标属性提出候选晶体结构。冻结的材料扩散生成模型虽然具有条件生成能力，其推理阶段仍面临额外计算如何分配、候选选择如何控制风险以及外部物理信号如何接入等问题。本文以参数冻结的 MatterGen 为基础，研究两类相互独立的推理增强机制：预算约束验证选择与神经力场反馈引导。“可信”在本文中指状态、随机性、计算成本、代理约束和负结果可审计，不等同于真实物理稳定性已获验证。

首先，提出预算约束验证选择推理框架，通过保存参考轨迹、有限候选后缀扩展、终点代理验证和不合格时回退 C0 实现可控推理；Fixed-K2 是其固定双候选实例。在 128 个新配对种子的 C1 确认实验中，代理 Property MAE 从 0.034796 降至 0.026332，相对改善 24.32%，配对改善的 95% 区间为 [0.005817,0.011391]，代价约为 C0 的 2.2 倍 MatterGen score 调用。独立 Best-of-2 在另一新 cohort 中具有竞争力，学习型 Linear-K2 未取得独立确认优势，预算扩展表明 K=2 是成本折中而非最低误差宽度。因此，贡献不在于宣称优于全部多样本采样方法，而在于参考保持、约束选择与可回退的预算接口。

其次，提出面向冻结材料扩散模型的神经力场反馈引导方法 RC-NFGD。方法根据预测干净结构的阶段可靠性，在原始后期窗口将 MatterSim 原子力经周期坐标映射注入位置 predictor，并保留有界修正与异常回退。Formal256 中，MatterSim 代理 MaxF、Mean Force 与松弛 RMSD 分别降低 29.81%、29.72% 和 14.17%；CHGNet 代理均值方向一致，但 Property MAE 轻微恶化。独立 32-seed Timing 消融中，原始 Late 的 MaxF 相对 C0 从 0.18986 降至 0.13864 eV/Å；该实验未证实 Late 优于 Early/Middle，且不同注入窗口的力计算量不等。历史同组 POST 对照在所测终态代理力上更强。

综上，两类方法分别回答冻结生成模型如何利用推理预算与如何接入局部物理反馈。本文通过独立配对实验、强基线、成本记账及负结果限定其适用范围；两方法没有已验证的联合增益，现有物理证据仅来自机器学习势代理，尚无 DFT 或实验验证。

关键词建议保持：材料逆向设计；晶体结构生成；扩散模型；预算约束推理；神经网络原子间势。若学校对摘要字数有限制，优先压缩机制细节，不删 Best-of-N/POST/无 DFT 三项边界。

## Matching English abstract draft

Inverse materials design seeks crystal structures that satisfy target properties. Even with a frozen diffusion generator, inference must decide how to spend additional computation, how to control candidate-selection risk, and how to incorporate external physical feedback. Using frozen MatterGen, this thesis studies two independently evaluated inference-time mechanisms: budget-constrained verifier-guided selection and neural-force-field feedback. “Trustworthy” here means reproducible states and randomness, explicit costs and surrogate constraints, and auditable negative results; it does not imply validated physical stability.

First, a budget-constrained verifier-guided inference framework preserves a reference trajectory, expands a limited set of candidate suffixes, evaluates their terminal surrogate properties and quality, and falls back to C0 when no candidate qualifies. Fixed-K2 is a fixed-width instance. On 128 fresh paired C1 seeds, surrogate Property MAE decreased from 0.034796 to 0.026332 (24.32%; paired-improvement 95% CI [0.005817, 0.011391]) at approximately 2.2 times the C0 MatterGen score-call budget. Independent Best-of-2 remained competitive on a separate cohort, learned Linear-K2 allocation showed no independently supported advantage, and budget scaling did not identify K=2 as the lowest-error width. The claim is a reproducible, reference-preserving selection interface, not superiority over general Best-of-N sampling.

Second, RC-NFGD introduces neural-force-field feedback into the position predictor within the original late diffusion window after stage-reliability diagnosis and periodic-coordinate mapping, with bounded correction and anomaly fallback. On Formal256, MatterSim-surrogate MaxF, mean force, and relaxation RMSD decreased by 29.81%, 29.72%, and 14.17%, respectively; CHGNet showed directionally consistent mean-force evidence, whereas Property MAE deteriorated slightly. In a separate 32-paired-seed timing ablation, the original late arm reduced MaxF from 0.18986 to 0.13864 eV/Å relative to C0, but it was not established as superior to earlier starts, and the force-computation budgets differed. An equal-force-call post-processing baseline was stronger on the measured terminal surrogate force.

The two methods address distinct inference decisions and have not demonstrated joint synergy. Their physical evidence remains limited to machine-learning interatomic-potential surrogates; no DFT or experimental validation was performed.

## 第6章结论的收束句

“本文形成的不是已验证的双模块联合最优系统，而是在冻结材料扩散生成模型上分别确认的两类推理接口：预算约束验证选择在当前代理属性任务中以额外计算换取参考可回退的候选筛选，神经力场反馈在当前代理物理协议下降低部分局部受力指标。Best-of-2、Linear-K2、POST、Pareto 与 Timing 消融共同划定了独立优势、属性代价、计算预算、阶段最优性和代理偏差的边界。真实物理有效性、跨任务迁移与联合增益仍需新的预注册验证。” 

现行中英文摘要已经写入不少上述边界；本方案主要补齐 Timing/Pareto 后的一致性，不建议为追求更“强”结论删掉不利事实。

# Abstract Source Sheet

本表是后续中文摘要、英文摘要和答辩摘要的唯一数字来源。本轮不直接撰写摘要。

## Background

- 研究对象：基于 MatterGen 的无机晶体条件逆向生成。
- 两个目标：条件属性控制；局部结构代理物理一致性。
- 基础模型：冻结 MatterGen，不重新训练主干权重。

## Innovation 1

**正式名称：** 参考轨迹保留的预算约束多分支条件引导与安全回退方法
**英文名称：** Reference-Preserved Budget-Constrained Multi-Branch Guidance with Safety-Constrained Fallback

**方法版：** 保存精确 C0 参考轨迹和共享前缀，在固定预算下展开两个候选后缀，并依据终点属性和质量护栏接受候选，否则参考回退。

**C1-128 主结果：**

- Property MAE：0.034796 → 0.026332。
- 相对降低：24.32%。
- 绝对改善 CI：[0.005817, 0.011391]。
- W/T/L：53/75/0。
- 生成预算：约 2.2× C0。

**50字版：** 保存 C0 参考轨迹并展开固定双候选，以终点约束和参考回退改善 C1-128 代理属性误差。

**100字版：** 针对在线 Adaptive CFG 难以稳定判断有利偏离的问题，提出参考轨迹保留的预算约束多分支条件引导与安全回退方法。Fixed-K2 在 C1-128 上将 Property MAE 降低24.32%，但需约2.2倍生成预算，Linear-K2 未显示额外优势。

## Innovation 2

**正式名称：** 基于阶段可靠性校准的后期神经力场引导扩散方法
**英文名称：** Reliability-Calibrated Late-stage Neural Force-Guided Diffusion (RC-NFGD)

**方法版：** 依据预测干净结构的阶段可靠性在后期注入 MatterSim 力反馈，经笛卡尔—分数坐标映射后修正 position predictor，并保留有界修正和异常回退。

**Formal256 主结果：**

- MatterSim MaxF：0.226408 → 0.158911，降低29.81%。
- Mean Force：0.098702 → 0.069373，降低29.72%。
- RMSD：0.051137 → 0.043893，降低14.17%。
- CHGNet MaxF：降低14.36%。
- Property MAE：0.009757 → 0.009868，误差恶化约1.14%。

**50字版：** 通过阶段可靠性校准在扩散后期注入神经力反馈，RC-NFGD 改善 Formal256 代理受力和松弛 RMSD。

**100字版：** 针对属性满足但结构仍可能高受力的问题，提出 RC-NFGD，在预测干净结构可靠的后期将 MatterSim 力映射为位置反馈。Formal256 的 MaxF、Mean Force 和 RMSD 分别降低29.81%、29.72%和14.17%，但 Property MAE 轻微恶化。

## Independent surrogate

- CHGNet 不参与 RC-NFGD 在线力反馈。
- CHGNet MaxF 均值降低14.36%。
- CHGNet Mean Force 均值降低9.59%。
- CHGNet 仍是 MLIP，不是 DFT 或实验验证。

## Mandatory boundaries

- Linear-K2 优于 Fixed-K2：NOT_SUPPORTED。
- 在线 Adaptive CFG 稳定有效：NOT_SUPPORTED。
- RC-NFGD Property MAE 改善：NOT_SUPPORTED。
- 在线 RC-NFGD 优于等预算 POST：NOT_SUPPORTED。
- 有界修正是性能核心：NOT_SUPPORTED。
- Joint synergy：NOT_SUPPORTED。
- DFT：DFT_VERIFIED = false。

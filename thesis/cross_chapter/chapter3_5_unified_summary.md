# 第3—5章统一摘要

## 创新点1

**问题。** 固定 CFG=2.0 的 MatterGen 参考基线 C0 并非对每个样本都达到可观察的最佳条件属性终点，但基于局部残差的在线自适应动作跨样本不稳定，错误偏离还会不可逆地改变后续轨迹。

**方法。** 提出参考轨迹保留的预算约束多分支条件引导与安全回退方法（Reference-Preserved Budget-Constrained Multi-Branch Guidance with Safety-Constrained Fallback）。固定双候选分支策略（Fixed-K2）在共享 C0 前缀后展开 GPulse 和 PPulse 两个冻结后缀，通过预定义属性与质量条件进行终点评价；没有安全且更优的候选时返回精确 C0。

**确认结果。** 在128个全新配对种子的 C1-128 上，Fixed-K2 将 Property MAE 从0.034796降至0.026332，相对改善24.32%，绝对改善95% CI为[0.005817, 0.011391]，W/T/L=53/75/0。E-hull和Stable改善，NUS上升，Validity保持100%；Novel从71.09%降至65.63%，因此不能声称所有质量指标改善。

**负结果。** 学习型 Linear-K2 在 Phase B 留出测试上只有区间跨零的方向性结果，并在 C1-128 上未超过 Fixed-K2：有利相对差−4.52%，CI [−0.003148, 0.000740]，31/62/35。`Linear-K2 > Fixed-K2 = NOT_SUPPORTED`。多轮在线 Adaptive CFG 探索同样未得到稳定支持。

**计算。** C0为每样本2 000次 MatterGen score调用，Fixed-K2为4 400次和3次终点评价，约2.2×部署计算量；Phase B约3.4×是离线数据构建成本。

**限制。** 收益依赖终点评价器和冻结接受规则；0个最终属性损失来自选择与回退设计，不代表候选从不失败；没有证明 Fixed-K2 优于所有等预算通用 Best-of-N，也没有跨属性、模型或材料域验证。

## 创新点2

**问题。** 条件属性满足不等价于生成结构处于低代理受力状态；同时，早期含噪状态上的神经势反馈缺乏可靠结构语义。

**方法。** 提出基于阶段可靠性校准的后期神经力场引导扩散方法（Reliability-Calibrated Late-stage Neural Force-Guided Diffusion，RC-NFGD）。方法用预测干净结构的阶段诊断冻结 (t\le0.02) 的后期窗口，将 MatterSim 原子力映射到周期分数坐标，只修正 position predictor，并使用有界位置修正和异常安全回退。有界设计是保守工程约束，不作为已证实的性能核心。

**确认结果。** Formal256上，MatterSim MaxF、Mean Force和RMSD分别由0.226408、0.098702、0.051137降至0.158911、0.069373、0.043893，相对降低29.81%、29.72%和14.17%，对应区间均处于有利方向。P0、Formal32和Formal256三个分离cohort方向一致。

**独立代理与负结果。** CHGNet独立代理评估器给出MaxF 14.36%和Mean Force 9.59%的均值改善，但效应较小且尾部混合。Property MAE从0.009757增至0.009868，恶化1.14%；Stable和Validity保持不变。Unbounded消融不支持bound为性能核心，等预算POST得到更低MatterSim代理力，因此`online > POST = NOT_SUPPORTED`。

**计算。** RC-NFGD不增加2 000次MatterGen score调用，另增加20次MatterSim在线调用。记录时间92.663 s与C0的92.849 s接近，但不据此主张加速。

**限制。** MatterSim主指标存在同源评价风险；CHGNet仍是代理模型；无DFT、实验合成、声子或长时分子动力学验证；当前参数与效应不能直接跨任务外推。

## 总体结论

得到支持的是：Fixed-K2在冻结C1-128代理协议下改善C0的条件属性准确性；RC-NFGD在Formal256上改善代理力与松弛RMSD，并获得CHGNet均值方向支持。

未得到支持的是：学习型Linear-K2优于Fixed-K2、在线Adaptive CFG稳定有效、有界修正是性能核心、RC-NFGD改善目标属性、在线方法优于等预算POST，以及两项创新具有已验证协同。

仍未验证的是：DFT力、实验可合成性、跨MatterGen checkpoint/属性/材料域泛化，以及Fixed-K2与RC-NFGD的直接联合确认。两项创新共同支撑“条件准确性与局部结构一致性需要分别建模和验证”的主线，而不是一个已经确认的联合最优系统。


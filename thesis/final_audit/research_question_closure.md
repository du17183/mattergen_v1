# Research Question Closure

## RQ1

**问题定义：** 在 MatterGen 固定条件引导下，如何在有限推理预算中提高目标属性生成精度，同时控制偏离参考轨迹带来的质量风险？

| 环节 | 对应内容 | 状态 |
|---|---|---|
| Chapter 1 | 固定 CFG、在线 Adaptive CFG 不稳定与终点风险 | PASS |
| Chapter 3 | 参考轨迹保留、共享前缀、Fixed-K2、终点约束与参考回退 | PASS |
| Chapter 5 | C1-128：Property MAE 0.034796→0.026332，改善24.32%；质量护栏通过；Linear-K2 未获支持 | PASS |
| Chapter 6 | 在约2.2×生成预算下，Fixed-K2 支持代理属性改善；结论不外推到普适最优 | PASS |

RESEARCH_QUESTION_1_CLOSURE = PASS

## RQ2

**问题定义：** 如何在晶体扩散生成过程中引入局部物理反馈，改善生成结构在原子力层面的代理物理一致性？

| 环节 | 对应内容 | 状态 |
|---|---|---|
| Chapter 1 | 条件属性满足不等价于局部低受力 | PASS |
| Chapter 4 | 阶段可靠性诊断、MatterSim 力、周期映射、后期 position predictor 注入 | PASS |
| Chapter 5 | Formal256：MaxF、Mean Force、RMSD 分别降低29.81%、29.72%、14.17%；CHGNet均值方向一致 | PASS |
| Chapter 6 | RC-NFGD 在冻结代理协议下得到支持，同时保留属性代价、POST边界和无DFT限制 | PASS |

RESEARCH_QUESTION_2_CLOSURE = PASS

## Overall

两个问题均已闭环，但对应两条独立推理路线；JOINT_SYNERGY = NOT_SUPPORTED，不构成统一联合模型结论。
